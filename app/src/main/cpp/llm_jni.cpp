// The language model, through llama.cpp on the CPU. One model and one context per handle; the conversation is kept in the context and only the new
// text of each turn is decoded (Qwen3.5's Gated DeltaNet layers keep a recurrent state, so a prefix cannot be trimmed: a new chat clears it).
#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include "llama.h"

#define TAG "UnoLabLlm"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

struct Session {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    int nPast = 0;
    int nCtx = 0;
    int nBatch = 512;
    std::atomic<bool> stop{false};
    // A second context on the same weights (nativeNewContext) does not own the model; free it before the one that does.
    bool ownsModel = true;
};

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

void logToAndroid(ggml_log_level level, const char* text, void*) {
    int prio = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : level == GGML_LOG_LEVEL_WARN ? ANDROID_LOG_WARN : ANDROID_LOG_DEBUG;
    __android_log_print(prio, "llama", "%s", text);
}

void throwJava(JNIEnv* env, const std::string& message) {
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(cls, message.c_str());
}

// How many bytes at the start of [s] are whole UTF-8 characters: a token can end part-way through one (an emoji split over two tokens).
size_t completeUtf8(const std::string& s) {
    size_t i = 0, ok = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        if (i + len > s.size()) break;
        i += len;
        ok = i;
    }
    return ok;
}

std::string pieceOf(const llama_vocab* vocab, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::string big(-n, '\0');
        n = llama_token_to_piece(vocab, t, big.data(), (int)big.size(), 0, true);
        return n > 0 ? big.substr(0, n) : std::string();
    }
    return std::string(buf, n);
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_everettpenne_unolab_Llm_nativeLoad(JNIEnv* env, jobject, jstring jpath, jint nCtx, jint threads) {
    static bool inited = false;
    if (!inited) { llama_log_set(logToAndroid, nullptr); llama_backend_init(); inited = true; }
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params mp = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!model) { throwJava(env, "The model could not be loaded"); return 0; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = nCtx;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); throwJava(env, "The context could not be created"); return 0; }
    auto* s = new Session();
    s->model = model; s->ctx = ctx; s->vocab = llama_model_get_vocab(model);
    s->nCtx = (int)llama_n_ctx(ctx);
    LOGI("loaded: n_ctx=%d threads=%d", s->nCtx, threads);
    return reinterpret_cast<jlong>(s);
}

JNIEXPORT void JNICALL Java_com_everettpenne_unolab_Llm_nativeFree(JNIEnv*, jobject, jlong h) {
    auto* s = reinterpret_cast<Session*>(h);
    if (!s) return;
    llama_free(s->ctx);
    if (s->ownsModel) llama_model_free(s->model);
    delete s;
}

// Another context on [h]'s model, for a second conversation that does not disturb the first (the assistant's one-shot questions beside the chat).
JNIEXPORT jlong JNICALL Java_com_everettpenne_unolab_Llm_nativeNewContext(JNIEnv* env, jobject, jlong h, jint nCtx, jint threads) {
    auto* owner = reinterpret_cast<Session*>(h);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = nCtx; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = threads; cp.n_threads_batch = threads;
    llama_context* ctx = llama_init_from_model(owner->model, cp);
    if (!ctx) { throwJava(env, "The context could not be created"); return 0; }
    auto* s = new Session();
    s->model = owner->model; s->ctx = ctx; s->vocab = owner->vocab; s->ownsModel = false;
    s->nCtx = (int)llama_n_ctx(ctx);
    return reinterpret_cast<jlong>(s);
}

// The context's state (its one sequence), to come back to later: the recurrent layers cannot drop tokens, so a prompt's prefix is saved once
// and restored before each question instead of decoded again.
JNIEXPORT jbyteArray JNICALL Java_com_everettpenne_unolab_Llm_nativeSaveState(JNIEnv* env, jobject, jlong h) {
    auto* s = reinterpret_cast<Session*>(h);
    size_t size = llama_state_seq_get_size(s->ctx, 0);
    std::vector<uint8_t> buf(size);
    size_t n = llama_state_seq_get_data(s->ctx, buf.data(), size, 0);
    jbyteArray out = env->NewByteArray((jsize)n);
    env->SetByteArrayRegion(out, 0, (jsize)n, reinterpret_cast<const jbyte*>(buf.data()));
    return out;
}

// Puts back a state from nativeSaveState, with [nPast] tokens in it. False if llama.cpp did not take it.
JNIEXPORT jboolean JNICALL Java_com_everettpenne_unolab_Llm_nativeRestoreState(JNIEnv* env, jobject, jlong h, jbyteArray data, jint nPast) {
    auto* s = reinterpret_cast<Session*>(h);
    llama_memory_clear(llama_get_memory(s->ctx), true);
    jsize len = env->GetArrayLength(data);
    std::vector<uint8_t> buf(len);
    env->GetByteArrayRegion(data, 0, len, reinterpret_cast<jbyte*>(buf.data()));
    size_t read = llama_state_seq_set_data(s->ctx, buf.data(), buf.size(), 0);
    if (read == 0) { s->nPast = 0; return JNI_FALSE; }
    s->nPast = nPast;
    return JNI_TRUE;
}

JNIEXPORT void JNICALL Java_com_everettpenne_unolab_Llm_nativeReset(JNIEnv*, jobject, jlong h) {
    auto* s = reinterpret_cast<Session*>(h);
    llama_memory_clear(llama_get_memory(s->ctx), true);
    s->nPast = 0;
}

JNIEXPORT jint JNICALL Java_com_everettpenne_unolab_Llm_nativeUsed(JNIEnv*, jobject, jlong h) { return reinterpret_cast<Session*>(h)->nPast; }
JNIEXPORT jint JNICALL Java_com_everettpenne_unolab_Llm_nativeCtx(JNIEnv*, jobject, jlong h) { return reinterpret_cast<Session*>(h)->nCtx; }
JNIEXPORT void JNICALL Java_com_everettpenne_unolab_Llm_nativeStop(JNIEnv*, jobject, jlong h) { reinterpret_cast<Session*>(h)->stop = true; }

// Decodes [text] (special tokens such as <|im_start|> parsed) into the context. Returns {tokens, milliseconds}, or throws when it does not fit.
JNIEXPORT jdoubleArray JNICALL Java_com_everettpenne_unolab_Llm_nativePrefill(JNIEnv* env, jobject, jlong h, jstring jtext) {
    auto* s = reinterpret_cast<Session*>(h);
    const char* text = env->GetStringUTFChars(jtext, nullptr);
    int len = (int)strlen(text);
    std::vector<llama_token> tokens(len + 16);
    int n = llama_tokenize(s->vocab, text, len, tokens.data(), (int)tokens.size(), false, true);
    if (n < 0) { tokens.resize(-n); n = llama_tokenize(s->vocab, text, len, tokens.data(), (int)tokens.size(), false, true); }
    env->ReleaseStringUTFChars(jtext, text);
    if (n < 0) { throwJava(env, "Tokenizing failed"); return nullptr; }
    tokens.resize(n);
    if (s->nPast + n >= s->nCtx) { throwJava(env, "The conversation is too long for the context; start a new chat"); return nullptr; }
    double t0 = nowMs();
    for (int i = 0; i < n; i += s->nBatch) {
        int m = std::min(s->nBatch, n - i);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, m);
        if (llama_decode(s->ctx, batch) != 0) { throwJava(env, "Decoding the prompt failed"); return nullptr; }
        s->nPast += m;
    }
    double ms = nowMs() - t0;
    jdoubleArray out = env->NewDoubleArray(2);
    double v[2] = {double(n), ms};
    env->SetDoubleArrayRegion(out, 0, 2, v);
    return out;
}

// Samples up to [maxTokens] (within [grammar], a GBNF string, when it is not null), handing each complete piece of UTF-8 to sink.onPiece(byte[]) as it comes. Stops at an end-of-generation token (which is
// not added to the context: the caller adds <|im_end|> itself), at the context's end, or on nativeStop. Returns {tokens, milliseconds, endedNaturally}.
JNIEXPORT jdoubleArray JNICALL Java_com_everettpenne_unolab_Llm_nativeGenerate(JNIEnv* env, jobject, jlong h, jint maxTokens, jfloat temp, jfloat topP,
                                                                                jint topK, jstring jgrammar, jobject sink) {
    auto* s = reinterpret_cast<Session*>(h);
    s->stop = false;
    jclass sinkCls = env->GetObjectClass(sink);
    jmethodID onPiece = env->GetMethodID(sinkCls, "onPiece", "([B)V");
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // A GBNF grammar, when given, keeps the output to what it allows (the assistant's actions: one JSON object of a fixed shape).
    if (jgrammar) {
        const char* g = env->GetStringUTFChars(jgrammar, nullptr);
        llama_sampler* gs = llama_sampler_init_grammar(s->vocab, g, "root");
        env->ReleaseStringUTFChars(jgrammar, g);
        if (!gs) { llama_sampler_free(smpl); throwJava(env, "The grammar could not be read"); return nullptr; }
        llama_sampler_chain_add(smpl, gs);
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(topK));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    std::string pending;
    int produced = 0;
    bool natural = false;
    double t0 = nowMs();
    auto flush = [&](bool all) {
        size_t k = all ? pending.size() : completeUtf8(pending);
        if (k == 0) return;
        jbyteArray arr = env->NewByteArray((jsize)k);
        env->SetByteArrayRegion(arr, 0, (jsize)k, reinterpret_cast<const jbyte*>(pending.data()));
        env->CallVoidMethod(sink, onPiece, arr);
        env->DeleteLocalRef(arr);
        pending.erase(0, k);
    };
    while (produced < maxTokens && !s->stop) {
        llama_token t = llama_sampler_sample(smpl, s->ctx, -1);
        if (llama_vocab_is_eog(s->vocab, t)) { natural = true; break; }
        if (s->nPast + 1 >= s->nCtx) break;
        pending += pieceOf(s->vocab, t);
        flush(false);
        if (env->ExceptionCheck()) break;
        llama_batch batch = llama_batch_get_one(&t, 1);
        if (llama_decode(s->ctx, batch) != 0) break;
        s->nPast += 1;
        produced++;
    }
    flush(true);
    double ms = nowMs() - t0;
    llama_sampler_free(smpl);
    jdoubleArray out = env->NewDoubleArray(3);
    double v[3] = {double(produced), ms, natural ? 1.0 : 0.0};
    env->SetDoubleArrayRegion(out, 0, 3, v);
    return out;
}

JNIEXPORT jstring JNICALL Java_com_everettpenne_unolab_Llm_nativeSystemInfo(JNIEnv* env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

}  // extern "C"
