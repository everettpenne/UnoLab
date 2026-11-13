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
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/system_properties.h>
#include "llama.h"
#include "ggml-cpu.h"

#define TAG "UnoLabLlm"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

// What the threads are run with when no property says otherwise (see Tuning). Chosen from measurements on a Pixel 10a (Tensor G4),
// asking from Uno's search with Uno Lab in the background, where Android allows the process cores 0 to 6 (not the one fastest core):
//   4 threads, made per call (0.3.8):  reads the prompt at ~36 tok/s, writes at 4.9 to 5.1 tok/s
//   4 threads in a pool:               ~40 tok/s, 6.6 to 7.2 tok/s
//   3 threads in a pool (one per allowed fast core): ~51 tok/s, 7.7 to 10.3 tok/s
//   more threads for reading than writing (6), 2 for writing, a raised priority, no polling: no better
//   4 threads kept to "the four fastest cores": 0.8 tok/s, since one of those is not allowed and four spinning threads shared three cores
constexpr int DEFAULT_THREADS_BATCH = 0; // 0: the same as the decode threads
constexpr int DEFAULT_POOL = 1;
constexpr int DEFAULT_POLL = 50;
constexpr int DEFAULT_PRIO = 0;

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
    // Threads the context was made with, the worker pools attached to it (see applyTuning), and the tuning they were made for.
    int threads = 4;
    ggml_threadpool* pool = nullptr;
    ggml_threadpool* poolBatch = nullptr;
    std::string tuned;
};

// How the context's threads are run. Decoding reads every weight for each token, so it is bound by memory bandwidth and by its slowest
// thread: the phone's scheduler moving one of them to a little core (or a fresh set of threads being made for every token, which llama.cpp
// does when no pool is attached) costs far more than it looks. So the threads live in a pool, kept on the phone's fastest cores.
// Each value can be overridden with a system property (adb shell setprop debug.unolab.<name> <n>) to measure on a phone without a rebuild.
struct Tuning {
    int threads;      // threads for writing the answer (decode)
    int threadsBatch; // threads for reading the prompt (prefill)
    int pool;         // 0: no pool (threads made per call); 1: a pool, placed by the scheduler; 2: a pool kept on the fastest cores
    int poll;         // how hard idle pool threads spin before sleeping, 0 to 100
    int prio;         // ggml_sched_priority for the pool's threads, 0 (normal) to 3
    std::string key() const {
        char b[64]; snprintf(b, sizeof(b), "%d/%d/%d/%d/%d", threads, threadsBatch, pool, poll, prio); return b;
    }
};

int propInt(const char* name, int fallback) {
    char value[PROP_VALUE_MAX] = {0};
    if (__system_property_get(name, value) <= 0) return fallback;
    char* end = nullptr;
    long v = strtol(value, &end, 10);
    return end != value ? (int)v : fallback;
}

std::vector<int> coresFastestFirst();

// One thread for each fast core the process may run on right now, and never more than the context was made for. Android gives a
// background process fewer cores than the app in front (on a Tensor G4: three of the four fast ones), and a thread more than there are
// fast cores lands on a little core, which every step then waits for.
int fastCoresAllowed(int most) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return most;
    std::vector<int> cores = coresFastestFirst();
    if (cores.empty()) return most;
    auto hz = [](int cpu) {
        char path[96]; long v = 0;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
        if (FILE* f = fopen(path, "r")) { if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f); }
        return v;
    };
    long slowest = hz(cores.back()), fastestHz = hz(cores.front());
    int n = 0;
    for (int cpu : cores) if (CPU_ISSET(cpu, &allowed) && (fastestHz == slowest || hz(cpu) > slowest)) n++;
    return std::clamp(n, 1, most);
}

Tuning tuningFor(const Session* s) {
    Tuning t;
    t.threads = std::clamp(propInt("debug.unolab.threads", fastCoresAllowed(s->threads)), 1, 8);
    t.threadsBatch = std::clamp(propInt("debug.unolab.tbatch", DEFAULT_THREADS_BATCH > 0 ? DEFAULT_THREADS_BATCH : t.threads), 1, 8);
    t.pool = std::clamp(propInt("debug.unolab.pool", DEFAULT_POOL), 0, 2);
    t.poll = std::clamp(propInt("debug.unolab.poll", DEFAULT_POLL), 0, 100);
    t.prio = std::clamp(propInt("debug.unolab.prio", DEFAULT_PRIO), 0, 3);
    return t;
}

// The phone's cores, fastest first (by their top frequency), so a pool of n threads can be kept on the n fastest.
std::vector<int> coresFastestFirst() {
    std::vector<std::pair<long, int>> cores;
    for (int cpu = 0; cpu < 64; cpu++) {
        char path[96];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
        FILE* f = fopen(path, "r");
        if (!f) break;
        long hz = 0;
        if (fscanf(f, "%ld", &hz) != 1) hz = 0;
        fclose(f);
        cores.push_back({hz, cpu});
    }
    std::stable_sort(cores.begin(), cores.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<int> out;
    for (auto& c : cores) out.push_back(c.second);
    return out;
}

ggml_threadpool* newPool(int n, const Tuning& t, const std::vector<int>& fastest) {
    ggml_threadpool_params p = ggml_threadpool_params_default(n);
    p.poll = (uint32_t)t.poll;
    p.prio = (ggml_sched_priority)t.prio;
    if (t.pool == 2 && !fastest.empty()) {
        // Every thread may use any of the n fastest cores (not one core each: the scheduler still balances among them).
        for (int i = 0; i < n && i < (int)fastest.size(); i++) if (fastest[i] < GGML_MAX_N_THREADS) p.cpumask[fastest[i]] = true;
        p.strict_cpu = false;
    }
    return ggml_threadpool_new(&p);
}

void freePools(Session* s) {
    if (s->ctx) llama_detach_threadpool(s->ctx);
    if (s->poolBatch && s->poolBatch != s->pool) ggml_threadpool_free(s->poolBatch);
    if (s->pool) ggml_threadpool_free(s->pool);
    s->pool = s->poolBatch = nullptr;
}

// Brings the context's threads in line with the tuning, if it changed. Called on the model's thread before prompt reading and writing.
void applyTuning(Session* s) {
    Tuning t = tuningFor(s);
    std::string key = t.key();
    if (key == s->tuned) return;
    freePools(s);
    llama_set_n_threads(s->ctx, t.threads, t.threadsBatch);
    // Kept to the fastest cores (pool 2) only among those the process is allowed: naming one it is not allowed squeezed the threads
    // onto the rest.
    std::vector<int> fastest;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    bool known = sched_getaffinity(0, sizeof(allowed), &allowed) == 0;
    for (int cpu : coresFastestFirst()) if (!known || CPU_ISSET(cpu, &allowed)) fastest.push_back(cpu);
    if (t.pool > 0) {
        s->pool = newPool(t.threads, t, fastest);
        s->poolBatch = t.threadsBatch == t.threads ? s->pool : newPool(t.threadsBatch, t, fastest);
        if (s->pool && s->poolBatch) llama_attach_threadpool(s->ctx, s->pool, s->poolBatch);
    }
    s->tuned = key;
    LOGI("tuning: threads=%d batch=%d pool=%d poll=%d prio=%d", t.threads, t.threadsBatch, t.pool, t.poll, t.prio);
}

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
    s->threads = threads;
    LOGI("loaded: n_ctx=%d threads=%d", s->nCtx, threads);
    return reinterpret_cast<jlong>(s);
}

JNIEXPORT void JNICALL Java_com_everettpenne_unolab_Llm_nativeFree(JNIEnv*, jobject, jlong h) {
    auto* s = reinterpret_cast<Session*>(h);
    if (!s) return;
    freePools(s);
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
    s->threads = threads;
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
    applyTuning(s);
    double t0 = nowMs();
    for (int i = 0; i < n; i += s->nBatch) {
        int m = std::min(s->nBatch, n - i);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, m);
        if (llama_decode(s->ctx, batch) != 0) { throwJava(env, "Decoding the prompt failed"); return nullptr; }
        s->nPast += m;
    }
    double ms = nowMs() - t0;
    LOGI("prefill: %d tok in %.0f ms, %.1f tok/s", n, ms, ms > 0 ? n * 1000.0 / ms : 0.0);
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
    applyTuning(s);
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
    LOGI("decode: %d tok in %.0f ms, %.1f tok/s", produced, ms, ms > 0 ? produced * 1000.0 / ms : 0.0);
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
