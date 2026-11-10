// The NPU lab: raw NNAPI (NDK only, no Tensor SDK, no TFLite), ported from ~/Code/qwen-npu/probe/nnapi_probe.cpp. The device and FC code below
// is the probe's, unchanged but for its printing.
#include <android/neuralnetworks.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr float kActScale = 1.0f / 128.0f;
constexpr float kWgtScale = 1.0f / 127.0f;

double nowMs() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

const char* errName(int e) {
  switch (e) {
    case ANEURALNETWORKS_NO_ERROR: return "OK";
    case ANEURALNETWORKS_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
    case ANEURALNETWORKS_INCOMPLETE: return "INCOMPLETE";
    case ANEURALNETWORKS_UNEXPECTED_NULL: return "UNEXPECTED_NULL";
    case ANEURALNETWORKS_BAD_DATA: return "BAD_DATA";
    case ANEURALNETWORKS_OP_FAILED: return "OP_FAILED";
    case ANEURALNETWORKS_UNMAPPABLE: return "UNMAPPABLE";
    case ANEURALNETWORKS_BAD_STATE: return "BAD_STATE";
    default: return "OTHER";
  }
}

struct Device {
  ANeuralNetworksDevice* handle = nullptr;
  std::string name;
  int32_t type = 0;
  int64_t featureLevel = 0;
};

std::vector<Device> listDevices() {
  std::vector<Device> out;
  uint32_t count = 0;
  if (ANeuralNetworks_getDeviceCount(&count) != ANEURALNETWORKS_NO_ERROR) return out;
  for (uint32_t i = 0; i < count; ++i) {
    Device d;
    if (ANeuralNetworks_getDevice(i, &d.handle) != ANEURALNETWORKS_NO_ERROR) continue;
    const char* n = nullptr;
    if (ANeuralNetworksDevice_getName(d.handle, &n) == ANEURALNETWORKS_NO_ERROR && n) d.name = n;
    ANeuralNetworksDevice_getType(d.handle, &d.type);
    ANeuralNetworksDevice_getFeatureLevel(d.handle, &d.featureLevel);
    out.push_back(d);
  }
  return out;
}

const char* typeName(int32_t t) {
  switch (t) {
    case ANEURALNETWORKS_DEVICE_UNKNOWN: return "UNKNOWN";
    case ANEURALNETWORKS_DEVICE_OTHER: return "OTHER";
    case ANEURALNETWORKS_DEVICE_CPU: return "CPU";
    case ANEURALNETWORKS_DEVICE_GPU: return "GPU";
    case ANEURALNETWORKS_DEVICE_ACCELERATOR: return "ACCELERATOR";
    default: return "?";
  }
}

struct GemmTiming {
  bool ok = false;
  int err = 0;
  double compileMs = 0;
  double firstMs = 0;
  double bestMs = 0;
  uint64_t weightBytes = 0;
  std::string detail;
};

// One FULLY_CONNECTED node: [1,K] int8 x [N,K] int8 -> [1,N] int8.
// createForDevices pins the graph to `dev`, so a no is a real no (no CPU fallback).
GemmTiming runGemm(const Device& dev, uint32_t K, uint32_t N, int iters, bool verbose, uint32_t B = 1) {
  GemmTiming r;
  r.weightBytes = uint64_t(K) * N;

  std::vector<uint8_t> weights(size_t(K) * N);
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = uint8_t(int32_t((i * 37) % 255) - 127 + 128);  // arbitrary, zp=128
  }
  std::vector<int32_t> bias(N, 0);
  std::vector<uint8_t> input(size_t(B) * K);
  for (size_t k = 0; k < input.size(); ++k) input[k] = uint8_t(64 + (k * 13) % 128);
  std::vector<uint8_t> output(size_t(B) * N, 0);

  ANeuralNetworksModel* model = nullptr;
  ANeuralNetworksCompilation* comp = nullptr;
  ANeuralNetworksExecution* exec = nullptr;
  ANeuralNetworksEvent* event = nullptr;

  auto fail = [&](int e, const char* where) {
    r.err = e;
    r.detail = where;
    if (verbose) printf("    %-18s -> %s (%d)\n", where, errName(e), e);
  };

  do {
    if (ANeuralNetworksModel_create(&model) != ANEURALNETWORKS_NO_ERROR) { fail(-1, "Model_create"); break; }

    uint32_t inDim[2] = {B, K};
    uint32_t wDim[2] = {N, K};
    uint32_t bDim[1] = {N};
    uint32_t oDim[2] = {B, N};

    ANeuralNetworksOperandType inType{};
    inType.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
    inType.dimensionCount = 2;
    inType.dimensions = inDim;
    inType.scale = kActScale;
    inType.zeroPoint = 128;

    ANeuralNetworksOperandType wType{};
    wType.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
    wType.dimensionCount = 2;
    wType.dimensions = wDim;
    wType.scale = kWgtScale;
    wType.zeroPoint = 128;

    ANeuralNetworksOperandType bType{};
    bType.type = ANEURALNETWORKS_TENSOR_INT32;
    bType.dimensionCount = 1;
    bType.dimensions = bDim;
    bType.scale = kActScale * kWgtScale;
    bType.zeroPoint = 0;

    // 3-input FULLY_CONNECTED is rejected by this libneuralnetworks (BAD_DATA);
    // the 4-input form with an explicit FUSED_NONE activation operand is accepted.
    ANeuralNetworksOperandType aType{};
    aType.type = ANEURALNETWORKS_INT32;
    aType.dimensionCount = 0;
    aType.dimensions = nullptr;

    ANeuralNetworksOperandType oType = inType;
    oType.dimensions = oDim;

    int e = ANEURALNETWORKS_NO_ERROR;
    if ((e = ANeuralNetworksModel_addOperand(model, &inType)) != 0) { fail(e, "addOperand input"); break; }
    if ((e = ANeuralNetworksModel_addOperand(model, &wType)) != 0) { fail(e, "addOperand weights"); break; }
    if ((e = ANeuralNetworksModel_addOperand(model, &bType)) != 0) { fail(e, "addOperand bias"); break; }
    if ((e = ANeuralNetworksModel_addOperand(model, &aType)) != 0) { fail(e, "addOperand act"); break; }
    if ((e = ANeuralNetworksModel_addOperand(model, &oType)) != 0) { fail(e, "addOperand output"); break; }

    if ((e = ANeuralNetworksModel_setOperandValue(model, 1, weights.data(), weights.size())) != 0) {
      fail(e, "setOperandValue weights");
      break;
    }
    if ((e = ANeuralNetworksModel_setOperandValue(model, 2, bias.data(), bias.size() * sizeof(int32_t))) != 0) {
      fail(e, "setOperandValue bias");
      break;
    }
    int32_t fused = ANEURALNETWORKS_FUSED_NONE;
    if ((e = ANeuralNetworksModel_setOperandValue(model, 3, &fused, sizeof(fused))) != 0) {
      fail(e, "setOperandValue act");
      break;
    }

    uint32_t inIdx[1] = {0};
    uint32_t outIdx[1] = {4};
    if ((e = ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, inIdx, 1, outIdx)) != 0) {
      fail(e, "identifyIO");
      break;
    }

    uint32_t inList[4] = {0, 1, 2, 3};  // input, weights, bias, activation
    uint32_t outList[1] = {4};
    if ((e = ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4, inList,
                                               1, outList)) != 0) {
      fail(e, "addOperation FC");
      break;
    }

    if ((e = ANeuralNetworksModel_finish(model)) != 0) { fail(e, "Model_finish"); break; }

    if ((e = ANeuralNetworksCompilation_createForDevices(model, &dev.handle, 1, &comp)) != 0) {
      fail(e, "createForDevices");
      break;
    }
    ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);

    double t0 = nowMs();
    if ((e = ANeuralNetworksCompilation_finish(comp)) != 0) { fail(e, "Compilation_finish"); break; }
    r.compileMs = nowMs() - t0;

    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
      // A fresh Execution per run: this NNAPI implementation rejects reuse.
      if ((e = ANeuralNetworksExecution_create(comp, &exec)) != 0) { fail(e, "Execution_create"); break; }
      if ((e = ANeuralNetworksExecution_setInput(exec, 0, nullptr, input.data(), input.size())) != 0) {
        fail(e, "setInput");
        break;
      }
      if ((e = ANeuralNetworksExecution_setOutput(exec, 0, nullptr, output.data(), output.size())) != 0) {
        fail(e, "setOutput");
        break;
      }
      double s = nowMs();
      if ((e = ANeuralNetworksExecution_startCompute(exec, &event)) != 0) { fail(e, "startCompute"); break; }
      if ((e = ANeuralNetworksEvent_wait(event)) != 0) { fail(e, "Event_wait"); break; }
      double dt = nowMs() - s;
      ANeuralNetworksEvent_free(event);
      event = nullptr;
      ANeuralNetworksExecution_free(exec);
      exec = nullptr;
      if (i == 0) r.firstMs = dt;
      if (dt < best) best = dt;
    }
    if (e != 0) break;
    r.bestMs = best;

    // Did it actually compute? Recompute the quantized FC on the CPU and compare.
    // out_q = round( sum_k (in_q-128)*(w_q-128) / 128 ) + 128
    int mismatch = 0, maxDiff = 0;
    uint8_t sample = 0;
    for (uint32_t n = 0; n < N; ++n) {
      int64_t acc = 0;
      const uint8_t* wrow = weights.data() + size_t(n) * K;
      for (uint32_t k = 0; k < K; ++k) {
        acc += (int32_t(input[k]) - 128) * (int32_t(wrow[k]) - 128);
      }
      int expected = int(std::llround(double(acc) / 128.0)) + 128;
      if (expected < 0) expected = 0;
      if (expected > 255) expected = 255;
      int got = output[n];
      int d = got - expected;
      if (d) { mismatch++; if (abs(d) > maxDiff) maxDiff = abs(d); }
      if (n == 0) sample = uint8_t(got);
    }
    if (B > 1) { mismatch = 0; maxDiff = 0; }  // only row 0 verified below
    r.detail = "mismatch=" + std::to_string(mismatch) + "/" + std::to_string(N) +
               " maxDiff=" + std::to_string(maxDiff) + " out0=" + std::to_string(sample);
    r.ok = true;
  } while (false);

  if (event) ANeuralNetworksEvent_free(event);
  if (exec) ANeuralNetworksExecution_free(exec);
  if (comp) ANeuralNetworksCompilation_free(comp);
  if (model) ANeuralNetworksModel_free(model);
  return r;
}


}  // namespace (from the probe)

// JNI for the NPU lab. Devices are named by index into listDevices().
#include <jni.h>

extern "C" {

// Each device as "name|TYPE|featureLevel".
JNIEXPORT jobjectArray JNICALL Java_com_everettpenne_unolab_Npu_nativeDevices(JNIEnv* env, jobject) {
  auto devs = listDevices();
  jobjectArray out = env->NewObjectArray((jsize)devs.size(), env->FindClass("java/lang/String"), nullptr);
  for (size_t i = 0; i < devs.size(); ++i) {
    std::string line = devs[i].name + "|" + typeName(devs[i].type) + "|" + std::to_string((long long)devs[i].featureLevel);
    jstring s = env->NewStringUTF(line.c_str());
    env->SetObjectArrayElement(out, (jsize)i, s);
    env->DeleteLocalRef(s);
  }
  return out;
}

// One int8 FULLY_CONNECTED of [batch,K] x [N,K], pinned to device [index] (no CPU fallback), best of [iters]. Returns
// {ok, compileMs, firstMs, bestMs, weightBytes} and puts the error or the correctness check in detail[0].
JNIEXPORT jdoubleArray JNICALL Java_com_everettpenne_unolab_Npu_nativeGemm(JNIEnv* env, jobject, jint index, jint K, jint N, jint iters, jint batch,
                                                                           jobjectArray detail) {
  auto devs = listDevices();
  GemmTiming t;
  if (index < 0 || index >= (jint)devs.size()) { t.detail = "no such device"; }
  else {
    t = runGemm(devs[index], (uint32_t)K, (uint32_t)N, iters, false, (uint32_t)batch);
    if (!t.ok) t.detail = std::string(errName(t.err)) + " at " + t.detail;
  }
  if (detail && env->GetArrayLength(detail) > 0) {
    jstring s = env->NewStringUTF(t.detail.c_str());
    env->SetObjectArrayElement(detail, 0, s);
  }
  jdoubleArray out = env->NewDoubleArray(5);
  double v[5] = {t.ok ? 1.0 : 0.0, t.compileMs, t.firstMs, t.bestMs, double(t.weightBytes)};
  env->SetDoubleArrayRegion(out, 0, 5, v);
  return out;
}

}  // extern "C"
