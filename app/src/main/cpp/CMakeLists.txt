cmake_minimum_required(VERSION 3.22.1)
project(unolab C CXX)

# llama.cpp for the CPU, without the native-CPU probe (it would detect the build machine). A Tensor G4 has dotprod and i8mm; the emulator's CPU
# may lack i8mm (SIGILL), so the baseline here is armv8.2-a with dotprod and fp16, which every arm64 Pixel since the 6 has.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/llama.cpp llama-build)

add_library(unolab SHARED llm_jni.cpp npu_jni.cpp)
target_compile_features(unolab PRIVATE cxx_std_17)
target_link_libraries(unolab llama ggml neuralnetworks android log)
