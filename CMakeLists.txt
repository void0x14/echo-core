cmake_minimum_required(VERSION 3.16)
project(echo_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags for AVX2 + FMA + F16C (required)
set(ECHO_AVX2_FLAGS -mavx2 -mfma -mf16c)

# --- Main executable ---
add_executable(echo_core
    src/main.cpp
    src/gguf_reader.cpp
    src/kernels/matvec.cpp
    src/kernels/quant.cpp
    src/kv_cache.cpp
    src/inference.cpp
)
target_compile_options(echo_core PRIVATE ${ECHO_AVX2_FLAGS} -O3)
target_include_directories(echo_core PRIVATE ${CMAKE_SOURCE_DIR}/include)

# --- Inference smoke test ---
add_executable(inference_test
    src/inference.cpp
    src/gguf_reader.cpp
    src/kv_cache.cpp
    src/kernels/matvec.cpp
    src/kernels/quant.cpp
)
target_compile_definitions(inference_test PRIVATE INFERENCE_TEST_MAIN)
target_compile_options(inference_test PRIVATE ${ECHO_AVX2_FLAGS} -O2)
target_include_directories(inference_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# --- Quant kernel unit tests ---
add_executable(quant_test src/kernels/quant.cpp)
target_compile_definitions(quant_test PRIVATE QUANT_TEST_MAIN)
target_compile_options(quant_test PRIVATE ${ECHO_AVX2_FLAGS} -O2)
target_include_directories(quant_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(quant_test PRIVATE m)
