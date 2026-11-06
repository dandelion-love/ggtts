cmake_minimum_required(VERSION 3.14)
project(layer0_ref CXX C)

# QWEN3_TTS_CPP: a checkout of https://github.com/Danmoreng/qwen3-tts.cpp at 16bb5af with its ggml submodule.
# ggml is built from that checkout's own submodule, as qwen3-tts.cpp's CMakeLists.txt does (lines 48-58), so the
# reference side runs the ggml the reference vendors.
if(NOT QWEN3_TTS_CPP)
    message(FATAL_ERROR "Set -DQWEN3_TTS_CPP=<path to qwen3-tts.cpp at 16bb5af>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${QWEN3_TTS_CPP}/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)

add_executable(layer0_ref layer0_ref.cpp)
target_link_libraries(layer0_ref PRIVATE ggml ggml-base ggml-cpu)
