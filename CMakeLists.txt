cmake_minimum_required(VERSION 3.24)
project(Jarvis LANGUAGES CXX C)

# Modern C++ standard required for MLX and AI workloads
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. Apple Silicon Optimization ---
# Accelerate framework is Apple's native, highly optimized math/DSP library. 
# interfaces with the M1 architecture for audio processing (STT).
find_library(ACCELERATE_FRAMEWORK Accelerate)

# --- 2. MLX Core (The Tensor Engine) ---
find_package(MLX REQUIRED)

# --- 3. Tokenizer (Text-to-ID) ---
# We use FetchContent to pull tokenizers-cpp directly from GitHub during the build.
# Dont have to manage git submodules manually.
include(FetchContent)
FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG main
)

# Note: tokenizers-cpp requires Rust installed on your machine to compile its core.
FetchContent_MakeAvailable(tokenizers_cpp)

# --- 4. The Jarvis Executable ---
add_executable(Jarvis 
    src/main.cpp
    src/Model.cpp
    src/Weights.cpp
    src/KVCache.cpp
)

# Let the compiler know where custom headers live
target_include_directories(Jarvis PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${MLX_INCLUDE_DIRS}
)

# --- 5. Linking ---
target_link_libraries(Jarvis 
    PRIVATE 
    mlx                    # The Unified Memory tensor engine
    tokenizers_cpp              # The BPE tokenizer for Phi-3
    "-framework Accelerate"     # Apple's native framework
)