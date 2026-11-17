cmake_minimum_required(VERSION 3.16)
project(basicpp_pitch VERSION 2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Instruction set for the kernels. x86-64-v3 = AVX2 + FMA (Haswell / Zen and newer).
# Default: AVX2 + FMA on x86-64, the compiler's default elsewhere (scalar kernels, e.g. ARM).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
  set(_bpp_default_arch "x86-64-v3")
else()
  set(_bpp_default_arch "")
endif()
set(BPP_ARCH "${_bpp_default_arch}" CACHE STRING
    "Value for -march (native, x86-64-v3, x86-64 for a scalar build; empty = compiler default). MSVC: x86-64-v3/native -> /arch:AVX2")
set(_bpp_march "${BPP_ARCH}")
if(BPP_ZIG_TOOLCHAIN)
  string(REPLACE "-" "_" _bpp_march "${_bpp_march}")  # zig spells CPU models x86_64_v3
endif()

add_library(bpp_core OBJECT
  src/audio.cpp
  src/engine.cpp
  src/midi.cpp
  src/model_data.cpp
  src/notes.cpp
  src/resample.cpp
  src/window_model.cpp
  src/c_api.cpp)
target_include_directories(bpp_core PUBLIC include PRIVATE src)
set_target_properties(bpp_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(bpp_core PRIVATE BPP_BUILDING_LIBRARY)

if(MSVC)
  target_compile_options(bpp_core PRIVATE /O2 /fp:precise /W3)
  if(BPP_ARCH MATCHES "^(x86-64-v3|x86-64-v4|native)$")
    target_compile_options(bpp_core PRIVATE /arch:AVX2)
  endif()
else()
  # No implicit FMA contraction: scalar code must round exactly like NumPy; kernels use explicit FMA.
  target_compile_options(bpp_core PRIVATE -O3 -ffp-contract=off -fno-math-errno -Wall -Wextra)
  if(_bpp_march)
    target_compile_options(bpp_core PRIVATE -march=${_bpp_march})
  endif()
endif()

# libsoxr (LGPL-2.1), configured like python-soxr 1.1 so resampling matches librosa.load bit for bit.
option(BPP_WITH_SOXR "Bundle libsoxr for librosa-identical resampling of non-22050 Hz input" ON)
if(BPP_WITH_SOXR)
  set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(WITH_OPENMP OFF CACHE BOOL "" FORCE)
  set(WITH_LSR_BINDINGS OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(third_party/libsoxr EXCLUDE_FROM_ALL)
  if(NOT MSVC)
    target_compile_options(soxr PRIVATE -ffp-contract=off -w)  # MSVC-built wheels never fuse multiply-adds
  endif()
  if(MINGW OR (WIN32 AND NOT MSVC))
    # Use the Universal CRT's libm like the MSVC-built python-soxr wheels (see ucrt_math.h).
    target_sources(soxr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ucrt_math/ucrt_math.c)
    target_compile_options(soxr PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ucrt_math/ucrt_math.h)
  endif()
  target_compile_definitions(bpp_core PRIVATE BPP_HAVE_SOXR)
  target_include_directories(bpp_core PRIVATE third_party/libsoxr/src)
endif()

find_package(Threads REQUIRED)

add_library(bpp STATIC $<TARGET_OBJECTS:bpp_core>)
target_include_directories(bpp PUBLIC include)
target_link_libraries(bpp PUBLIC Threads::Threads)

add_library(bpp_shared SHARED $<TARGET_OBJECTS:bpp_core>)
set_target_properties(bpp_shared PROPERTIES OUTPUT_NAME basicpp_pitch PREFIX "")
target_link_libraries(bpp_shared PRIVATE Threads::Threads)
if(BPP_WITH_SOXR)
  target_link_libraries(bpp PUBLIC soxr)
  target_link_libraries(bpp_shared PRIVATE soxr)
endif()

add_executable(bpp_cli apps/bpp_cli.cpp)
set_target_properties(bpp_cli PROPERTIES OUTPUT_NAME bpp)
target_link_libraries(bpp_cli PRIVATE bpp)
if(NOT MSVC)
  target_compile_options(bpp_cli PRIVATE -O2)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(test_fixture tests/test_fixture.cpp)
  target_link_libraries(test_fixture PRIVATE bpp)
  add_test(NAME golden_fixture COMMAND test_fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures megalovania_6s)
endif()

option(BPP_BUILD_DEBUG_TOOLS "Build internal debugging tools" ON)
if(BPP_BUILD_DEBUG_TOOLS)
  add_executable(load_audio_dump tools/debug/load_audio_dump.cpp)
  target_link_libraries(load_audio_dump PRIVATE bpp)
  target_include_directories(load_audio_dump PRIVATE src)
  add_executable(window_dump tools/debug/window_dump.cpp)
  target_include_directories(window_dump PRIVATE src)
  target_link_libraries(window_dump PRIVATE bpp)
  if(NOT MSVC)
    target_compile_options(window_dump PRIVATE -O2 -ffp-contract=off)
    if(_bpp_march)
      target_compile_options(window_dump PRIVATE -march=${_bpp_march})
    endif()
  endif()
endif()
