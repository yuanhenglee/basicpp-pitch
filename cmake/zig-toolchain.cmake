# Toolchain that uses Zig's bundled clang + mingw-w64 (pip install ziglang).
# cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -B build
set(CMAKE_SYSTEM_NAME Windows)
set(_zig_dir "${CMAKE_CURRENT_LIST_DIR}/zig")
set(CMAKE_C_COMPILER "${_zig_dir}/zig-cc.cmd")
set(CMAKE_CXX_COMPILER "${_zig_dir}/zig-cxx.cmd")
set(CMAKE_AR "${_zig_dir}/zig-ar.cmd")
set(CMAKE_RANLIB "${_zig_dir}/zig-ranlib.cmd")
set(CMAKE_RC_COMPILER "${_zig_dir}/zig-rc.cmd")
set(BPP_ZIG_TOOLCHAIN ON)
