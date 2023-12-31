cmake_minimum_required(VERSION 3.12)
project(BasiCPP_Pitch VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Global settings
set(LDFLAGS "-O3 -fopenmp -fno-math-errno -march=native -DNDEBUG -w")

if ( GPROF )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# MKL
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
include_directories(${MKL_INCLUDE})
# set(ENV{MKLROOT} /opt/intel/oneapi/mkl/2023.2.0)
set(LDFLAGS "${LDFLAGS} -L{MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 -m64 -I${MKLROOT}/include")
# set(BLAS_VENDOR Intel10_64lp)
# find_package(BLAS REQUIRED)
# print MKLROOT
# include_directories(${MKLROOT}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LDFLAGS}")

# cnpy
set( CNPY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/ext/cnpy )
include_directories(${CNPY_ROOT})

# nlohmann json
set( JSON_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/ext/nlohmann-json )
include_directories(${JSON_ROOT})

# Eigen
set(EIGEN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/ext/eigen-3.3.7)
# set(EIGEN_ROOT /usr/include/eigen3)
include_directories(${EIGEN_ROOT})

# Find Python and pybind11 package
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

find_package(ZLIB REQUIRED)
link_libraries(${ZLIB_LIBRARIES})

file(GLOB EXE_SRCS
    ./src/*.cpp
    ./src/ext/cnpy/cnpy.cpp
)
# Exclude bind.cpp from sources
list(FILTER EXE_SRCS EXCLUDE REGEX ".*bind.cpp$")

file(GLOB PY_SRCS
    ./src/*.cpp
    ./src/ext/cnpy/cnpy.cpp
)
# Exclude main.cpp from sources
list(FILTER PY_SRCS EXCLUDE REGEX ".*main.cpp$")

file(GLOB HEADERS
    ./src/*.h
    ./src/ext/cnpy/cnpy.h
    ./src/ext/nlohmann-json/json.hpp
)

if ( BUILD_PY )
    # Set so library output path
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

    pybind11_add_module(BasiCPP_Pitch MODULE ${PY_SRCS} ${HEADERS})

    # Create so library
    target_compile_definitions(BasiCPP_Pitch PRIVATE VERSION_INFO=${PROJECT_VERSION})
endif()

if ( BUILD_EXE )
    # Set executable output path
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
    # Create C++ executable
    add_executable(run ${EXE_SRCS} ${HEADERS})
endif()