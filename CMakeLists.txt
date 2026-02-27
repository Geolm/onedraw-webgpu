cmake_minimum_required(VERSION 3.25)
project(onedraw_webgpu)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra ")
set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -Wextra")
set(WGPU_VERSION "27.0.4.0")
set(WGPU_BUILD_TYPE "debug")  # or release

if(APPLE)
    set(WGPU_PLATFORM "macos-aarch64")
endif()

if(WIN32)
    set(WGPU_PLATFORM "windows-x86_64-msvc")
endif()

set(WGPU_URL https://github.com/gfx-rs/wgpu-native/releases/download/v${WGPU_VERSION}/wgpu-${WGPU_PLATFORM}-${WGPU_BUILD_TYPE}.zip)

set(WGPU_DIR "${CMAKE_BINARY_DIR}/wgpu-native")

if(NOT EXISTS ${WGPU_DIR})
    file(DOWNLOAD
        ${WGPU_URL}
        "${CMAKE_BINARY_DIR}/wgpu.zip"
    )
    file(MAKE_DIRECTORY ${WGPU_DIR})
    file(ARCHIVE_EXTRACT
        INPUT "${CMAKE_BINARY_DIR}/wgpu.zip"
        DESTINATION ${WGPU_DIR}
    )
endif()

include_directories(${WGPU_DIR}/include)
include_directories(${WGPU_DIR}/include/webgpu)
include_directories(${WGPU_DIR}/include/wgpu)

if(NOT CMAKE_GENERATOR STREQUAL "Xcode")
    add_subdirectory(prebuild)
endif()
