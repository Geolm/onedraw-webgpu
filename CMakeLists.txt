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

macro(setup_app app)
    if(APPLE)
        target_link_libraries(${app}
                "${WGPU_DIR}/lib/libwgpu_native.a"
                "-framework QuartzCore"
                "-framework Cocoa"
                "-framework MetalKit"
                "-framework Metal")
    endif()

    if(WIN32)
        target_link_libraries(${app} "${WGPU_DIR}/lib/wgpu_native.lib" Opengl32 ws2_32 D3DCompiler BCrypt Ntdll Userenv RuntimeObject Propsys)
        # Copy wgpu_native.dll where we produce the binary for the executable:
        add_custom_command(TARGET ${app} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${WGPU_DIR}/lib/wgpu_native.dll
                $<TARGET_FILE_DIR:${app}>
        )
    endif()
endmacro()

# --- Minimal ---
add_executable(minimal
    ./lib/onedraw.c
    ./tests/minimal.c
    ./tests/sokol_app.mm
)

if(TARGET build_lib)
    add_dependencies(minimal build_lib)
endif()

setup_app(minimal)