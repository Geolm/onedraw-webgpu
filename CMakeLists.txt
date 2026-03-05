cmake_minimum_required(VERSION 3.25)
project(onedraw_webgpu)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra ")
set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -Wextra")
set(WGPU_VERSION "27.0.4.0")
set(WGPU_BUILD_TYPE "debug")  # or release

# ---- wgpu ---
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

# --- Objective-c poutine ---
if(APPLE)
    enable_language(OBJC)

    set_source_files_properties(
        ./tests/native_webgpu.c
        PROPERTIES
        COMPILE_FLAGS "-x objective-c -std=gnu11"
    )
endif()

# --- GLFW ---
include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(glfw)

# --- Macro ---

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

# --- Minimal app ---
add_executable(minimal
    ./lib/onedraw.c
    ./tests/minimal.c
    ./tests/native_webgpu.c
)

target_link_libraries(minimal glfw)

if(TARGET build_lib)
    add_dependencies(minimal build_lib)
endif()

setup_app(minimal)