#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <GLFW/glfw3.h>

#include "native_webgpu.h"
#include "../lib/onedraw.h"

#define FROM_HTML(html)   ((html&0xff)<<16) | ((html>>16)&0xff) | (html&0x00ff00) | 0xff000000

// https://lospec.com/palette-list/miyazaki-16
static const uint32_t miya_black = FROM_HTML(0x232228);
static const uint32_t miya_dark_blue = FROM_HTML(0x284261);
static const uint32_t miya_dark_grey = FROM_HTML(0x5f5854);
static const uint32_t miya_grey = FROM_HTML(0x878573);
static const uint32_t miya_light_grey = FROM_HTML(0xb8b095);
static const uint32_t miya_pale_blue = FROM_HTML(0xc3d5c7);
static const uint32_t miya_white = FROM_HTML(0xebecdc);
static const uint32_t miya_blue = FROM_HTML(0x2485a6);
static const uint32_t miya_light_blue = FROM_HTML(0x54bad2);
static const uint32_t miya_brown = FROM_HTML(0x754d45);
static const uint32_t miya_red = FROM_HTML(0xc65046);
static const uint32_t miya_pink = FROM_HTML(0xe6928a);
static const uint32_t miya_dark_green = FROM_HTML(0x1e7453);
static const uint32_t miya_green = FROM_HTML(0x55a058);
static const uint32_t miya_light_green = FROM_HTML(0xa1bf41);
static const uint32_t miya_yellow = FROM_HTML(0xe3c054);

struct onedraw* renderer;
struct webgpu_platform wgpu;
struct GLFWwindow* window;

//-----------------------------------------------------------------------------------------------------------------------------
void init(void)
{
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    init_webgpu(&wgpu, window);

    renderer = od_init( &(onedraw_def)
    {
        .device = wgpu.device,
        .viewport_width = width,
        .viewport_height = height,
        .srgb_rendertarget = true,
        .clear_rendertarget = true,
        .surface_format = wgpu.surface_cfg.format
    });

    od_set_clear_color(renderer, miya_light_grey);
}

//-----------------------------------------------------------------------------------------------------------------------------
void frame(void)
{
    WGPUSurfaceTexture surfaceTexture;
    wgpuSurfaceGetCurrentTexture(wgpu.surface, &surfaceTexture);

    if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
        return;

    WGPUTextureView frame = wgpuTextureCreateView(surfaceTexture.texture, NULL);

    od_begin_frame(renderer);
    od_draw_ring(renderer, 500.f, 500.f, 250.f, 20.f, miya_red);
    od_draw_text(renderer, 0, 0, "Hello world!", 0xffffffff);
    od_end_frame(renderer, frame);

    wgpuSurfacePresent(wgpu.surface);
    wgpuTextureViewRelease(frame);
    wgpuTextureRelease(surfaceTexture.texture);
}

//-----------------------------------------------------------------------------------------------------------------------------
void cleanup(void)
{
    od_terminate(renderer);
    terminate_webgpu(&wgpu);
}

//-----------------------------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    (void)(argc);
    (void)(argv);

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 0);

    window = glfwCreateWindow(1280, 720, "onedraw-webgpu", NULL, NULL);
    assert(window != NULL);

    init();

    while (!glfwWindowShouldClose(window))
    {
        frame();
        glfwPollEvents();
    }

    cleanup();

    return 0;
}