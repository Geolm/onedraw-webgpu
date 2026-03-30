#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <GLFW/glfw3.h>

#include "native_webgpu.h"

#include "../lib/onedraw.h"

struct onedraw* renderer;
struct webgpu_platform wgpu;
struct GLFWwindow* window;

void init(void)
{
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    init_webgpu(&wgpu, window);

    renderer = od_init( &(onedraw_def)
    {
        .device = wgpu.device,
        .viewport_width = wgpu.surface_cfg.width,
        .viewport_height = wgpu.surface_cfg.height,
        .srgb_rendertarget = true,
        .clear_rendertarget = true,
        .surface_format = wgpu.surface_cfg.format
    });

    od_set_clear_color(renderer, 0xff457623);
}

void frame(void)
{
    WGPUSurfaceTexture surfaceTexture;
    wgpuSurfaceGetCurrentTexture(wgpu.surface, &surfaceTexture);

    if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
        return;

    WGPUTextureView frame = wgpuTextureCreateView(surfaceTexture.texture, NULL);

    od_begin_frame(renderer);
    // od_draw_text(renderer, 0, 0, "Hello world!", 0xffffffff);
    od_end_frame(renderer, frame);

    wgpuSurfacePresent(wgpu.surface);
    wgpuTextureViewRelease(frame);
    wgpuTextureRelease(surfaceTexture.texture);
}

void cleanup(void)
{
    od_terminate(renderer);
    terminate_webgpu(&wgpu);
}

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