#ifndef SOKOL_NATIVE_WEBGPU

#include <webgpu.h>
#include <GLFW/glfw3.h>

typedef struct webgpu_platform
{
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUSurface surface;
    WGPUSurfaceConfiguration surface_cfg;
} webgpu_platform;


#ifdef __cplusplus
extern "C" {
#endif

void init_webgpu(webgpu_platform* wgpu, struct GLFWwindow* window);
void terminate_webgpu(webgpu_platform* wgpu);

#ifdef __cplusplus
}
#endif

#endif