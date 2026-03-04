#ifndef SOKOL_NATIVE_WEBGPU

#include <webgpu.h>

typedef struct webgpu_platform
{
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
} webgpu_platform;


#ifdef __cplusplus
extern "C" {
#endif

void init_webgpu(webgpu_platform* wgpu);

#ifdef __cplusplus
}
#endif

#endif