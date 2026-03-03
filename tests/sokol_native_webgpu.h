#ifndef SOKOL_NATIVE_WEBGPU

#include <webgpu.h>

typedef struct webgpu_platform
{
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
} webgpu_platform;


void init_webgpu(webgpu_platform* wgpu);

#endif