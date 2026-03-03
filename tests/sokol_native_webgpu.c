
#if defined(__APPLE__)
#define SOKOL_METAL 1
#else
#define SOKOL_NOAPI 1
#endif

#define SOKOL_APP_IMPL
#include "sokol_app.h"


#include "sokol_native_webgpu.h"
#include <stdio.h>

// ---------------------------------------------------------------------------------------------------------------------------
static void request_adapter_callback(WGPURequestAdapterStatus status, WGPUAdapter received,  WGPUStringView message, void* userdata1, void* userdata2)
{
    webgpu_platform* wgpu = (webgpu_platform*) userdata1;
    if (status == WGPURequestAdapterStatus_Success) 
    {
        wgpu->adapter = received;
    } 
    else 
    {
        fprintf(stderr, "Failed to get adapter: %s\n", message.data);
        exit(1);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
void init_webgpu(webgpu_platform* wgpu)
{
    *wgpu = (webgpu_platform) {};

    WGPUInstanceDescriptor instance_desc = {0};
    wgpu->instance = wgpuCreateInstance(&instance_desc);

    WGPURequestAdapterOptions adapter_opts = {0};
    adapter_opts.powerPreference = WGPUPowerPreference_HighPerformance;
    adapter_opts.compatibleSurface = NULL; // no surface yet

    WGPURequestAdapterCallbackInfo cb_info = 
    {
        .callback = request_adapter_callback,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .userdata1 = wgpu
    };

    wgpuInstanceRequestAdapter(wgpu->instance, &adapter_opts, cb_info);
    while (wgpu->adapter == NULL) 
        wgpuInstanceProcessEvents(wgpu->instance);
}


