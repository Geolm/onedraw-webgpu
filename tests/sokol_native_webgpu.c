
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
        fprintf(stderr, "failed to get adapter: %.*s\n", (int)message.length, message.data);
        exit(1);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
static void request_device_callback(WGPURequestDeviceStatus status, WGPUDevice received,  WGPUStringView message, void* userdata1, void* userdata2)
{
    webgpu_platform* wgpu = (webgpu_platform*) userdata1;
    if (status == WGPURequestDeviceStatus_Success) 
    {
        wgpu->device = received;
    } 
    else 
    {
        fprintf(stderr, "failed to get device: %.*s\n", (int)message.length, message.data);
        exit(1);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
static void device_error_callback(WGPUDevice const* device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* userdata2)
{
    fprintf(stderr,
            "[WebGPU][%d] %.*s\n",
            type,
            (int)message.length,
            message.data);
}

// ---------------------------------------------------------------------------------------------------------------------------
void init_webgpu(webgpu_platform* wgpu)
{
    *wgpu = (webgpu_platform) {};

    // Instance
    WGPUInstanceDescriptor instance_desc = {0};
    wgpu->instance = wgpuCreateInstance(&instance_desc);

    // Surface
    WGPUSurfaceDescriptor surface_desc = {};

#if defined(__APPLE__)
    id metal_layer = NULL;
    NSWindow *ns_window = reinterpret_cast<NSWindow*>(sapp_macos_get_window());
    [ns_window.contentView setWantsLayer:YES];
    metal_layer = [CAMetalLayer layer];
    [ns_window.contentView setLayer:metal_layer];
    WGPUSurfaceSourceMetalLayer layer = {};
    layer.layer = metal_layer;
    layer.chain.sType = WGPUSType_SurfaceSourceMetalLayer;
#else
    HINSTANCE hinstance =  GetModuleHandle(NULL);
    WGPUSurfaceSourceWindowsHWND layer = {};
    layer.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
    layer.hinstance = hinstance;
    layer.hwnd = (void*) sapp_win32_get_hwnd();
#endif

    surface_desc.nextInChain = &layer.chain;
    wgpu->surface = wgpuInstanceCreateSurface(wgpu->instance, &surface_desc);

    // Adapter
    WGPURequestAdapterOptions adapter_opts = 
    {
        .powerPreference = WGPUPowerPreference_HighPerformance,
        .compatibleSurface = wgpu->surface
    };

    WGPURequestAdapterCallbackInfo adapter_cb_info = 
    {
        .callback = request_adapter_callback,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .userdata1 = wgpu
    };

    wgpuInstanceRequestAdapter(wgpu->instance, &adapter_opts, adapter_cb_info);
    while (wgpu->adapter == NULL) 
        wgpuInstanceProcessEvents(wgpu->instance);

    WGPUAdapterInfo adapter_info;
    wgpuAdapterGetInfo(wgpu->adapter, &adapter_info);

    fprintf(stdout, "\n== WebGPU Adapter Info:\n");
    fprintf(stdout, "Vendor: %.*s\n", (int)adapter_info.vendor.length, adapter_info.vendor.data);
    fprintf(stdout, "Architecture: %.*s\n", (int)adapter_info.architecture.length, adapter_info.architecture.data);
    fprintf(stdout, "Device: %.*s\n", (int)adapter_info.device.length, adapter_info.device.data);
    fprintf(stdout, "Description: %.*s\n", (int)adapter_info.description.length, adapter_info.description.data);
    fprintf(stdout, "VendorID: 0x%08X\n", adapter_info.vendorID);
    fprintf(stdout, "DeviceID: 0x%08X\n", adapter_info.deviceID);
    fprintf(stdout, "BackendType: %d\n", adapter_info.backendType);
    fprintf(stdout, "AdapterType: %d\n\n", adapter_info.adapterType);

    wgpuAdapterInfoFreeMembers(adapter_info);

    // Device
    WGPURequestDeviceCallbackInfo device_cb_info =
    {
        .callback = request_device_callback,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .userdata1 = wgpu
    };

    WGPUDeviceDescriptor device_desc = {.uncapturedErrorCallbackInfo = {.callback = device_error_callback}};
    wgpuAdapterRequestDevice(wgpu->adapter, &device_desc, device_cb_info);
    while (wgpu->device == NULL) 
        wgpuInstanceProcessEvents(wgpu->instance);

    // Surface capabilities
    WGPUSurfaceCapabilities surface_caps = {};
    wgpuSurfaceGetCapabilities(wgpu->surface, wgpu->adapter, &surface_caps);

    wgpu->surface_cfg = (WGPUSurfaceConfiguration)
    {
        .device = wgpu->device,
        .usage = WGPUTextureUsage_RenderAttachment,
        .format = surface_caps.formats[0],
        .viewFormatCount = 1,
        .presentMode = WGPUPresentMode_Fifo,
        .alphaMode = surface_caps.alphaModes[0],
        .width = 1,
        .height = 1
    };

    wgpu->surface_cfg.viewFormats = &wgpu->surface_cfg.format;

    wgpuSurfaceConfigure(wgpu->surface, &wgpu->surface_cfg);

    fprintf(stdout, "== WebGPU Surface Capabilities\n");

    fprintf(stdout, "Formats (%zu):\n", surface_caps.formatCount);
    for (uint32_t i = 0; i < surface_caps.formatCount; ++i)
        fprintf(stdout, "  [%u] %d\n", i, surface_caps.formats[i]);

    fprintf(stdout, "Present Modes (%zu):\n", surface_caps.presentModeCount);
    for (uint32_t i = 0; i < surface_caps.presentModeCount; ++i)
        fprintf(stdout, "  [%u] %d\n", i, surface_caps.presentModes[i]);

    fprintf(stdout, "Alpha Modes (%zu):\n", surface_caps.alphaModeCount);
    for (uint32_t i = 0; i < surface_caps.alphaModeCount; ++i)
        fprintf(stdout, "  [%u] %d\n", i, surface_caps.alphaModes[i]);

    wgpuSurfaceCapabilitiesFreeMembers(surface_caps);
}

// ---------------------------------------------------------------------------------------------------------------------------
void terminate_webgpu(webgpu_platform* wgpu)
{
    wgpuDeviceRelease(wgpu->device);
    wgpuAdapterRelease(wgpu->adapter);
    wgpuSurfaceRelease(wgpu->surface);
}


