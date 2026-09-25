#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include <GLFW/glfw3.h>

#include "native_webgpu.h"
#include "../lib/onedraw.h"

#define ATLAS_SIZE (256)
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

struct onedraw* g_renderer;
struct webgpu_platform g_wgpu;
struct GLFWwindow* g_window;

bool g_screenshot = false;
const char* g_screenshot_path = "screenshot.bmp";

// Last framebuffer size applied to the surface and renderer, used to dedupe the GLFW size callback.
uint32_t g_resize_width = 0;
uint32_t g_resize_height = 0;
od_stats g_stats;

// Last status returned by wgpuSurfaceGetCurrentTexture, checked in main() after the resize round-trip.
WGPUSurfaceGetCurrentTextureStatus g_last_surface_status = WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal;

void all_primitives(float width, float height);

//-----------------------------------------------------------------------------------------------------------------------------
static inline uint32_t lerp_color(uint32_t a, uint32_t b, float t)
{
    int tt = (int)(t * 256.f);
    int oneminust = 256 - tt;

    uint32_t A = (((a >> 24) & 0xFF) * oneminust + ((b >> 24) & 0xFF) * tt) >> 8;
    uint32_t B = (((a >> 16) & 0xFF) * oneminust + ((b >> 16) & 0xFF) * tt) >> 8;
    uint32_t G = (((a >> 8)  & 0xFF) * oneminust + ((b >> 8)  & 0xFF) * tt) >> 8;
    uint32_t R = (((a >> 0)  & 0xFF) * oneminust + ((b >> 0)  & 0xFF) * tt) >> 8;

    return (A << 24) | (B << 16) | (G << 8) | R;
}

// ---------------------------------------------------------------------------------------------------------------------------
void make_checker(uint32_t* p, uint32_t w, uint32_t h, uint32_t a, uint32_t b)
{
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++)
            *p++ = (((x >> 5) ^ (y >> 5)) & 1) ? a : b;
}

// ---------------------------------------------------------------------------------------------------------------------------
void make_rings(uint32_t* p, uint32_t w, uint32_t h, uint32_t a, uint32_t b)
{
    float cx = (float)w * 0.5f, cy = (float)h * 0.5f;
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) 
        {
            float dx = (float)x - cx, dy = (float)y - cy;
            uint32_t ring = (uint32_t)(sqrtf(dx * dx + dy * dy)) / 16;
            *p++ = (ring & 1) ? a : b;
        }
}

// ---------------------------------------------------------------------------------------------------------------------------
void make_hgradient(uint32_t* p, uint32_t w, uint32_t h, uint32_t a, uint32_t b)
{
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++)
        {
            float t = (float)x / (float)(w - 1);
            *p++ = lerp_color(a, b, t);
        }
}

// ---------------------------------------------------------------------------------------------------------------------------
void make_radial(uint32_t* p, uint32_t w, uint32_t h, uint32_t a, uint32_t b)
{
    float cx = (float)w * 0.5f;
    float cy = (float)h * 0.5f;
    float maxd = sqrtf(cx*cx + cy*cy);

    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) 
        {
            float dx = (float)x - cx;
            float dy = (float)y - cy;
            float t = sqrtf(dx*dx + dy*dy) / maxd;
            if (t > 1.f) t = 1.f;
            *p++ = lerp_color(a, b, t);
        }
}

// ---------------------------------------------------------------------------------------------------------------------------
void fill_texture_array(void)
{
    uint32_t* pixel_data = malloc(ATLAS_SIZE * ATLAS_SIZE * sizeof(uint32_t));

    make_checker(pixel_data, ATLAS_SIZE, ATLAS_SIZE, miya_green, miya_yellow);
    od_upload_slice(g_renderer, pixel_data, 0);

    make_hgradient(pixel_data, ATLAS_SIZE, ATLAS_SIZE, miya_dark_blue, miya_light_blue);
    od_upload_slice(g_renderer, pixel_data, 1);

    make_rings(pixel_data, ATLAS_SIZE, ATLAS_SIZE, miya_brown, miya_pink);
    od_upload_slice(g_renderer, pixel_data, 2);

    make_radial(pixel_data, ATLAS_SIZE, ATLAS_SIZE, miya_black, miya_light_green);
    od_upload_slice(g_renderer, pixel_data, 3);

    free(pixel_data);
}

//-----------------------------------------------------------------------------------------------------------------------------
void init(void)
{
    int width, height;
    glfwGetFramebufferSize(g_window, &width, &height);

    // The size callback dedupes against the last applied size, start from the initial framebuffer size.
    g_resize_width = (uint32_t)width;
    g_resize_height = (uint32_t)height;

    init_webgpu(&g_wgpu, g_window);

    g_renderer = od_init( &(onedraw_def)
    {
        .device = g_wgpu.device,
        .viewport_width = width,
        .viewport_height = height,
        .srgb_rendertarget = true,
        .surface_format = g_wgpu.surface_cfg.format,
        .atlas = 
        {
            .format = WGPUTextureFormat_RGBA8Unorm,
            .height = ATLAS_SIZE,
            .width = ATLAS_SIZE,
            .num_slices = 4
        }
    });

    fill_texture_array();

    od_set_clear_color(g_renderer, miya_white);
}

//-----------------------------------------------------------------------------------------------------------------------------
struct screenshot_ctx
{
    WGPUBuffer buffer;
    uint64_t size;
    uint32_t width;
    uint32_t height;
    uint32_t pitch;
    WGPUTextureFormat format;
    const char* path;
};

// ---------------------------------------------------------------------------------------------------------------------------
static void save_bmp(const char* path, uint32_t width, uint32_t height, uint32_t pitch, const uint8_t* data, WGPUTextureFormat format)
{
    bool bgra = (format == WGPUTextureFormat_BGRA8Unorm || format == WGPUTextureFormat_BGRA8UnormSrgb);
    bool rgba = (format == WGPUTextureFormat_RGBA8Unorm || format == WGPUTextureFormat_RGBA8UnormSrgb);
    if (!bgra && !rgba)
    {
        fprintf(stderr, "screenshot: unsupported format %d\n", (int)format);
        return;
    }

    FILE* file = fopen(path, "wb");
    if (file == NULL)
    {
        fprintf(stderr, "screenshot: could not open '%s' for writing\n", path);
        return;
    }

    uint32_t row_size = width * 4;

    uint16_t type = 0x4D42;                       // 'BM' little-endian
    uint32_t file_size = 54 + row_size * height;
    uint16_t reserved = 0;
    uint32_t pixel_offset = 54;

    fwrite(&type, 2, 1, file);
    fwrite(&file_size, 4, 1, file);
    fwrite(&reserved, 2, 1, file);
    fwrite(&reserved, 2, 1, file);
    fwrite(&pixel_offset, 4, 1, file);

    struct bitmap_info_header
    {
        uint32_t header_size;
        int32_t width;
        int32_t height;
        uint16_t planes;
        uint16_t bits_per_pixel;
        uint32_t compression;
        uint32_t image_size;
        uint32_t pixels_per_meter_x;
        uint32_t pixels_per_meter_y;
        uint32_t palette_used;
        uint32_t palette_important;
    } info_header =
    {
        .header_size = 40,
        .width = (int32_t)width,
        .height = -(int32_t)height,               // negative height = top-down rows
        .planes = 1,
        .bits_per_pixel = 32
    };

    fwrite(&info_header, sizeof(info_header), 1, file);

    for (uint32_t y = 0; y < height; y++)
    {
        const uint8_t* row = data + y * pitch;
        if (bgra)
            fwrite(row, row_size, 1, file);
        else
            for (uint32_t x = 0; x < row_size; x += 4)
            {
                uint8_t pixel[4] = { row[x + 2], row[x + 1], row[x], row[x + 3] };   // RGBA -> BGRA
                fwrite(pixel, 4, 1, file);
            }
    }

    fclose(file);

    printf("screenshot: saved %s (%ux%u)\n", path, width, height);
}

// ---------------------------------------------------------------------------------------------------------------------------
static void screenshot_map_cb(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2)
{
    struct screenshot_ctx* ctx = userdata1;
    (void)userdata2;

    if (status == WGPUMapAsyncStatus_Success)
    {
        const uint8_t* data = wgpuBufferGetConstMappedRange(ctx->buffer, 0, ctx->size);
        save_bmp(ctx->path, ctx->width, ctx->height, ctx->pitch, data, ctx->format);

        wgpuBufferUnmap(ctx->buffer);
    }
    else
    {
        if (message.data != NULL)
            fprintf(stderr, "screenshot: map failed (%d) %.*s\n", (int)status, (int)message.length, message.data);
        else
            fprintf(stderr, "screenshot: map failed (%d)\n", (int)status);
    }

    wgpuBufferRelease(ctx->buffer);
    free(ctx);
}

// ---------------------------------------------------------------------------------------------------------------------------
static void capture_screenshot(WGPUDevice device, WGPUTexture texture, uint32_t width, uint32_t height, WGPUTextureFormat format, const char* path)
{
    uint32_t pitch = ((width * 4) + 255) / 256 * 256;   // bytesPerRow must be a multiple of 256
    uint64_t size = (uint64_t)pitch * height;

    WGPUBufferDescriptor buffer_desc =
    {
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        .size = size
    };
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &buffer_desc);

    WGPUCommandEncoderDescriptor encoder_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encoder_desc);

    WGPUTexelCopyTextureInfo src =
    {
        .texture = texture,
        .origin = {0, 0, 0},
        .aspect = WGPUTextureAspect_All
    };

    WGPUTexelCopyBufferInfo dst =
    {
        .layout = {0, pitch, height},
        .buffer = buffer
    };

    WGPUExtent3D extent = {width, height, 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);

    WGPUCommandBufferDescriptor cmd_desc = {0};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuCommandEncoderRelease(encoder);

    WGPUQueue queue = wgpuDeviceGetQueue(device);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);

    struct screenshot_ctx* ctx = malloc(sizeof(*ctx));
    if (ctx == NULL)
    {
        wgpuBufferRelease(buffer);
        return;
    }

    ctx->buffer = buffer;
    ctx->size = size;
    ctx->width = width;
    ctx->height = height;
    ctx->pitch = pitch;
    ctx->format = format;
    ctx->path = path;

    WGPUBufferMapCallbackInfo cb_info =
    {
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = screenshot_map_cb,
        .userdata1 = ctx
    };
    // The map callback owns the buffer reference: it must stay alive until the callback runs.
    (void)wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, size, cb_info);
}

// ---------------------------------------------------------------------------------------------------------------------------
struct bmp_header
{
    uint32_t width;
    uint32_t height;
};

// Parses a 32-bit uncompressed BMP header and positions the stream on the pixel data.
static bool read_bmp_header(FILE* file, struct bmp_header* header)
{
    uint8_t raw[54];
    if (fread(raw, 1, sizeof(raw), file) != sizeof(raw) || raw[0] != 'B' || raw[1] != 'M')
        return false;

    uint32_t data_offset;
    uint32_t header_size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bits_per_pixel;
    uint32_t compression;

    memcpy(&data_offset, raw + 10, sizeof(data_offset));
    memcpy(&header_size, raw + 14, sizeof(header_size));
    memcpy(&width, raw + 18, sizeof(width));
    memcpy(&height, raw + 22, sizeof(height));
    memcpy(&planes, raw + 26, sizeof(planes));
    memcpy(&bits_per_pixel, raw + 28, sizeof(bits_per_pixel));
    memcpy(&compression, raw + 30, sizeof(compression));

    bool valid = header_size >= 40 &&
        planes == 1 &&
        bits_per_pixel == 32 &&
        compression == 0 &&
        width > 0 &&
        height != 0 &&
        data_offset >= 54;

    if (!valid)
        return false;

    header->width = (uint32_t)width;
    header->height = height < 0 ? (uint32_t)(-height) : (uint32_t)height;   // top-down rows are stored with a negative height
    fseek(file, (long)data_offset, SEEK_SET);

    return true;
}

// Unit test : compares the captured screenshot against the reference image. Pass if pixel-identical or PSNR > 45 dB.
static int compare_screenshots(const char* captured_path, const char* reference_path)
{
    FILE* captured = fopen(captured_path, "rb");
    if (captured == NULL)
    {
        fprintf(stderr, "screenshot: FAIL: cannot open '%s' (capture did not run or failed)\n", captured_path);
        return 1;
    }

    FILE* reference = fopen(reference_path, "rb");
    if (reference == NULL)
    {
        fprintf(stderr, "screenshot: FAIL: cannot open reference '%s'\n", reference_path);
        fclose(captured);
        return 1;
    }

    struct bmp_header captured_header;
    struct bmp_header reference_header;
    if (!read_bmp_header(captured, &captured_header) || !read_bmp_header(reference, &reference_header))
    {
        fprintf(stderr, "screenshot: FAIL: malformed or unsupported BMP header\n");
        fclose(captured);
        fclose(reference);
        return 1;
    }

    if (captured_header.width != reference_header.width || captured_header.height != reference_header.height)
    {
        fprintf(stderr, "screenshot: FAIL: size mismatch (%ux%u captured, %ux%u reference)\n", captured_header.width, captured_header.height,
                reference_header.width, reference_header.height);
        fprintf(stderr, "  (framebuffer size follows the display scale; the reference was captured on a different display or scale)\n");
        fclose(captured);
        fclose(reference);
        return 1;
    }

    size_t row_size = (size_t)captured_header.width * 4;
    uint8_t* captured_row = malloc(row_size);
    uint8_t* reference_row = malloc(row_size);
    if (captured_row == NULL || reference_row == NULL)
    {
        fprintf(stderr, "screenshot: FAIL: out of memory\n");
        free(captured_row);
        free(reference_row);
        fclose(captured);
        fclose(reference);
        return 1;
    }

    // Rows are streamed one at a time: the sum of squared errors is order-independent, so no full-frame buffer is needed.
    double sse = 0.0;
    for (uint32_t y = 0; y < captured_header.height; y++)
    {
        if (fread(captured_row, row_size, 1, captured) != 1 || fread(reference_row, row_size, 1, reference) != 1)
        {
            fprintf(stderr, "screenshot: FAIL: truncated pixel data in '%s' or '%s'\n", captured_path, reference_path);
            free(captured_row);
            free(reference_row);
            fclose(captured);
            fclose(reference);
            return 1;
        }

        for (size_t x = 0; x < row_size; x++)
        {
            int d = (int)captured_row[x] - (int)reference_row[x];
            sse += (double)(d * d);
        }
    }

    free(captured_row);
    free(reference_row);
    fclose(captured);
    fclose(reference);

    if (sse == 0.0)
    {
        printf("screenshot: PASS: images are identical\n");
        return 0;
    }

    double mse = sse / ((double)captured_header.width * (double)captured_header.height * 4.0);
    double psnr = 10.0 * log10(255.0 * 255.0 / mse);

    if (psnr > 45.0)
    {
        printf("screenshot: PASS: PSNR = %.2f dB (> 45.00 dB)\n", psnr);
        return 0;
    }

    printf("screenshot: FAIL: PSNR = %.2f dB (<= 45.00 dB)\n", psnr);
    return 1;
}

// ---------------------------------------------------------------------------------------------------------------------------
void frame(void)
{
    WGPUSurfaceTexture surfaceTexture;
    wgpuSurfaceGetCurrentTexture(g_wgpu.surface, &surfaceTexture);

    g_last_surface_status = surfaceTexture.status;

    if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
        return;

    WGPUTextureView frame = wgpuTextureCreateView(surfaceTexture.texture, NULL);

    int width, height;
    glfwGetFramebufferSize(g_window, &width, &height);

    // After a surface reconfigure, queued textures come back at the old size (suboptimal), and the framebuffer
    // can be 0x0 during window transitions (od_resize asserts > 16). Present without drawing to drain them,
    // then resume normal rendering.
    if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal || width == 0 || height == 0)
    {
        wgpuSurfacePresent(g_wgpu.surface);
        wgpuTextureViewRelease(frame);
        wgpuTextureRelease(surfaceTexture.texture);
        return;
    }

    od_begin_frame(g_renderer);
    all_primitives((float)width, (float)height);
    od_end_frame(g_renderer, frame);
    od_get_stats(g_renderer, &g_stats);

    if (g_screenshot)
    {
        g_screenshot = false;
        capture_screenshot(g_wgpu.device, surfaceTexture.texture, (uint32_t)width, (uint32_t)height, g_wgpu.surface_cfg.format, g_screenshot_path);
    }

    wgpuSurfacePresent(g_wgpu.surface);
    wgpuTextureViewRelease(frame);
    wgpuTextureRelease(surfaceTexture.texture);
}

//-----------------------------------------------------------------------------------------------------------------------------
void cleanup(void)
{
    od_terminate(g_renderer);
    terminate_webgpu(&g_wgpu);
}

//----------------------------------------------------------------------------------------------------------------------------
void key_cb(struct GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)(window);
    (void)(scancode);

    static bool culling_debug = false;
    if (key == GLFW_KEY_D && action == GLFW_PRESS && mods&GLFW_MOD_SUPER)
    {
        culling_debug = !culling_debug;
        od_set_culling_debug(g_renderer, culling_debug);
    }

    if (key == GLFW_KEY_S && action == GLFW_PRESS && mods&GLFW_MOD_SUPER)
    {
        g_screenshot_path = "screenshot.bmp";
        g_screenshot = true;
    }
}

//-----------------------------------------------------------------------------------------------------------------------------
void size_cb(struct GLFWwindow* window, int width, int height)
{
    (void)(width);
    (void)(height);

    int framebuffer_width, framebuffer_height;
    glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);

    // The callback arguments are in logical points (content-scale dependent), the app keys off framebuffer pixels.
    // Skip zero-sized frames (window transitions) and sizes that are already applied : GLFW can deliver several
    // callbacks per drag, and od_resize() must not run on a 0x0 framebuffer.
    if (framebuffer_width <= 0 || framebuffer_height <= 0)
        return;

    if ((uint32_t)framebuffer_width == g_resize_width && (uint32_t)framebuffer_height == g_resize_height)
        return;

    g_resize_width = (uint32_t)framebuffer_width;
    g_resize_height = (uint32_t)framebuffer_height;

    resize_webgpu(&g_wgpu, g_resize_width, g_resize_height);
    od_resize(g_renderer, g_resize_width, g_resize_height);
}

//-----------------------------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    bool automatic_exit = true;
    if (argc > 1 && strcmp(argv[1], "-noexit") == 0)
        automatic_exit = false;

    glfwInit();
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    float xscale, yscale;
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);

    // Fixed 1280x720 framebuffer regardless of display scale: request reference pixels / current content scale as logical points.
    // On 1x displays that is 1280x720 points; on 2x displays 640x360 points, both yielding the same 1280x720 framebuffer.
    uint32_t window_width = (uint32_t)(1280.0f / xscale);
    uint32_t window_height = (uint32_t)(720.0f / yscale);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 0);

    g_window = glfwCreateWindow(window_width, window_height, "onedraw-webgpu", NULL, NULL);
    assert(g_window != NULL);

    glfwSetKeyCallback(g_window, key_cb);
    glfwSetWindowSizeCallback(g_window, size_cb);

    init();

    // Original sizes, used to restore the window and to check the state after the round-trip.
    int fb_width, fb_height;
    glfwGetFramebufferSize(g_window, &fb_width, &fb_height);
    uint32_t original_framebuffer_width = (uint32_t)fb_width;
    uint32_t original_framebuffer_height = (uint32_t)fb_height;

    bool resized = false;
    bool restored = false;

    uint32_t frame_count = 0;
    while (!glfwWindowShouldClose(g_window))
    {
        if (frame_count == 30)
        {
            g_screenshot_path = "test0.bmp";
            g_screenshot = true;
        }
        else if (frame_count == 35 && !resized)
        {
            // Half the logical size : the size callback reconfigures the surface and grows the renderer.
            glfwSetWindowSize(g_window, (int)window_width/2, (int)window_height/2);
            resized = true;
        }
        else if (frame_count == 65 && resized && !restored)
        {
            // Restore the original size : the renderer shrinks again, exercising both directions of the resize.
            glfwSetWindowSize(g_window, (int)window_width, (int)window_height);
            restored = true;
        }
        else if (frame_count == 95 && restored)
        {
            g_screenshot_path = "test1.bmp";
            g_screenshot = true;
        }

        frame();
        glfwPollEvents();

        // wgpu-native is single-threaded: async map callbacks only fire while events are pumped
        wgpuInstanceProcessEvents(g_wgpu.instance);

        if (++frame_count >= 130 && automatic_exit)
            break;
    }

    // The scripted round-trip must have completed, and the window must be back at the original framebuffer
    // size with the surface drained back to optimal.
    if (!resized || !restored)
    {
        fprintf(stderr, "resize: FAIL: scripted round-trip did not complete\n");
        return 1;
    }

    glfwGetFramebufferSize(g_window, &fb_width, &fb_height);
    if ((uint32_t)fb_width != original_framebuffer_width || (uint32_t)fb_height != original_framebuffer_height ||
        g_last_surface_status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal)
    {
        fprintf(stderr, "resize: FAIL: after round-trip framebuffer is %dx%d (expected %ux%u), surface status %d\n",
                fb_width, fb_height, original_framebuffer_width, original_framebuffer_height, (int)g_last_surface_status);
        return 1;
    }

    printf("resize: PASS: round-trip restored %ux%u framebuffer, surface optimal\n", original_framebuffer_width, original_framebuffer_height);

    cleanup();

    // The unit test : both screenshots must match tests/reference.bmp (pixel-identical or PSNR > 45 dB).
    // The demo is deterministic (fixed seed, layout proportional to the framebuffer), so the restored size
    // reproduces the original image : this asserts the full resize round-trip, including the rebuilt bind groups.
    return compare_screenshots("test0.bmp", "tests/reference.bmp") ||
           compare_screenshots("test1.bmp", "tests/reference.bmp");
}

// ---------------------------------------------------------------------------------------------------------------------------
static inline float float_rand(uint32_t *seed)
{
    union
    {
        uint32_t i;
        float f;
    } u;

    // SplitMix32
    *seed += 0x9e3779b9u;
    uint64_t z = *seed;
    z = (z ^ (z >> 15)) * 0x85ebca6bULL;
    z = (z ^ (z >> 13)) * 0xc2b2ae35ULL;
    z ^= z >> 16;

    uint32_t mant = (uint32_t)(z >> 9) & 0x007FFFFFu;
    u.i = mant | 0x3f800000u;
    return u.f - 1.0f;
}

// ---------------------------------------------------------------------------------------------------------------------------
void slot(float width, float height, uint32_t index, float* cx, float* cy, float* radius)
{
    float step_x = width / 8.f;
    float step_y = height / 3.375f;

    *cx = (index%8) * step_x + step_x * .5f;
    *cy = (index/8) * step_y + step_y * .5f;
    *radius = fminf(step_x, step_y) * .4f;
}

const float g_angle = 0.78539816f;

//-----------------------------------------------------------------------------------------------------------------------------
void all_primitives(float width, float height)
{
    float cx, cy, radius;
    char string[256];

    slot(width, height, 0, &cx, &cy, &radius);
    od_draw_disc_gradient(g_renderer, cx, cy, radius, miya_dark_blue, miya_light_blue);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "disc_gradient", miya_brown);

    slot(width, height, 1, &cx, &cy, &radius);
    od_draw_ring(g_renderer, cx, cy, radius, radius * .1f, miya_green);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "ring", miya_brown);

    slot(width, height, 2, &cx, &cy, &radius);
    od_draw_box(g_renderer, cx - radius, cy - radius*.5f, cx + radius, cy + radius*.5f, radius * 0.05f, miya_grey);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "box", miya_brown);

    slot(width, height, 3, &cx, &cy, &radius);
    od_draw_blurred_box(g_renderer, cx, cy, radius*.5f, radius, radius * 0.1f, miya_blue);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "blurred_box", miya_brown);

    slot(width, height, 4, &cx, &cy, &radius);
    od_draw_oriented_rect(g_renderer, cx - cosf(g_angle) * radius, cy - sinf(g_angle) * radius, cx + cosf(g_angle) * radius, cy + sinf(g_angle) * radius,
                          radius * 0.4f, 0.f, radius * 0.1f, miya_pale_blue);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "oriented_rect", miya_brown);

    slot(width, height, 5, &cx, &cy, &radius);
    od_draw_oriented_box(g_renderer, cx + cosf(g_angle) * radius, cy - sinf(g_angle) * radius, cx - cosf(g_angle) * radius, cy + sinf(g_angle) * radius,
                         radius * 0.5f, radius * 0.05f, miya_red);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "oriented_box", miya_brown);

    slot(width, height, 6, &cx, &cy, &radius);
    od_draw_triangle(g_renderer, (float[]){cx, cy, cx - cosf(g_angle) * radius, cy + sinf(g_angle) * radius,
                     cx + cosf(g_angle) * radius, cy +sinf(g_angle) * radius}, radius * 0.1f, miya_dark_green);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "triangle", miya_brown);

    slot(width, height, 7, &cx, &cy, &radius);
    od_draw_triangle_ring(g_renderer, (float[]){cx, cy, cx - cosf(g_angle) * radius, cy - sinf(g_angle) * radius,
                          cx + cosf(g_angle) * radius, cy - sinf(g_angle) * radius}, 0.f, radius * 0.1f, miya_dark_grey);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "triangle_ring", miya_brown);

    slot(width, height, 8, &cx, &cy, &radius);
    od_draw_ellipse(g_renderer, cx + cosf(g_angle) * radius, cy - sinf(g_angle) * radius, cx - cosf(g_angle) * radius, cy + sinf(g_angle) * radius,
                    radius, miya_yellow);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "ellipse", miya_brown);

    slot(width, height, 9, &cx, &cy, &radius);
    od_draw_ellipse_ring(g_renderer, cx + cosf(g_angle) * radius, cy - sinf(g_angle) * radius, cx - cosf(g_angle) * radius, cy + sinf(g_angle) * radius,
                         radius, radius * 0.1f, miya_light_grey);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "ellipse_ring", miya_brown);

    slot(width, height, 10, &cx, &cy, &radius);
    od_draw_sector(g_renderer, cx, cy, radius, g_angle, 0.78539816f, miya_pink);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "sector", miya_brown);

    slot(width, height, 11, &cx, &cy, &radius);
    od_draw_sector_ring(g_renderer, cx, cy, radius, -g_angle, 0.78539816f, radius * 0.1f, miya_dark_blue);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "sector_ring", miya_brown);

    slot(width, height, 12, &cx, &cy, &radius);
    od_draw_arc(g_renderer, cx, cy, cosf(g_angle), sinf(g_angle), 0.78539816f, radius, radius * 0.1f, miya_red);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "arc", miya_brown);

    slot(width, height, 13, &cx, &cy, &radius);
    od_draw_text(g_renderer, cx-radius, cy-radius, "Some text\nABCDEFGHILMNO\nPQRSTUVWYZ\n1234567890!@#$%?&*()\nSphinx of black\n quartz, judge my vow.\n"
                 "!\"#$%&'()*+,-./01234\n56789:;<=>?@\n[\\]^_`abcdefghijklmnop\nqrstuvwxyz{|}~", miya_black);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "text", miya_brown);

    // slot(width, height, 14, &cx, &cy, &radius);
    // od_begin_group(g_renderer, true, radius * 0.25f, radius * 0.05f);
    // od_draw_disc(g_renderer, cx, cy, radius*0.25f, miya_light_green);
    // od_draw_disc(g_renderer, cx + cosf(g_angle) * radius * .5f, cy - sinf(g_angle) * radius * .5f, radius*0.25f, miya_light_green);
    // od_draw_box(g_renderer, cx-radius*.7f, cy-radius*.7f, cx-radius*.3f, cy+radius*.7f, 0.f, miya_yellow);
    // od_end_group(g_renderer, miya_brown);
    // od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "smoothmin", miya_brown);

    slot(width, height, 15, &cx, &cy, &radius);
    od_draw_quad(g_renderer, cx-radius, cy-radius, cx, cy, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 0, 0x7fffffff);
    od_draw_quad(g_renderer, cx, cy-radius, cx+radius, cy, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 1, 0xffffffff);
    od_draw_quad(g_renderer, cx-radius, cy, cx, cy+radius, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 2, 0xffffffff);
    od_draw_quad(g_renderer, cx, cy, cx+radius, cy+radius, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 3, 0xffffffff);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "quad", miya_brown);

    slot(width, height, 16, &cx, &cy, &radius);
    od_draw_oriented_quad(g_renderer, cx, cy, radius, radius*.5f, g_angle * 0.75f, (od_quad_uv){0.f, 0.f, 1.f, 0.5f}, 2, 0xffffffff);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "oriented_quad", miya_brown);

    slot(width, height, 17, &cx, &cy, &radius);
    float quadratic_ctrl_pts[] = {cx, cy-radius*.8f, cx-radius, cy+radius*0.8f, cx, cy+radius};
    uint32_t num_capsules = od_draw_quadratic_bezier(g_renderer, quadratic_ctrl_pts, 20.f, miya_red);
    snprintf(string, 256, "%u capsules", num_capsules);
    for(uint32_t i=0; i<3; i++)
        od_draw_disc(g_renderer, quadratic_ctrl_pts[i*2], quadratic_ctrl_pts[i*2+1], 10.f, miya_yellow);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "quadratic_bezier", miya_brown);
    od_draw_text(g_renderer, cx-radius, cy+radius*1.25f, string, miya_brown);

    slot(width, height, 18, &cx, &cy, &radius);
    float cubic_ctrl_pts[] = {cx, cy-radius*.8f, cx-radius, cy+radius*0.8f, cx, cy+radius, cx+radius*.8f, cy};
    num_capsules = od_draw_cubic_bezier(g_renderer, cubic_ctrl_pts, 20.f, miya_light_blue);
    snprintf(string, 256, "%u capsules", num_capsules);
    for(uint32_t i=0; i<4; i++)
        od_draw_disc(g_renderer, cubic_ctrl_pts[i*2], cubic_ctrl_pts[i*2+1], 10.f, miya_dark_green);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "cubic_bezier", miya_brown);
    od_draw_text(g_renderer, cx-radius, cy+radius*1.25f, string, miya_brown);

    slot(width, height, 19, &cx, &cy, &radius);
    od_begin_group(g_renderer, op_union, 10.f);
    od_draw_box(g_renderer, cx-radius, cy-radius*.25f, cx+radius, cy+radius*.25f, 0.f, miya_blue);
    od_draw_disc(g_renderer, cx-radius*.5f, cy, radius*.3f, miya_dark_green);
    od_draw_sector(g_renderer, cx, cy, radius, g_angle, 0.78539816f, miya_pink);
    od_draw_arc(g_renderer, cx, cy, cosf(g_angle), sinf(g_angle), 0.78539816f, radius, radius * 0.1f, miya_red);
    od_end_group(g_renderer, miya_yellow);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "outline", miya_brown);

    slot(width, height, 20, &cx, &cy, &radius);
    od_set_cliprect(g_renderer, cx-radius, cy-radius, cx+radius, cy+radius);
    uint32_t seed = 0x12345678;
    const uint32_t colors[] = {miya_green, miya_pale_blue, miya_yellow};
    for(uint32_t i=0; i<100; i++)
    {
        float angle = float_rand(&seed) * 6.28f;
        od_draw_capsule(g_renderer, cx, cy, cx + cosf(angle) * 1000.f, cy + sinf(angle) * 1000.f, radius * 0.1f, colors[i%3]);
    }

    od_set_cliprect(g_renderer, 0.f, 0.f, width, height);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "clip_rect", miya_brown);

    slot(width, height, 21, &cx, &cy, &radius);
    od_draw_capsule_gradient(g_renderer, cx - cosf(g_angle) * radius, cy - sinf(g_angle) * radius, cx + cosf(g_angle) * radius, cy + sinf(g_angle) * radius,
                             radius * 0.1f, miya_pale_blue, miya_red);
    od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "capsule_gradient", miya_brown);

    snprintf(string, 256, "GPU Memory usage : %zu kb", g_stats.gpu_memory_usage>>10);
    od_draw_text(g_renderer, 0, height - od_get_text_height(g_renderer) * 2.f, string, miya_blue);

    snprintf(string, 256, "num commands : %u", g_stats.num_draw_cmd);
    od_draw_text(g_renderer, (width - od_get_text_width(g_renderer, string)),
                            height - od_get_text_height(g_renderer) * 2.f, string, miya_blue);
}