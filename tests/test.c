#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

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

struct onedraw* g_renderer;
struct webgpu_platform g_wgpu;
struct GLFWwindow* g_window;

void all_primitives(float width, float height);

//-----------------------------------------------------------------------------------------------------------------------------
void init(void)
{
    int width, height;
    glfwGetFramebufferSize(g_window, &width, &height);

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
            .height = 256,
            .width = 256,
            .num_slices = 4
        }
    });

    od_set_clear_color(g_renderer, miya_white);
}

//-----------------------------------------------------------------------------------------------------------------------------
void frame(void)
{
    WGPUSurfaceTexture surfaceTexture;
    wgpuSurfaceGetCurrentTexture(g_wgpu.surface, &surfaceTexture);

    if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
        return;

    WGPUTextureView frame = wgpuTextureCreateView(surfaceTexture.texture, NULL);

    int width, height;
    glfwGetFramebufferSize(g_window, &width, &height);

    od_begin_frame(g_renderer);
    all_primitives((float)width, (float)height);
    od_end_frame(g_renderer, frame);

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

    g_window = glfwCreateWindow(1280, 720, "onedraw-webgpu", NULL, NULL);
    assert(g_window != NULL);

    glfwSetKeyCallback(g_window, key_cb);

    init();

    while (!glfwWindowShouldClose(g_window))
    {
        frame();
        glfwPollEvents();
    }

    cleanup();

    return 0;
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

    // slot(width, height, 15, &cx, &cy, &radius);
    // od_draw_quad(g_renderer, cx-radius, cy-radius, cx, cy, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 0, 0x7fffffff);
    // od_draw_quad(g_renderer, cx, cy-radius, cx+radius, cy, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 1, 0xffffffff);
    // od_draw_quad(g_renderer, cx-radius, cy, cx, cy+radius, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 2, 0xffffffff);
    // od_draw_quad(g_renderer, cx, cy, cx+radius, cy+radius, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 3, 0xffffffff);
    // od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "quad", miya_brown);

    // slot(width, height, 16, &cx, &cy, &radius);
    // od_draw_oriented_quad(g_renderer, cx, cy, radius, radius*.5f, g_angle * 0.75f, (od_quad_uv){0.f, 0.f, 1.f, 0.5f}, 2, 0xffffffff);
    // od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "oriented_quad", miya_brown);

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

    // slot(width, height, 19, &cx, &cy, &radius);
    // od_begin_group(g_renderer, false, 0.f, 10.f);
    // od_draw_box(g_renderer, cx-radius, cy-radius*.25f, cx+radius, cy+radius*.25f, 0.f, miya_blue);
    // od_draw_disc(g_renderer, cx-radius*.5f, cy, radius*.3f, miya_dark_green);
    // od_draw_sector(g_renderer, cx, cy, radius, g_angle, 0.78539816f, miya_pink);
    // od_draw_arc(g_renderer, cx, cy, cosf(g_angle), sinf(g_angle), 0.78539816f, radius, radius * 0.1f, miya_red);
    // od_end_group(g_renderer, miya_yellow);
    // od_draw_text(g_renderer, cx-radius, cy-radius*1.25f, "outline", miya_brown);

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

    od_stats stats;
    od_get_stats(g_renderer, &stats);
    snprintf(string, 256, "GPU Memory usage : %zu kb", stats.gpu_memory_usage>>10);
    od_draw_text(g_renderer, 0, height - od_get_text_height(g_renderer) * 2.f, string, miya_blue);

    snprintf(string, 256, "num commands : %u", stats.peak_num_draw_cmd);
    od_draw_text(g_renderer, (width - od_get_text_width(g_renderer, string)),
                            height - od_get_text_height(g_renderer) * 2.f, string, miya_blue);
}