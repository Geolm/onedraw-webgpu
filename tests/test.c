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

struct onedraw* renderer;
struct webgpu_platform wgpu;
struct GLFWwindow* window;

void all_primitives(float width, float height);

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

    od_set_clear_color(renderer, miya_white);
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

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    od_begin_frame(renderer);
    all_primitives((float)width, (float)height);
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

// ---------------------------------------------------------------------------------------------------------------------------
void slot(float width, float height, uint32_t index, float* cx, float* cy, float* radius)
{
    float step_x = width / 8.f;
    float step_y = height / 3.375f;

    *cx = (index%8) * step_x + step_x * .5f;
    *cy = (index/8) * step_y + step_y * .5f;
    *radius = fminf(step_x, step_y) * .4f;
}

const float angle = 0.78539816f;

//-----------------------------------------------------------------------------------------------------------------------------
void all_primitives(float width, float height)
{
    float cx, cy, radius;
    char string[256];

    slot(width, height, 0, &cx, &cy, &radius);
    od_draw_disc_gradient(renderer, cx, cy, radius, miya_dark_blue, miya_light_blue);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "disc_gradient", miya_brown);

    slot(width, height, 1, &cx, &cy, &radius);
    od_draw_ring(renderer, cx, cy, radius, radius * .1f, miya_green);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "ring", miya_brown);

    slot(width, height, 2, &cx, &cy, &radius);
    od_draw_box(renderer, cx - radius, cy - radius*.5f, cx + radius, cy + radius*.5f, radius * 0.05f, miya_grey);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "box", miya_brown);

    slot(width, height, 3, &cx, &cy, &radius);
    od_draw_blurred_box(renderer, cx, cy, radius*.5f, radius, radius * 0.1f, miya_blue);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "blurred_box", miya_brown);

    slot(width, height, 4, &cx, &cy, &radius);
    od_draw_oriented_rect(renderer, cx - cosf(angle) * radius, cy - sinf(angle) * radius, cx + cosf(angle) * radius, cy + sinf(angle) * radius,
                          radius * 0.4f, 0.f, radius * 0.1f, miya_pale_blue);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "oriented_rect", miya_brown);

    slot(width, height, 5, &cx, &cy, &radius);
    od_draw_oriented_box(renderer, cx + cosf(angle) * radius, cy - sinf(angle) * radius, cx - cosf(angle) * radius, cy + sinf(angle) * radius,
                         radius * 0.5f, radius * 0.05f, miya_red);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "oriented_box", miya_brown);

    slot(width, height, 6, &cx, &cy, &radius);
    od_draw_triangle(renderer, (float[]){cx, cy, cx - cosf(angle) * radius, cy + sinf(angle) * radius,
                     cx + cosf(angle) * radius, cy +sinf(angle) * radius}, radius * 0.1f, miya_dark_green);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "triangle", miya_brown);

    slot(width, height, 7, &cx, &cy, &radius);
    od_draw_triangle_ring(renderer, (float[]){cx, cy, cx - cosf(angle) * radius, cy - sinf(angle) * radius,
                          cx + cosf(angle) * radius, cy - sinf(angle) * radius}, 0.f, radius * 0.1f, miya_dark_grey);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "triangle_ring", miya_brown);

    slot(width, height, 8, &cx, &cy, &radius);
    od_draw_ellipse(renderer, cx + cosf(angle) * radius, cy - sinf(angle) * radius, cx - cosf(angle) * radius, cy + sinf(angle) * radius,
                    radius, miya_yellow);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "ellipse", miya_brown);

    slot(width, height, 9, &cx, &cy, &radius);
    od_draw_ellipse_ring(renderer, cx + cosf(angle) * radius, cy - sinf(angle) * radius, cx - cosf(angle) * radius, cy + sinf(angle) * radius,
                         radius, radius * 0.1f, miya_light_grey);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "ellipse_ring", miya_brown);

    slot(width, height, 10, &cx, &cy, &radius);
    od_draw_sector(renderer, cx, cy, radius, angle, 0.78539816f, miya_pink);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "sector", miya_brown);

    slot(width, height, 11, &cx, &cy, &radius);
    od_draw_sector_ring(renderer, cx, cy, radius, -angle, 0.78539816f, radius * 0.1f, miya_dark_blue);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "sector_ring", miya_brown);

    slot(width, height, 12, &cx, &cy, &radius);
    od_draw_arc(renderer, cx, cy, cosf(angle), sinf(angle), 0.78539816f, radius, radius * 0.1f, miya_red);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "arc", miya_brown);

    slot(width, height, 13, &cx, &cy, &radius);
    od_draw_text(renderer, cx-radius, cy-radius, "Some text\nABCDEFGHILMNO\nPQRSTUVWYZ\n1234567890!@#$%?&*()\nSphinx of black\n quartz, judge my vow.\n"
                 "!\"#$%&'()*+,-./01234\n56789:;<=>?@\n[\\]^_`abcdefghijklmnop\nqrstuvwxyz{|}~", miya_black);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "text", miya_brown);

    // slot(width, height, 14, &cx, &cy, &radius);
    // od_begin_group(renderer, true, radius * 0.25f, radius * 0.05f);
    // od_draw_disc(renderer, cx, cy, radius*0.25f, miya_light_green);
    // od_draw_disc(renderer, cx + cosf(angle) * radius * .5f, cy - sinf(angle) * radius * .5f, radius*0.25f, miya_light_green);
    // od_draw_box(renderer, cx-radius*.7f, cy-radius*.7f, cx-radius*.3f, cy+radius*.7f, 0.f, miya_yellow);
    // od_end_group(renderer, miya_brown);
    // od_draw_text(renderer, cx-radius, cy-radius*1.25f, "smoothmin", miya_brown);

    // slot(width, height, 15, &cx, &cy, &radius);
    // od_draw_quad(renderer, cx-radius, cy-radius, cx, cy, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 0, 0x7fffffff);
    // od_draw_quad(renderer, cx, cy-radius, cx+radius, cy, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 1, 0xffffffff);
    // od_draw_quad(renderer, cx-radius, cy, cx, cy+radius, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 2, 0xffffffff);
    // od_draw_quad(renderer, cx, cy, cx+radius, cy+radius, (od_quad_uv){0.f, 0.f, 1.f, 1.f}, 3, 0xffffffff);
    // od_draw_text(renderer, cx-radius, cy-radius*1.25f, "quad", miya_brown);

    // slot(width, height, 16, &cx, &cy, &radius);
    // od_draw_oriented_quad(renderer, cx, cy, radius, radius*.5f, angle * 0.75f, (od_quad_uv){0.f, 0.f, 1.f, 0.5f}, 2, 0xffffffff);
    // od_draw_text(renderer, cx-radius, cy-radius*1.25f, "oriented_quad", miya_brown);

    slot(width, height, 17, &cx, &cy, &radius);
    float quadratic_ctrl_pts[] = {cx, cy-radius*.8f, cx-radius, cy+radius*0.8f, cx, cy+radius};
    uint32_t num_capsules = od_draw_quadratic_bezier(renderer, quadratic_ctrl_pts, 20.f, miya_red);
    snprintf(string, 256, "%u capsules", num_capsules);
    for(uint32_t i=0; i<3; i++)
        od_draw_disc(renderer, quadratic_ctrl_pts[i*2], quadratic_ctrl_pts[i*2+1], 10.f, miya_yellow);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "quadratic_bezier", miya_brown);
    od_draw_text(renderer, cx-radius, cy+radius*1.25f, string, miya_brown);

    slot(width, height, 18, &cx, &cy, &radius);
    float cubic_ctrl_pts[] = {cx, cy-radius*.8f, cx-radius, cy+radius*0.8f, cx, cy+radius, cx+radius*.8f, cy};
    num_capsules = od_draw_cubic_bezier(renderer, cubic_ctrl_pts, 20.f, miya_light_blue);
    snprintf(string, 256, "%u capsules", num_capsules);
    for(uint32_t i=0; i<4; i++)
        od_draw_disc(renderer, cubic_ctrl_pts[i*2], cubic_ctrl_pts[i*2+1], 10.f, miya_dark_green);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "cubic_bezier", miya_brown);
    od_draw_text(renderer, cx-radius, cy+radius*1.25f, string, miya_brown);

    // slot(width, height, 19, &cx, &cy, &radius);
    // od_begin_group(renderer, false, 0.f, 10.f);
    // od_draw_box(renderer, cx-radius, cy-radius*.25f, cx+radius, cy+radius*.25f, 0.f, miya_blue);
    // od_draw_disc(renderer, cx-radius*.5f, cy, radius*.3f, miya_dark_green);
    // od_draw_sector(renderer, cx, cy, radius, angle, 0.78539816f, miya_pink);
    // od_draw_arc(renderer, cx, cy, cosf(angle), sinf(angle), 0.78539816f, radius, radius * 0.1f, miya_red);
    // od_end_group(renderer, miya_yellow);
    // od_draw_text(renderer, cx-radius, cy-radius*1.25f, "outline", miya_brown);

    // slot(width, height, 20, &cx, &cy, &radius);
    // od_set_clipdisc(renderer, cx, cy, radius);
    // int seed = 0x12345678;
    // const uint32_t colors[] = {miya_green, miya_pale_blue, miya_yellow};
    // for(uint32_t i=0; i<100; i++)
    // {
    //     float angle = iq_random_float(&seed) * 6.28f;
    //     od_draw_capsule(renderer, cx, cy, cx + cosf(angle) * 1000.f, cy + sinf(angle) * 1000.f, radius * 0.1f, colors[i%3]);
    // }

    // od_set_cliprect(renderer, 0.f, 0.f, UINT16_MAX, UINT16_MAX);
    // od_draw_text(renderer, cx-radius, cy-radius*1.25f, "disc clip", miya_brown);

    slot(width, height, 21, &cx, &cy, &radius);
    od_draw_capsule_gradient(renderer, cx - cosf(angle) * radius, cy - sinf(angle) * radius, cx + cosf(angle) * radius, cy + sinf(angle) * radius,
                             radius * 0.1f, miya_pale_blue, miya_red);
    od_draw_text(renderer, cx-radius, cy-radius*1.25f, "capsule_gradient", miya_brown);


    // od_stats stats;
    // od_get_stats(renderer, &stats);
    // snprintf(string, 256, "GPU Memory usage : %zu kb", stats.gpu_memory_usage>>10);
    // od_draw_text(renderer, 0, sapp_heightf() - od_text_height(renderer) * 2.f, string, miya_blue);

    // snprintf(string, 256, "num commands : %u", stats.peak_num_draw_cmd);
    // od_draw_text(renderer, (sapp_widthf() - od_text_width(renderer, string)) * .5f,
    //              sapp_heightf() - od_text_height(renderer) * 2.f, string, miya_blue);
}