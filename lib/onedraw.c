/*

    onedraw-webgpu — a GPU-driven 2D renderer

    Project URL : https://github.com/Geolm/onedraw-webgpu

    zlib License

    (C) 2026 Geolm

    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.

*/

#include "onedraw.h"
#include <stdarg.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "default_font.h"
#include "default_font_atlas.h"
#include "rasterizer.h"
#include "binning.h"

// ---------------------------------------------------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------------------------------------------------

#define STRING_BUFFER_SIZE (512U)
#define VEC2_SQR2 (1.41421356237f)
#define HALF_PIXEL (.5f)
#define VEC2_PI (3.14159265f)
#define TESSELATION_STACK_MAX (1024U)
#define COLINEAR_THRESHOLD (.1f)
#define FRAME_COUNT (3)
#define TILE_SIZE (16)

// ---------------------------------------------------------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------------------------------------------------------

#define assert_msg(expr, msg) assert((expr) && (msg))
#define UNUSED_VARIABLE(a) (void)(a)
#define SAFE_RELEASE(x, fn) if (x) { fn(x); x = NULL; }
#if defined(_MSC_VER)
#define OD_STATIC_ASSERT(cond, msg) assert_msg(cond, msg)
#else
#define OD_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

// ---------------------------------------------------------------------------------------------------------------------------
// Private structures
// ---------------------------------------------------------------------------------------------------------------------------

typedef struct {float x, y, z, w;} float4;
typedef uint32_t quantized_aabb;
typedef enum sdf_operator {op_additive, op_subtractive} sdf_operator;

typedef struct dynamic_buffer
{
    WGPUBuffer buffers[FRAME_COUNT];
} dynamic_buffer;

struct onedraw
{
    WGPUDevice device;
    WGPUQueue queue;
    WGPUCommandBuffer command_buffer;

    struct
    {
        dynamic_buffer buffer;
        dynamic_buffer colors;
        dynamic_buffer aabb_buffer;
        dynamic_buffer data_buffer;
        dynamic_buffer clipshapes_buffer;
        uint32_t count;
        quantized_aabb* group_aabb;
    } commands;

    // region binning
    struct
    {
        uint16_t num_width;
        uint16_t num_height;
        uint16_t count;
        uint32_t num_groups;
    } regions;

    // tile binning
    struct 
    {
        WGPUBuffer head; 
        WGPUComputePipeline binning_pso;
        WGPUComputePipeline write_indirect_buffer_pso;
        WGPUBuffer counters;
        WGPUBuffer indirect_arg;
        WGPUBuffer indices;
        WGPUBuffer nodes;
        WGPUBuffer indirect_draw_params;
        uint32_t num_width;
        uint32_t num_height;
        uint32_t count;
        bool culling_debug;
    } tiles;

    // rasterizer
    struct
    {
        WGPURenderPipeline pso;
        WGPUDepthStencilState depth_stencil_state;
        WGPUTexture atlas;
        float4 clear_color;
        uint32_t width;
        uint32_t height;
        float aa_width;
        sdf_operator group_op;
        float outline_width;
        bool srgb_backbuffer;
        bool clear_backbuffer;
    } rasterizer;

    // font
    struct
    {
        WGPUTexture texture;
        WGPUTextureView view;
        WGPUBuffer glyphs;
        od_font desc;
    } font;

    // screenshot service
    struct
    {
        WGPUTexture texture;
        void* out_pixels;
        uint32_t region_x, region_y;
        uint32_t region_width, region_height;
        bool show_region;
        bool capture_image;
        bool allocate_resources;
    } screenshot;

    // stats
    struct
    {
        uint32_t peak_num_draw_cmd;
        uint32_t num_draw_data;
        uint32_t frame_index;
    } stats;

    struct
    {
        WGPUBindGroup static_group;
        WGPUBindGroup dynamic_group[FRAME_COUNT];
        WGPUBindGroupLayout rasterizer_layout;
        WGPUBindGroupLayout binning_layout;
        WGPUBindGroupLayout frame_layout;
    } binding;

    void (*custom_log)(const char* string);
    char string_buffer[STRING_BUFFER_SIZE];
};

typedef struct vec2 {float x, y;} vec2;
typedef struct aabb {vec2 min, max;} aabb;
typedef struct quadratic_bezier {vec2 c0, c1, c2;} quadratic_bezier;
typedef struct cubic_bezier {vec2 c0, c1, c2, c3;} cubic_bezier;

// ---------------------------------------------------------------------------------------------------------------------------
// private functions
// ---------------------------------------------------------------------------------------------------------------------------

static inline float float_max(float a, float b) {return (a>b) ? a : b;}
static inline float float_clamp(float f, float a, float b) {if (f<a) return a; if (f>b) return b; return f;}
static inline vec2 vec2_splat(float value) {return (vec2) {value, value};}
static inline vec2 vec2_set(float x, float y) {return (vec2) {x, y};}
static inline vec2 vec2_add(vec2 a, vec2 b) {return (vec2) {a.x + b.x, a.y + b.y};}
static inline vec2 vec2_sub(vec2 a, vec2 b) {return (vec2) {a.x - b.x, a.y - b.y};}
static inline vec2 vec2_min(vec2 v, vec2 op) {return (vec2) {.x = (v.x < op.x) ? v.x : op.x, .y = (v.y < op.y) ? v.y : op.y};}
static inline vec2 vec2_min3(vec2 a, vec2 b, vec2 c) {return vec2_min(a, vec2_min(b, c));}
static inline vec2 vec2_min4(vec2 a, vec2 b, vec2 c, vec2 d) {return vec2_min(a, vec2_min3(b, c, d));}
static inline vec2 vec2_max(vec2 v, vec2 op) {return (vec2) {.x = (v.x > op.x) ? v.x : op.x, .y = (v.y > op.y) ? v.y : op.y};}
static inline vec2 vec2_max3(vec2 a, vec2 b, vec2 c) {return vec2_max(a, vec2_max(b, c));}
static inline vec2 vec2_max4(vec2 a, vec2 b, vec2 c, vec2 d) {return vec2_max(a, vec2_max3(b, c, d));}
static inline vec2 vec2_skew(vec2 v) {return (vec2) {-v.y, v.x};}
static inline vec2 vec2_scale(vec2 a, float f) {return (vec2) {a.x * f, a.y * f};}
static inline float vec2_dot(vec2 a, vec2 b) {return fmaf(a.x, b.x, a.y * b.y);}
static inline float vec2_sq_length(vec2 v) {return vec2_dot(v, v);}
static inline float vec2_length(vec2 v) {return sqrtf(vec2_sq_length(v));}
static inline bool vec2_similar(vec2 a, vec2 b, float epsilon) {return (fabsf(a.x - b.x) < epsilon) && (fabsf(a.y - b.y) < epsilon);}
static inline vec2 vec2_direction(float angle) {return (vec2) {cosf(angle), sinf(angle)};}
static inline float vec2_distance(vec2 a, vec2 b) {return vec2_length(vec2_sub(b, a));}

//----------------------------------------------------------------------------------------------------------------------------
static inline float vec2_normalize(vec2* v)
{
    float norm = vec2_length(*v);
    if (norm <= FLT_EPSILON)
        return 0.f;

    *v = vec2_scale(*v, 1.f / norm);
    return norm;
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------
static inline vec2 vec2_lerp(vec2 a, vec2 b, float t) 
{
    float one_minus_t = 1.f - t;
    return (vec2) {.x = fmaf(a.x , one_minus_t, b.x * t), .y = fmaf(a.y , one_minus_t, b.y * t)};
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------
static inline bool is_colinear(vec2 p0, vec2 p1, vec2 p2, float threshold)
{
    vec2 v0 = vec2_sub(p1, p0);
    vec2 v1 = vec2_sub(p2, p0);
    float squared_area = fabsf(v0.x * v1.y - v0.y * v1.x);
    float base2 = vec2_dot(v0, v0);

    if (base2 < FLT_EPSILON)
        return true;

    float squared_height = (squared_area * squared_area) / base2;
    return squared_height <= (threshold * threshold);
}

//----------------------------------------------------------------------------------------------------------------------------
static inline void aabb_grow(aabb* box, vec2 amount)
{
    box->min = vec2_sub(box->min, amount);
    box->max = vec2_add(box->max, amount);
}

//----------------------------------------------------------------------------------------------------------------------------
static inline aabb aabb_from_circle(vec2 center, float radius)
{
    return (aabb)
    {
        .min = vec2_sub(center, vec2_splat(radius)),
        .max = vec2_add(center, vec2_splat(radius))
    };
}

//----------------------------------------------------------------------------------------------------------------------------
static inline aabb aabb_from_triangle(vec2 v0, vec2 v1, vec2 v2)
{
    return (aabb)
    {
        .min = vec2_min3(v0, v1, v2),
        .max = vec2_max3(v0, v1, v2)
    };
}

//----------------------------------------------------------------------------------------------------------------------------
static inline aabb aabb_from_rounded_obb(vec2 p0, vec2 p1, float width, float border)
{
    aabb box;

    vec2 dir = vec2_sub(p1, p0);
    vec2_normalize(&dir);
    vec2 normal = vec2_skew(dir);

    normal = vec2_scale(normal, width*.5f + border);
    dir = vec2_scale(dir, border);
    p0 = vec2_sub(p0, dir);
    p1 = vec2_add(p1, dir);

    vec2 vertices[4];
    vertices[0] = vec2_add(p0, normal);
    vertices[1] = vec2_sub(p0, normal);
    vertices[2] = vec2_sub(p1, normal);
    vertices[3] = vec2_add(p1, normal);

    box.min = vec2_min4(vertices[0], vertices[1], vertices[2], vertices[3]);
    box.max = vec2_max4(vertices[0], vertices[1], vertices[2], vertices[3]);

    return box;
}

//----------------------------------------------------------------------------------------------------------------------------
static inline float srgb_to_linear(float c)
{
    if (c <= 0.04045f)
        return c / 12.92f;
    else
        return powf((c + 0.055f) / 1.055f, 2.4f);
}

//----------------------------------------------------------------------------------------------------------------------------
static inline float bitcast_u32_to_float(uint32_t value)
{
    union {float f; uint32_t u;} c;
    c.u = value;
    return c.f;
}

//----------------------------------------------------------------------------------------------------------------------------
void od_log(struct onedraw* r, const char* string, ...)
{
    if (r->custom_log != NULL)
    {
        va_list args;
        va_start(args, string);
        vsnprintf(r->string_buffer, STRING_BUFFER_SIZE, string, args);
        va_end(args);

        r->custom_log(r->string_buffer);
    }
}

//-----------------------------------------------------------------------------------------------------------------------------
size_t od_min_memory_size(void)
{
    return sizeof(struct onedraw);
}

//-----------------------------------------------------------------------------------------------------------------------------
void build_binding(struct onedraw* r)
{
    WGPUBindGroupLayoutEntry rasterizer_layout_entries[] = 
    {
        {   // tile_node
            .binding = 0,
            .visibility = WGPUShaderStage_Fragment,
            .buffer =
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        {   // tile_indices
            .binding = 1,
            .visibility = WGPUShaderStage_Vertex,
            .buffer =
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        {   // counters
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .buffer =
            {
                .type = WGPUBufferBindingType_Storage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        {   // glyphs
            .binding = 3,
            .visibility = WGPUShaderStage_Fragment,
            .buffer =
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
    };

    r->binding.rasterizer_layout = wgpuDeviceCreateBindGroupLayout(r->device, &(WGPUBindGroupLayoutDescriptor)
    {
        .nextInChain = NULL,
        .label = 
        {
            .data = "rasterizer_layout",
            .length = 17
        },
        .entries = rasterizer_layout_entries,
        .entryCount = 4
    });

    WGPUBindGroupLayoutEntry frame_layout_entries[] =
    {
        { // g_draw_args (uniform)
            .binding = 0,
            .visibility = WGPUShaderStage_Compute | WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
            .buffer = 
            {
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        {
            // g_commands
            .binding = 1,
            .visibility = WGPUShaderStage_Compute | WGPUShaderStage_Fragment,
            .buffer = 
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        { // g_quantized_aabb
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .buffer = 
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        { // g_draw_data
            .binding = 3,
            .visibility = WGPUShaderStage_Compute | WGPUShaderStage_Fragment,
            .buffer = 
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        { // g_clips
            .binding = 4,
            .visibility = WGPUShaderStage_Compute | WGPUShaderStage_Fragment,
            .buffer = 
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        { // g_colors
            .binding = 5,
            .visibility = WGPUShaderStage_Fragment,
            .buffer = 
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
    };

    r->binding.frame_layout = wgpuDeviceCreateBindGroupLayout(r->device, &(WGPUBindGroupLayoutDescriptor)
    {
        .nextInChain = NULL,
        .label = 
        {
            .data = "frame_layout",
            .length = 12
        },
        .entries = frame_layout_entries,
        .entryCount = 6
    });

    WGPUBindGroupLayoutEntry binning_layout_entries[] = 
    {
        {   // tile_node
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer =
            {
                .type = WGPUBufferBindingType_Storage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        {   // tile_indices
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer =
            {
                .type = WGPUBufferBindingType_Storage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        {   // counters
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .buffer =
            {
                .type = WGPUBufferBindingType_Storage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        {   // glyphs
            .binding = 3,
            .visibility = WGPUShaderStage_Fragment,
            .buffer =
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
    };

    r->binding.binning_layout = wgpuDeviceCreateBindGroupLayout(r->device, &(WGPUBindGroupLayoutDescriptor)
    {
        .nextInChain = NULL,
        .label = 
        {
            .data = "binning_layout",
            .length = 14
        },
        .entries = binning_layout_entries,
        .entryCount = 4
    });

//     WGPUBindGroupEntry entries[4] = 
//     {
//         {
//             .binding = 0,
//             .buffer = r->tiles.nodes,
//             .offset = 0,
//             .size = WGPU_WHOLE_SIZE,
//         },
//         {
//             .binding = 1,
//             .buffer = r->tiles.indices,
//             .offset = 0,
//             .size = WGPU_WHOLE_SIZE,
//         },
//         {
//             .binding = 2,
//             .buffer = r->tiles.counters,
//             .offset = 0,
//             .size = WGPU_WHOLE_SIZE,
//         },
//         {
//             .binding = 3,
//             .buffer = r->font.glyphs,
//             .offset = 0,
//             .size = WGPU_WHOLE_SIZE,
//         },
//     };

//     WGPUBindGroupDescriptor bind_group_desc = 
//     {
//     .nextInChain = NULL,
//     .label = "group0_static_gpu",
//     .layout = bind_group_layout0,
//     .entryCount = 4,
//     .entries = entries,
// };

    //wgpuDeviceCreateBindGroup()
}

//-----------------------------------------------------------------------------------------------------------------------------
void build_pso(struct onedraw* r, WGPUTextureFormat surface_format)
{
    WGPUShaderSourceWGSL wgsl = 
    {
        .chain = {.next = NULL, .sType = WGPUSType_ShaderSourceWGSL},
        .code = {.data = rasterizer_shader, .length = rasterizer_shader_size}
    };


    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(r->device, &(WGPUShaderModuleDescriptor)
    {
        .nextInChain = &wgsl.chain,
        .label = 
        {
            .data = "rasterizer",
            .length = 11
        }
    });


    WGPUPipelineLayout layout =
        wgpuDeviceCreatePipelineLayout(r->device, &(WGPUPipelineLayoutDescriptor)
        {
            .bindGroupLayoutCount = 2,
            .bindGroupLayouts = (WGPUBindGroupLayout[]) {r->binding.rasterizer_layout, r->binding.frame_layout}
        });

    WGPUColorTargetState color_target = 
    {
        .format = surface_format,
        .writeMask = WGPUColorWriteMask_All
    };

    WGPUFragmentState fragment = 
    {
        .module = shader,
        .entryPoint = (WGPUStringView){ "tile_fs", 7 },
        .targetCount = 1,
        .targets = &color_target
    };

    WGPURenderPipelineDescriptor desc = 
    {
        .layout = layout,
        .vertex = 
        {
            .module = shader,
            .entryPoint = (WGPUStringView){ "tile_vs", 7 },
            .bufferCount = 0,
            .buffers = NULL
        },

        .primitive = 
        {
            .topology = WGPUPrimitiveTopology_TriangleList,
            .frontFace = WGPUFrontFace_CCW,
            .cullMode = WGPUCullMode_None
        },

        .fragment = &fragment,
        .multisample = {.count = 1, .mask = ~0u}
    };

    r->rasterizer.pso = wgpuDeviceCreateRenderPipeline(r->device, &desc);
}

//-----------------------------------------------------------------------------------------------------------------------------
void build_font(struct onedraw* r)
{
    // font description is generated by builder 
    OD_STATIC_ASSERT(sizeof(od_font) == default_font_size, "size of od_font structure mismatch");
    memcpy(&r->font.desc, default_font, default_font_size);

    WGPUTextureDescriptor desc = 
    {
        .dimension = WGPUTextureDimension_2D,
        .size = 
        {
            .width = 256,
            .height = 256,
            .depthOrArrayLayers = 1
        },
        .sampleCount = 1,
        .mipLevelCount = 1,
        .format = WGPUTextureFormat_R8Unorm,
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
    };

    r->font.texture = wgpuDeviceCreateTexture(r->device, &desc);
    assert_msg(r->font.texture != NULL, "default font texture creation failed");

    r->font.view = wgpuTextureCreateView(r->font.texture, NULL);
    assert_msg(r->font.view != NULL, "default font texture view creation failed");

    WGPUTexelCopyTextureInfo copy_info = 
    {
        .mipLevel = 0,
        .texture = r->font.texture,
        .aspect = WGPUTextureAspect_All
    };

    WGPUTexelCopyBufferLayout layout = 
    {
        .bytesPerRow = 256,
        .offset = 0,
        .rowsPerImage = 256
    };

    WGPUExtent3D extent = 
    {
        .width = 256,
        .height = 256,
        .depthOrArrayLayers = 1
    };

    wgpuQueueWriteTexture(r->queue, &copy_info, default_font_atlas, default_font_atlas_size, &layout, &extent);
}

// ---------------------------------------------------------------------------------------------------------------------------
// public functions
// ---------------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------------
struct onedraw* od_init(onedraw_def* def)
{
    assert_msg(def->preallocated_buffer != NULL, "forgot to allocate memory?");
    assert_msg(((uintptr_t)def->preallocated_buffer)%sizeof(uintptr_t) == 0, "preallocated_buffer must be aligned on sizeof(uintptr_t)");
    assert(def->device != NULL);

    struct onedraw* r = (struct onedraw*) def->preallocated_buffer;

    // clear everything to zero
    *r = (struct onedraw) {0};

    r->custom_log = def->log_func;
    r->device = def->device;
    r->rasterizer.srgb_backbuffer = def->srgb_backbuffer;
    r->rasterizer.aa_width = VEC2_SQR2;
    r->rasterizer.clear_backbuffer = true;
    r->rasterizer.clear_color = (float4) {.x = 0.f, .y = 0.f, .z = 0.f, .w = 1.f};
    r->queue = wgpuDeviceGetQueue(r->device);

    assert_msg(r->queue != NULL, "cannot create queue from device");

    build_binding(r);
    build_pso(r, def->surface_format);
    build_font(r);
    // od_build_depthstencil_state(r);
    od_resize(r, def->viewport_width, def->viewport_height);

    // if (def->atlas.width != 0)
    //     od_create_atlas(r, def->atlas.width, def->atlas.height, def->atlas.num_slices);

    return r;
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_upload_slice(struct onedraw* r, const void* pixel_data, uint32_t slice_index)
{
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(pixel_data);
    UNUSED_VARIABLE(slice_index);
    assert_msg(0, "not yet implemented");
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_set_capture_region(struct onedraw* r, uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(y);
    UNUSED_VARIABLE(width);
    UNUSED_VARIABLE(height);
    assert_msg(0, "not yet implemented");
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_get_capture_region_dimensions(struct onedraw *r, uint32_t* width, uint32_t* height)
{
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(width);
    UNUSED_VARIABLE(height);
    assert_msg(0, "not yet implemented");
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_take_screenshot(struct onedraw* r, void* out_pixels)
{
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(out_pixels);
    assert_msg(0, "not yet implemented");
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_resize(struct onedraw* r, uint32_t width, uint32_t height)
{
    if (width != r->rasterizer.width || height != r->rasterizer.height)
    {
        od_log(r, "resizing the framebuffer to %dx%d", width, height);

        r->rasterizer.width = width;
        r->rasterizer.height = height;
        r->tiles.num_width = (width + TILE_SIZE - 1) / TILE_SIZE;
        r->tiles.num_height = (height + TILE_SIZE - 1) / TILE_SIZE;
        r->tiles.count = r->tiles.num_width * r->tiles.num_height;

        // TODO : add region support

        SAFE_RELEASE(r->tiles.head, wgpuBufferRelease);
        r->tiles.head = wgpuDeviceCreateBuffer(r->device, &(WGPUBufferDescriptor)
        {
            .size = r->tiles.count * sizeof(uint32_t),
            .usage = WGPUBufferUsage_Storage
        });

        SAFE_RELEASE(r->tiles.indices, wgpuBufferRelease);
        r->tiles.indices = wgpuDeviceCreateBuffer(r->device, &(WGPUBufferDescriptor)
        {
            .size = r->tiles.num_width * r->tiles.num_height * sizeof(uint32_t),
            .usage = WGPUBufferUsage_Storage
        });

        od_log(r, "%ux%u tiles", r->tiles.num_width, r->tiles.num_height);

        // TODO : screenshot resources
    }
    
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_begin_frame(struct onedraw* r)
{
    r->stats.frame_index++;
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_end_frame(struct onedraw* r, WGPUTextureView target_view)
{
    WGPUCommandEncoderDescriptor encoder_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(r->device, &encoder_desc);

    WGPURenderPassColorAttachment color_attachment = {0};
    color_attachment.view = target_view;
    color_attachment.resolveTarget = NULL;
    color_attachment.loadOp = WGPULoadOp_Clear;
    color_attachment.storeOp = WGPUStoreOp_Store;
    color_attachment.clearValue = (WGPUColor){r->rasterizer.clear_color.x, r->rasterizer.clear_color.y, r->rasterizer.clear_color.z, r->rasterizer.clear_color.w};
    color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

    WGPURenderPassDescriptor render_pass_desc = {0};
    render_pass_desc.colorAttachmentCount = 1;
    render_pass_desc.colorAttachments = &color_attachment;
    render_pass_desc.depthStencilAttachment = NULL;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_desc);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    WGPUCommandBufferDescriptor cmd_buffer_desc = {0};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_buffer_desc);

    wgpuCommandEncoderRelease(encoder);

    WGPUQueue queue = wgpuDeviceGetQueue(r->device);
    wgpuQueueSubmit(queue, 1, &cmd);

    wgpuCommandBufferRelease(cmd);
}

//-----------------------------------------------------------------------------------------------------------------------------
float od_get_gputime(struct onedraw* r)
{
    UNUSED_VARIABLE(r);
    assert(false);
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------
void od_terminate(struct onedraw* r)
{
    SAFE_RELEASE(r->font.view, wgpuTextureViewRelease);
    SAFE_RELEASE(r->font.texture, wgpuTextureRelease);
    SAFE_RELEASE(r->binding.rasterizer_layout, wgpuBindGroupLayoutRelease);
    SAFE_RELEASE(r->binding.binning_layout, wgpuBindGroupLayoutRelease);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_get_stats(struct onedraw* r, od_stats* stats)
{
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(stats);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_set_clear_color(struct onedraw* r, draw_color srgb_color)
{
    float r8 = (float)(srgb_color & 0xFF) / 255.f;
    float g8 = (float)((srgb_color >> 8) & 0xFF) / 255.f;
    float b8 = (float)((srgb_color >> 16) & 0xFF) / 255.f;
    float a8 = (float)((srgb_color >> 24) & 0xFF) / 255.f;

    if (r->rasterizer.srgb_backbuffer)
    {
        r->rasterizer.clear_color.x = srgb_to_linear(r8);
        r->rasterizer.clear_color.y = srgb_to_linear(g8);
        r->rasterizer.clear_color.z = srgb_to_linear(b8);
        r->rasterizer.clear_color.w = a8;
    }
    else
    {
        r->rasterizer.clear_color.x = r8;
        r->rasterizer.clear_color.y = g8;
        r->rasterizer.clear_color.z = b8;
        r->rasterizer.clear_color.w = a8;
    }
}