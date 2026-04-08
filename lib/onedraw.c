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
#include <stdlib.h>
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
#define BUFFER_FRAME_COUNT (3)
#define TILE_SIZE (16)
#define MAX_NODES_COUNT (1U<<22)
#define MAX_COMMANDS (1U<<16)
#define MAX_DRAWDATA (MAX_COMMANDS*4)
#define MAX_CLIPS (256)
#define OPTION_SRGB_BACKBUFFER (1U<<0)
#define OPTION_DEBUG_BINNING (1U<<1)

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
#define WGPU_STRING_VIEW(s) (WGPUStringView){.data = (s), .length = sizeof(s) - 1}
#define ARRAY_SIZE(x) (sizeof(x) / sizeof(*(x)))
#define DB_PUSH(db, TYPE, VALUE)                                      \
    do {                                                              \
        assert((db)->element_size == sizeof(TYPE));                   \
        assert((db)->num_elements < (db)->num_elements_max);           \
                                                                      \
        TYPE* _ptr = (TYPE*)((db)->cpu_buffer);                        \
        _ptr[(db)->num_elements++] = (VALUE);                          \
    } while (0)

#define LAST_CLIP_INDEX (uint8_t)(r->commands.clipshapes.num_elements-1)

static inline uint32_t min_u32(uint32_t a, uint32_t b) {return a < b ? a : b;}
static inline uint32_t max_u32(uint32_t a, uint32_t b) {return a > b ? a : b;}
static inline float min_f32(float a, float b) {return a < b ? a : b;}
static inline float max_f32(float a, float b) {return a > b ? a : b;}

#define OD_MIN(a, b) _Generic((a), \
    uint32_t: min_u32,          \
    float:    min_f32           \
)(a, b)

#define OD_MAX(a, b) _Generic((a), \
    uint32_t: max_u32,          \
    float:    max_f32           \
)(a, b)

// ---------------------------------------------------------------------------------------------------------------------------
// Private structures
// ---------------------------------------------------------------------------------------------------------------------------

typedef struct {float x, y, z, w;} vec4;
typedef uint32_t quantized_aabb;
typedef enum sdf_operator {op_additive, op_subtractive} sdf_operator;

typedef struct dynamic_buffer
{
    WGPUBuffer buffers[BUFFER_FRAME_COUNT];
    size_t element_size;
    size_t num_elements;
    size_t num_elements_max;
    void* cpu_buffer;
} dynamic_buffer;

struct onedraw
{
    WGPUDevice device;
    WGPUQueue queue;
    WGPUCommandBuffer command_buffer;

    struct
    {
        dynamic_buffer draw_args;
        dynamic_buffer list;
        dynamic_buffer colors;
        dynamic_buffer aabb;
        dynamic_buffer float_data;
        dynamic_buffer clipshapes;
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
        WGPUBuffer heads;
        WGPUComputePipeline binning_pso;
        WGPUComputePipeline write_indirect_buffer_pso;
        WGPUBuffer counters;
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
        vec4 clear_color;
        uint32_t width;
        uint32_t height;
        float aa_width;
        sdf_operator group_op;
        float outline_width;
        bool srgb_rendertarget;
        bool clear_rendertarget;
    } rasterizer;

    struct 
    {
        WGPUTexture texture;
        WGPUTextureView view;
        WGPUSampler sampler;
        uint32_t num_slices;
        uint32_t width;
        uint32_t height;
    } atlas;

    // font
    struct
    {
        WGPUTexture texture;
        WGPUTextureView view;
        WGPUSampler sampler;
        WGPUBuffer glyphs;
        od_font desc;
    } font;

    // stats
    struct
    {
        uint32_t peak_num_draw_cmd;
        uint32_t num_draw_data;
        uint32_t frame_index;
    } stats;

    struct
    {
        WGPUBindGroup rasterizer_bindgroup;
        WGPUBindGroup binning_bindgroup;
        WGPUBindGroup frame_bindgroup[BUFFER_FRAME_COUNT];
        WGPUBindGroupLayout rasterizer_layout;
        WGPUBindGroupLayout binning_layout;
        WGPUBindGroupLayout frame_layout;
    } binding;

    od_mem_interface mem_interface;
    void (*custom_log)(const char* string);
    char string_buffer[STRING_BUFFER_SIZE];
};

typedef struct vec2 {float x, y;} vec2;
typedef struct aabb {vec2 min, max;} aabb;
typedef struct quadratic_bezier {vec2 c0, c1, c2;} quadratic_bezier;
typedef struct cubic_bezier {vec2 c0, c1, c2, c3;} cubic_bezier;

// ---------------------------------------------------------------------------------------------------------------------------
// Warning : the gpu structures must be in sync with the one in common.wgsl
typedef struct gpu_draw_args
{
    vec4 clear_color;
    vec2 screen_div;
    uint32_t num_commands;
    uint32_t num_tile_width;
    uint32_t num_tile_height;
    uint32_t max_nodes;
    float aa_width;
    uint32_t options;
} gpu_draw_args;

typedef struct gpu_draw_command 
{
    uint32_t data_index;
    uint32_t flags; // extra (8) | clip_index (8) | fillmode (8) | type (8)
} gpu_draw_command;

typedef struct gpu_char
{
    vec2 uv_topleft;
    vec2 uv_bottomright;
    float width;
    float height;
} gpu_char;

typedef struct gpu_indirect_params
{
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
} gpu_indirect_params;

enum primitive_fillmode
{
    fill_solid = 0,
    fill_outline = 1,
    fill_hollow = 2,
    fill_gradient = 3
};

enum command_type
{
    primitive_char = 0,
    primitive_aabox = 1,
    primitive_oriented_box = 2,
    primitive_disc = 3,
    primitive_triangle = 4,
    primitive_ellipse = 5,
    primitive_pie = 6,
    primitive_arc = 7,
    primitive_blurred_box = 8,
    primitive_quad = 9,
    primitive_oriented_quad = 10,
    
    begin_group = 32,
    end_group = 33
};

// ---------------------------------------------------------------------------------------------------------------------------
// private functions
// ---------------------------------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------------------------------
static inline void* malloc_wrapper(size_t size, void* user) {(void)user; return malloc(size);}
static inline void* realloc_wrapper(void* old_ptr, size_t old_size, size_t new_size, void* user){(void)user;(void)old_size;return realloc(old_ptr, new_size);}
static inline void free_wrapper(void* ptr, void* user) {(void)user; free(ptr);}

//-----------------------------------------------------------------------------------------------------------------------------
static inline od_mem_interface default_allocator(void) 
{
    return (od_mem_interface) 
    {
        .malloc_fn  = malloc_wrapper,
        .realloc_fn = realloc_wrapper,
        .free_fn    = free_wrapper,
        .user       = NULL
    };
}

static inline float float_max(float a, float b) {return (a>b) ? a : b;}
static inline float float_clamp(float f, float a, float b) {if (f<a) return a; if (f>b) return b; return f;}
static inline void float_swap(float *a, float *b) {float temp = *a; *a = *b; *b = temp;} 
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
static inline gpu_draw_command gpu_draw_command_make(size_t data_index, uint8_t extra, uint8_t clip_index,
                                                     enum primitive_fillmode fillmode, enum command_type type)
{
    gpu_draw_command cmd;
    cmd.data_index = (uint32_t)data_index;
    cmd.flags = (uint32_t)extra | ((uint32_t)clip_index << 8) | ((uint32_t)fillmode << 16) | ((uint32_t)type << 24);
    return cmd;
}

//----------------------------------------------------------------------------------------------------------------------------
static inline quantized_aabb quantized_aabb_make(float min_x, float min_y, float max_x, float max_y)
{
    min_x = OD_MAX(min_x, 0.0f);
    min_y = OD_MAX(min_y, 0.0f);
    max_x = OD_MAX(max_x, 0.0f);
    max_y = OD_MAX(max_y, 0.0f);

    uint32_t qmin_x = OD_MIN((uint32_t)min_x / TILE_SIZE, (uint32_t)UINT8_MAX);
    uint32_t qmin_y = OD_MIN((uint32_t)min_y / TILE_SIZE, (uint32_t)UINT8_MAX);
    uint32_t qmax_x = OD_MIN((uint32_t)max_x / TILE_SIZE, (uint32_t)UINT8_MAX);
    uint32_t qmax_y = OD_MIN((uint32_t)max_y / TILE_SIZE, (uint32_t)UINT8_MAX);

    return  (qmin_x) | (qmin_y << 8) | (qmax_x << 16) | (qmax_y << 24);
}

//----------------------------------------------------------------------------------------------------------------------------
static inline void merge_quantized_aabb(quantized_aabb* merge, const quantized_aabb* other)
{
    if (merge != NULL)
    {
        uint32_t m = *merge;
        uint32_t o = *other;

        uint32_t min_x = OD_MIN(m & 0xFFu, o & 0xFFu);
        uint32_t min_y = OD_MIN((m >> 8) & 0xFFu, (o >> 8) & 0xFFu);
        uint32_t max_x = OD_MAX((m >> 16) & 0xFFu, (o >> 16) & 0xFFu);
        uint32_t max_y = OD_MAX((m >> 24) & 0xFFu, (o >> 24) & 0xFFu);

        *merge = (quantized_aabb)(min_x | (min_y << 8) | (max_x << 16) | (max_y << 24));
    }
}

//----------------------------------------------------------------------------------------------------------------------------
static inline quantized_aabb invalid_quantized_aabb() {return (quantized_aabb) (0x0000ffff);}

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
void dynamic_buffer_init(struct onedraw* r, dynamic_buffer* b, size_t element_size, size_t num_elements_max)
{
    b->element_size = element_size;
    b->num_elements = 0;
    b->num_elements_max = num_elements_max;

    size_t alloc_size = element_size * num_elements_max;
    for(uint32_t i=0; i<BUFFER_FRAME_COUNT; ++i)
    {
        b->buffers[i] = wgpuDeviceCreateBuffer(r->device, &(WGPUBufferDescriptor)
        {
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size = alloc_size
        });
        assert_msg(b->buffers[i] != NULL, "failed to create storage buffer");
    }

    b->cpu_buffer = r->mem_interface.malloc_fn(alloc_size, r->mem_interface.user);
}

//-----------------------------------------------------------------------------------------------------------------------------
void dynamic_buffer_upload(WGPUQueue queue, dynamic_buffer* b, uint32_t index)
{
    assert(index < BUFFER_FRAME_COUNT);
    if (b->num_elements > 0)
        wgpuQueueWriteBuffer(queue, b->buffers[index], 0, b->cpu_buffer, b->num_elements * b->element_size);
}

//-----------------------------------------------------------------------------------------------------------------------------
void dynamic_buffer_terminate(struct onedraw* r, dynamic_buffer* b)
{
    for(uint32_t i=0; i<BUFFER_FRAME_COUNT; ++i)
        SAFE_RELEASE(b->buffers[i], wgpuBufferRelease);
    
    r->mem_interface.free_fn(b->cpu_buffer, r->mem_interface.user);
}

//-----------------------------------------------------------------------------------------------------------------------------
void build_layout(struct onedraw* r)
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
        {   // glyphs
            .binding = 2,
            .visibility = WGPUShaderStage_Fragment,
            .buffer =
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        {   // heads
            .binding = 3,
            .visibility = WGPUShaderStage_Fragment,
            .buffer =
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        {   // atlas
            .binding = 4,
            .visibility = WGPUShaderStage_Fragment,
            .texture =
            {
                .sampleType = WGPUTextureSampleType_Float,
                .viewDimension = WGPUTextureViewDimension_2DArray,
                .multisampled = false
            }
        },
        {   // atlas sampler
            .binding = 5,
            .visibility = WGPUShaderStage_Fragment,
            .sampler = {.type = WGPUSamplerBindingType_Filtering}
        },
        { // font
            .binding = 6,
            .visibility = WGPUShaderStage_Fragment,
            .texture = 
            {
                .sampleType = WGPUTextureSampleType_Float,
                .viewDimension = WGPUTextureViewDimension_2D,
                .multisampled = false,
            }
        },
        {   // font sampler
            .binding = 7,
            .visibility = WGPUShaderStage_Fragment,
            .sampler = {.type = WGPUSamplerBindingType_Filtering}
        }
    };

    r->binding.rasterizer_layout = wgpuDeviceCreateBindGroupLayout(r->device, &(WGPUBindGroupLayoutDescriptor)
    {
        .label = WGPU_STRING_VIEW("rasterizer_layout"),
        .entries = rasterizer_layout_entries,
        .entryCount = ARRAY_SIZE(rasterizer_layout_entries)
    });

    WGPUBindGroupLayoutEntry frame_layout_entries[] =
    {
        { // g_draw_args
            .binding = 0,
            .visibility = WGPUShaderStage_Compute | WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
            .buffer = 
            {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
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
        .label = WGPU_STRING_VIEW("frame_layout"),
        .entries = frame_layout_entries,
        .entryCount = ARRAY_SIZE(frame_layout_entries)
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
        {   // heads
            .binding = 3,
            .visibility = WGPUShaderStage_Compute,
            .buffer =
            {
                .type = WGPUBufferBindingType_Storage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        },
        {   // indirect_draw_params
            .binding = 4,
            .visibility = WGPUShaderStage_Compute,
            .buffer =
            {
                .type = WGPUBufferBindingType_Storage,
                .hasDynamicOffset = false,
                .minBindingSize = 0,
            }
        }
    };

    r->binding.binning_layout = wgpuDeviceCreateBindGroupLayout(r->device, &(WGPUBindGroupLayoutDescriptor)
    {
        .label = WGPU_STRING_VIEW("binning_layout"),
        .entries = binning_layout_entries,
        .entryCount = ARRAY_SIZE(binning_layout_entries)
    });
}

//-----------------------------------------------------------------------------------------------------------------------------
void build_bind_groups(struct onedraw* r)
{
    WGPUBindGroupEntry rasterizer_entries[] = 
    {
        {.binding = 0, .buffer = r->tiles.nodes, .size = WGPU_WHOLE_SIZE},
        {.binding = 1, .buffer = r->tiles.indices, .size = WGPU_WHOLE_SIZE},
        {.binding = 2, .buffer = r->font.glyphs, .size = WGPU_WHOLE_SIZE},
        {.binding = 3, .buffer = r->tiles.heads, .size = WGPU_WHOLE_SIZE},
        {.binding = 4, .textureView = r->atlas.view},
        {.binding = 5, .sampler = r->atlas.sampler},
        {.binding = 6, .textureView = r->font.view},
        {.binding = 7, .sampler = r->font.sampler},
    };

    assert(r->tiles.nodes != NULL);
    assert(r->tiles.indices != NULL);
    assert(r->font.glyphs != NULL);
    assert(r->tiles.heads != NULL);
    assert(r->atlas.view != NULL);
    assert(r->atlas.sampler != NULL);
    assert(r->font.view != NULL);
    assert(r->font.sampler != NULL);

    r->binding.rasterizer_bindgroup =  wgpuDeviceCreateBindGroup(r->device, &(WGPUBindGroupDescriptor)
    {
        .label = WGPU_STRING_VIEW("rasterizer bindgroup"),
        .layout = r->binding.rasterizer_layout,
        .entryCount = ARRAY_SIZE(rasterizer_entries),
        .entries = rasterizer_entries
    });
    assert_msg(r->binding.rasterizer_bindgroup != NULL, "cannot create rasterizer binding group");

    WGPUBindGroupEntry binning_entries[] = 
    {
        {.binding = 0, .buffer = r->tiles.nodes, .size = WGPU_WHOLE_SIZE},
        {.binding = 1, .buffer = r->tiles.indices, .size = WGPU_WHOLE_SIZE},
        {.binding = 2, .buffer = r->tiles.counters, .size = WGPU_WHOLE_SIZE},
        {.binding = 3, .buffer = r->tiles.heads, .size = WGPU_WHOLE_SIZE},
        {.binding = 4, .buffer = r->tiles.indirect_draw_params, .size = WGPU_WHOLE_SIZE}
    };

    assert(r->tiles.counters != NULL);
    assert(r->tiles.heads != NULL);
    assert(r->tiles.indirect_draw_params != NULL);

    r->binding.binning_bindgroup =  wgpuDeviceCreateBindGroup(r->device, &(WGPUBindGroupDescriptor)
    {
        .label = WGPU_STRING_VIEW("binning bindgroup"),
        .layout = r->binding.binning_layout,
        .entryCount = ARRAY_SIZE(binning_entries),
        .entries = binning_entries
    });
    assert_msg(r->binding.binning_bindgroup != NULL, "cannot create rasterizer binding group");

    for(uint32_t i=0; i<BUFFER_FRAME_COUNT; ++i)
    {
        WGPUBindGroupEntry frame_entries[] = 
        {
            {.binding = 0, .buffer = r->commands.draw_args.buffers[i], .size = WGPU_WHOLE_SIZE},
            {.binding = 1, .buffer = r->commands.list.buffers[i], .size = WGPU_WHOLE_SIZE},
            {.binding = 2, .buffer = r->commands.aabb.buffers[i], .size = WGPU_WHOLE_SIZE},
            {.binding = 3, .buffer = r->commands.float_data.buffers[i], .size = WGPU_WHOLE_SIZE},
            {.binding = 4, .buffer = r->commands.clipshapes.buffers[i], .size = WGPU_WHOLE_SIZE},
            {.binding = 5, .buffer = r->commands.colors.buffers[i], .size = WGPU_WHOLE_SIZE}
        };

        r->binding.frame_bindgroup[i] =  wgpuDeviceCreateBindGroup(r->device, &(WGPUBindGroupDescriptor)
        {
            .label = WGPU_STRING_VIEW("frame bindgroup"),
            .layout = r->binding.frame_layout,
            .entryCount = ARRAY_SIZE(frame_entries),
            .entries = frame_entries
        });
        assert_msg(r->binding.frame_bindgroup[i] != NULL, "cannot create frame binding groups");
    }
}

//-----------------------------------------------------------------------------------------------------------------------------
void build_pso(struct onedraw* r, WGPUTextureFormat surface_format)
{
    // rasterizer pso
    WGPUShaderSourceWGSL rasterizer_wgsl = 
    {
        .chain = {.next = NULL, .sType = WGPUSType_ShaderSourceWGSL},
        .code = {.data = rasterizer_shader, .length = rasterizer_shader_size}
    };


    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(r->device, &(WGPUShaderModuleDescriptor)
    {
        .nextInChain = &rasterizer_wgsl.chain,
        .label = WGPU_STRING_VIEW("rasterizer")
    });

    WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(r->device, &(WGPUPipelineLayoutDescriptor)
    {
        .bindGroupLayoutCount = 2,
        .bindGroupLayouts = (WGPUBindGroupLayout[]) {r->binding.rasterizer_layout, r->binding.frame_layout}
    });

    WGPUBlendState alpha_blend =
    {
        .color = (WGPUBlendComponent)
        {
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_SrcAlpha,
            .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
        },
        .alpha = (WGPUBlendComponent)
        {
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_One,
            .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
        },
    };

    WGPUColorTargetState color_target = 
    {
        .format = surface_format,
        .blend = &alpha_blend,
        .writeMask = WGPUColorWriteMask_All
    };

    WGPUFragmentState fragment = 
    {
        .module = shader,
        .entryPoint = WGPU_STRING_VIEW("tile_fs"),
        .targetCount = 1,
        .targets = &color_target
    };

    WGPURenderPipelineDescriptor render_pipeline_desc = 
    {
        .layout = layout,
        .vertex = 
        {
            .module = shader,
            .entryPoint = WGPU_STRING_VIEW("tile_vs"),
            .bufferCount = 0,
            .buffers = NULL
        },

        .primitive = 
        {
            .topology = WGPUPrimitiveTopology_TriangleStrip,
            .frontFace = WGPUFrontFace_CCW,
            .cullMode = WGPUCullMode_None
        },

        .fragment = &fragment,
        .multisample = {.count = 1, .mask = ~0u},
    };

    r->rasterizer.pso = wgpuDeviceCreateRenderPipeline(r->device, &render_pipeline_desc);
    assert_msg(r->rasterizer.pso != NULL, "can't create rasterizer pso");

    wgpuShaderModuleRelease(shader);
    wgpuPipelineLayoutRelease(layout);

    // binning pso
    WGPUShaderSourceWGSL binning_wgsl = 
    {
        .chain = {.next = NULL, .sType = WGPUSType_ShaderSourceWGSL},
        .code = {.data = binning_shader, .length = binning_shader_size}
    };

    layout = wgpuDeviceCreatePipelineLayout(r->device, &(WGPUPipelineLayoutDescriptor)
    {
        .bindGroupLayoutCount = 2,
        .bindGroupLayouts = (WGPUBindGroupLayout[]) {r->binding.binning_layout, r->binding.frame_layout}
    });

    shader = wgpuDeviceCreateShaderModule(r->device, &(WGPUShaderModuleDescriptor)
    {
        .nextInChain = &binning_wgsl.chain,
        .label = WGPU_STRING_VIEW("binning")
    });

    WGPUComputePipelineDescriptor compute_pipeline_desc = (WGPUComputePipelineDescriptor)
    {
        .layout = layout,
        .compute = 
        {
            .module = shader,
            .entryPoint = WGPU_STRING_VIEW("tile_bin"),
            .constants = NULL
        }
    };

    r->tiles.binning_pso = wgpuDeviceCreateComputePipeline(r->device, &compute_pipeline_desc);
    assert_msg(r->tiles.binning_pso != NULL, "can't create binning pso");

    compute_pipeline_desc = (WGPUComputePipelineDescriptor)
    {
        .layout = layout,
        .compute = 
        {
            .module = shader,
            .entryPoint = WGPU_STRING_VIEW("write_indirect_args"),
            .constants = NULL
        }
    };

    r->tiles.write_indirect_buffer_pso = wgpuDeviceCreateComputePipeline(r->device, &compute_pipeline_desc);
    assert_msg(r->tiles.binning_pso != NULL, "can't create write indirect buffer pso");

    wgpuShaderModuleRelease(shader);
    wgpuPipelineLayoutRelease(layout);
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

    WGPUSamplerDescriptor sampler_desc = 
    {
        .label = WGPU_STRING_VIEW("font sampler"), 
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge,
        .addressModeW = WGPUAddressMode_ClampToEdge,
        .magFilter = WGPUFilterMode_Linear,
        .minFilter = WGPUFilterMode_Linear,
        .mipmapFilter = WGPUMipmapFilterMode_Nearest, // no mipmap
        .lodMinClamp = 0.0f,
        .lodMaxClamp = 1.0f,
        .maxAnisotropy = 1
    };

    r->font.sampler = wgpuDeviceCreateSampler(r->device, &sampler_desc);
    assert_msg(r->font.sampler != NULL, "cannot create font sampler");

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

    // fill the glyph description to be upload on the gpu
    gpu_char cpu_buffer[MAX_GLYPHS];
    float oo_texture_width = 1.f / (float)r->font.desc.texture_width;
    float oo_texture_height = 1.f / (float)r->font.desc.texture_height;
    for(uint32_t i=0; i<r->font.desc.num_glyphs; i++)
    {
        const od_glyph* glyph = &r->font.desc.glyphs[i];
        cpu_buffer[i] = (gpu_char)
        {
            .width = (float)glyph->x1 - (float)glyph->x0,
            .height = (float)glyph->y1 - (float)glyph->y0,
            .uv_topleft = 
            {
                .x = ((float)glyph->x0) * oo_texture_width, 
                .y = ((float)glyph->y0) * oo_texture_height
            },
            .uv_bottomright = 
            {
                .x = ((float)glyph->x1) * oo_texture_width,
                .y = ((float)glyph->y1) * oo_texture_height
            }
        };
    }

    size_t gpu_font_size = sizeof(cpu_buffer);
    r->font.glyphs = wgpuDeviceCreateBuffer(r->device, &(WGPUBufferDescriptor)
    {
        .size = gpu_font_size,
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .label = WGPU_STRING_VIEW("gpu_font")
    });

    assert_msg(r->font.glyphs != NULL, "can create font description gpu buffer");
    wgpuQueueWriteBuffer(r->queue, r->font.glyphs, 0, cpu_buffer, gpu_font_size);
}

// ---------------------------------------------------------------------------------------------------------------------------
void create_atlas(struct onedraw* r, const onedraw_def* def)
{
    // create a dummy if not needed to please the bind group
    if (def->atlas.height == 0 || def->atlas.width == 0 || def->atlas.num_slices == 0)
    {
        r->atlas.num_slices = 1;
        r->atlas.width = 1;
        r->atlas.height = 1;
    }
    else
    {
        assert_msg(false, "to be implemented");
    }

    WGPUTextureDescriptor atlas_desc = 
    {
        .size = {r->atlas.width, r->atlas.height, r->atlas.num_slices},
        .mipLevelCount = 1,
        .sampleCount = 1,
        .dimension = WGPUTextureDimension_2D,
        .format = WGPUTextureFormat_RGBA8Unorm,
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    };

    r->atlas.texture = wgpuDeviceCreateTexture(r->device, &atlas_desc);
    assert_msg(r->atlas.texture != NULL, "failed to create atlas texture");

    r->atlas.view = wgpuTextureCreateView(r->atlas.texture, &(WGPUTextureViewDescriptor)
    {
        .label = WGPU_STRING_VIEW("atlas texture view"),
        .format = WGPUTextureFormat_RGBA8Unorm,
        .dimension = WGPUTextureViewDimension_2DArray,
        .mipLevelCount = 1,
        .arrayLayerCount = r->atlas.num_slices,
        .aspect = WGPUTextureAspect_All,
    });
    assert_msg(r->atlas.view != NULL, "default atlas texture view creation failed");

    WGPUSamplerDescriptor sampler_desc = 
    {
        .label = WGPU_STRING_VIEW("atlas sampler"), 
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge,
        .addressModeW = WGPUAddressMode_ClampToEdge,
        .magFilter = WGPUFilterMode_Linear,
        .minFilter = WGPUFilterMode_Linear,
        .mipmapFilter = WGPUMipmapFilterMode_Nearest, // no mipmap
        .lodMinClamp = 0.0f,
        .lodMaxClamp = 1.0f,
        .maxAnisotropy = 1
    };

    r->atlas.sampler = wgpuDeviceCreateSampler(r->device, &sampler_desc);
    assert_msg(r->atlas.sampler != NULL, "cannot create atlas sampler");
}

//-----------------------------------------------------------------------------------------------------------------------------
void allocate_dynamic_buffers(struct onedraw* r)
{
    dynamic_buffer_init(r, &r->commands.draw_args, sizeof(gpu_draw_args), 1);
    dynamic_buffer_init(r, &r->commands.aabb, sizeof(uint32_t), MAX_COMMANDS);
    dynamic_buffer_init(r, &r->commands.list, sizeof(gpu_draw_command), MAX_COMMANDS);
    dynamic_buffer_init(r, &r->commands.colors, sizeof(uint32_t), MAX_COMMANDS);
    dynamic_buffer_init(r, &r->commands.clipshapes, sizeof(vec4), MAX_COMMANDS);
    dynamic_buffer_init(r, &r->commands.float_data, sizeof(float), MAX_DRAWDATA);
}

//-----------------------------------------------------------------------------------------------------------------------------
float draw_cmd_aabb_bump(struct onedraw* r)
{
    return r->rasterizer.aa_width + r->rasterizer.outline_width;
}

//-----------------------------------------------------------------------------------------------------------------------------
bool buffers_are_full(struct onedraw* r, size_t num_floats)
{
    return ((r->commands.float_data.num_elements + num_floats >= r->commands.float_data.num_elements_max) ||
            (r->commands.colors.num_elements >= r->commands.colors.num_elements_max) ||
            (r->commands.list.num_elements >= r->commands.list.num_elements_max) ||
            (r->commands.aabb.num_elements >= r->commands.aabb.num_elements_max));
}

// ---------------------------------------------------------------------------------------------------------------------------
// public functions
// ---------------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------------
struct onedraw* od_init(const onedraw_def* def)
{
    od_mem_interface mem_interface = (def->mem_interface != NULL) ? (*def->mem_interface) : default_allocator();
    
    struct onedraw* r = mem_interface.malloc_fn(sizeof(struct onedraw), mem_interface.user);
    assert(def->device != NULL);

    // clear everything to zero
    *r = (struct onedraw) {0};

    r->mem_interface = mem_interface;
    r->custom_log = def->log_func;
    r->device = def->device;
    r->rasterizer.srgb_rendertarget = def->srgb_rendertarget;
    r->rasterizer.aa_width = VEC2_SQR2;
    r->rasterizer.clear_rendertarget = def->clear_rendertarget;
    r->rasterizer.clear_color = (vec4) {.x = 0.f, .y = 0.f, .z = 0.f, .w = 1.f};
    r->queue = wgpuDeviceGetQueue(r->device);
    assert_msg(r->queue != NULL, "cannot create queue from device");

    // resource creation
    r->tiles.counters = wgpuDeviceCreateBuffer(r->device, &(WGPUBufferDescriptor)
    {
        .label = WGPU_STRING_VIEW("tile_counters"),
        .mappedAtCreation = false,
        .size = sizeof(uint32_t) * 2,
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst
    });
    assert_msg(r->tiles.counters != NULL, "failed to create tile counters buffer");
    
    r->tiles.nodes = wgpuDeviceCreateBuffer(r->device, &(WGPUBufferDescriptor)
    {
        .label = WGPU_STRING_VIEW("tile_nodes"),
        .mappedAtCreation = false,
        .size = sizeof(uint64_t) * MAX_NODES_COUNT,
        .usage = WGPUBufferUsage_Storage
    });
    assert_msg(r->tiles.nodes != NULL, "failed to create tile nodes buffer");

    r->tiles.indirect_draw_params = wgpuDeviceCreateBuffer(r->device, &(WGPUBufferDescriptor)
    {
        .label = WGPU_STRING_VIEW("indirect_draw_params"),
        .mappedAtCreation = false,
        .size = sizeof(gpu_indirect_params),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Indirect
    });
    assert_msg(r->tiles.indirect_draw_params != NULL, "failed to indirect draw params buffer");

    create_atlas(r, def);
    od_resize(r, def->viewport_width, def->viewport_height);
    allocate_dynamic_buffers(r);
    build_font(r);
    build_layout(r);
    build_bind_groups(r);
    build_pso(r, def->surface_format);
    // od_build_depthstencil_state(r);

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
void od_resize(struct onedraw* r, uint32_t width, uint32_t height)
{
    assert(width > 16 && height > 16);
    if (width != r->rasterizer.width || height != r->rasterizer.height)
    {
        od_log(r, "resizing the framebuffer to %dx%d", width, height);

        r->rasterizer.width = width;
        r->rasterizer.height = height;
        r->tiles.num_width = (width + TILE_SIZE - 1) / TILE_SIZE;
        r->tiles.num_height = (height + TILE_SIZE - 1) / TILE_SIZE;
        r->tiles.count = r->tiles.num_width * r->tiles.num_height;

        // TODO : add region support

        SAFE_RELEASE(r->tiles.heads, wgpuBufferRelease);
        r->tiles.heads = wgpuDeviceCreateBuffer(r->device, &(WGPUBufferDescriptor)
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
    }
    
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_begin_frame(struct onedraw* r)
{
    r->stats.frame_index++;
    r->commands.aabb.num_elements = 0;
    r->commands.clipshapes.num_elements = 0;
    r->commands.colors.num_elements = 0;
    r->commands.draw_args.num_elements = 0;
    r->commands.float_data.num_elements = 0;
    r->commands.list.num_elements = 0;

    // push default clip rect
    vec4 no_clip = (vec4) {.x = 0.f, .y = 0.f, .z = (float)r->rasterizer.width, .w = (float)r->rasterizer.height};
    DB_PUSH(&r->commands.clipshapes, vec4, no_clip);
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_end_frame(struct onedraw* r, WGPUTextureView target_view)
{
    assert_msg(r->commands.group_aabb == NULL, "begin/end group pair mismatch");

    // global gpu args
    gpu_draw_args args =
    {
        .aa_width = VEC2_SQR2,
        .clear_color = r->rasterizer.clear_color,
        .screen_div = {1.f / (float)r->rasterizer.width, 1.f / (float)r->rasterizer.height},
        .num_commands = (uint32_t)r->commands.list.num_elements,
        .max_nodes = MAX_NODES_COUNT,
        .num_tile_width = r->tiles.num_width,
        .num_tile_height = r->tiles.num_height,
        .options = r->rasterizer.srgb_rendertarget ? OPTION_SRGB_BACKBUFFER : 0U
    };
    DB_PUSH(&r->commands.draw_args, gpu_draw_args, args);

    // upload storage buffers to gpu
    uint32_t buffer_index = r->stats.frame_index % BUFFER_FRAME_COUNT;
    dynamic_buffer_upload(r->queue, &r->commands.aabb, buffer_index);
    dynamic_buffer_upload(r->queue, &r->commands.clipshapes, buffer_index);
    dynamic_buffer_upload(r->queue, &r->commands.colors, buffer_index);
    dynamic_buffer_upload(r->queue, &r->commands.float_data, buffer_index);
    dynamic_buffer_upload(r->queue, &r->commands.draw_args, buffer_index);
    dynamic_buffer_upload(r->queue, &r->commands.list, buffer_index);

    WGPUCommandEncoderDescriptor encoder_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(r->device, &encoder_desc);

    // clear counter
    wgpuCommandEncoderClearBuffer(encoder, r->tiles.counters, 0, sizeof(uint32_t) * 2);

    // binning pass
    {
        WGPUComputePassDescriptor desc = {.label = WGPU_STRING_VIEW("binning pass")};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &desc);
        wgpuComputePassEncoderSetPipeline(pass, r->tiles.binning_pso);
        wgpuComputePassEncoderSetBindGroup(pass, 0, r->binding.binning_bindgroup, 0, NULL);
        wgpuComputePassEncoderSetBindGroup(pass, 1, r->binding.frame_bindgroup[buffer_index], 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(pass, r->tiles.num_width, r->tiles.num_height, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // write indirect params pass
    {
        WGPUComputePassDescriptor desc = {.label = WGPU_STRING_VIEW("write_indirect_params pass")};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &desc);
        wgpuComputePassEncoderSetPipeline(pass, r->tiles.write_indirect_buffer_pso);
        wgpuComputePassEncoderSetBindGroup(pass, 0, r->binding.binning_bindgroup, 0, NULL);
        wgpuComputePassEncoderSetBindGroup(pass, 1, r->binding.frame_bindgroup[buffer_index], 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // rasterization pass
    WGPURenderPassColorAttachment color_attachment = 
    {
        .view = target_view,
        .resolveTarget = NULL,
        .loadOp = r->rasterizer.clear_rendertarget ? WGPULoadOp_Clear : WGPULoadOp_Load,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = (WGPUColor){r->rasterizer.clear_color.x, r->rasterizer.clear_color.y, r->rasterizer.clear_color.z, r->rasterizer.clear_color.w},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED
    };

    WGPURenderPassDescriptor render_pass_desc = 
    {
        .colorAttachmentCount = 1,
        .colorAttachments = &color_attachment,
        .depthStencilAttachment = NULL,
        .label = WGPU_STRING_VIEW("rasterization pass")
    };

    
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_desc);

    wgpuRenderPassEncoderSetViewport(pass, 0.0, 0.0, (float)r->rasterizer.width, (float)r->rasterizer.height, 0.0, 1.0);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, r->rasterizer.width, r->rasterizer.height);
    wgpuRenderPassEncoderSetPipeline(pass, r->rasterizer.pso);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, r->binding.rasterizer_bindgroup, 0, NULL);
    wgpuRenderPassEncoderSetBindGroup(pass, 1, r->binding.frame_bindgroup[buffer_index], 0, NULL);
    wgpuRenderPassEncoderDrawIndirect(pass, r->tiles.indirect_draw_params, 0);

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
    dynamic_buffer_terminate(r, &r->commands.aabb);
    dynamic_buffer_terminate(r, &r->commands.clipshapes);
    dynamic_buffer_terminate(r, &r->commands.colors);
    dynamic_buffer_terminate(r, &r->commands.draw_args);
    dynamic_buffer_terminate(r, &r->commands.float_data);
    SAFE_RELEASE(r->atlas.sampler, wgpuSamplerRelease);
    SAFE_RELEASE(r->atlas.view, wgpuTextureViewRelease);
    SAFE_RELEASE(r->atlas.texture, wgpuTextureRelease);
    SAFE_RELEASE(r->font.sampler, wgpuSamplerRelease);
    SAFE_RELEASE(r->font.view, wgpuTextureViewRelease);
    SAFE_RELEASE(r->font.texture, wgpuTextureRelease);
    SAFE_RELEASE(r->binding.rasterizer_layout, wgpuBindGroupLayoutRelease);
    SAFE_RELEASE(r->binding.binning_layout, wgpuBindGroupLayoutRelease);
    SAFE_RELEASE(r->tiles.counters, wgpuBufferRelease);
    SAFE_RELEASE(r->tiles.nodes, wgpuBufferRelease);
    SAFE_RELEASE(r->tiles.heads, wgpuBufferRelease);
    SAFE_RELEASE(r->rasterizer.pso, wgpuRenderPipelineRelease);
    SAFE_RELEASE(r->tiles.binning_pso, wgpuComputePipelineRelease);
    SAFE_RELEASE(r->tiles.write_indirect_buffer_pso, wgpuComputePipelineRelease);
    r->mem_interface.free_fn(r, r->mem_interface.user);
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

    if (r->rasterizer.srgb_rendertarget)
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

//----------------------------------------------------------------------------------------------------------------------------
void private_draw_disc(struct onedraw* r, vec2 center, float radius, float thickness, enum primitive_fillmode fillmode,
                       draw_color primary_color, draw_color secondary_color)
{
    if (buffers_are_full(r, 4))
    {
        od_log(r, "buffers for primitive are full, expect graphical artefacts");
        return;
    }

    thickness *= .5f;

    gpu_draw_command cmd = gpu_draw_command_make(r->commands.float_data.num_elements, 0, LAST_CLIP_INDEX, fillmode, primitive_disc);
    DB_PUSH(&r->commands.list, gpu_draw_command, cmd);
    DB_PUSH(&r->commands.colors, draw_color, primary_color);

    float max_radius = radius + draw_cmd_aabb_bump(r);

    DB_PUSH(&r->commands.float_data, float, center.x);
    DB_PUSH(&r->commands.float_data, float, center.y);
    DB_PUSH(&r->commands.float_data, float, radius);

    if (fillmode == fill_hollow)
    {
        max_radius += thickness;
        DB_PUSH(&r->commands.float_data, float, thickness);
    }
    else if (fillmode == fill_gradient)
    {
        DB_PUSH(&r->commands.float_data, float, bitcast_u32_to_float(secondary_color));
    }

    quantized_aabb aabb = quantized_aabb_make(center.x - max_radius, center.y - max_radius, center.x + max_radius, center.y + max_radius);
    merge_quantized_aabb(r->commands.group_aabb, &aabb);
    DB_PUSH(&r->commands.aabb, quantized_aabb, aabb);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_ring(struct onedraw* r, float cx, float cy, float radius, float thickness, draw_color color)
{
    private_draw_disc(r, vec2_set(cx, cy), radius, thickness, fill_hollow, color, 0);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_disc(struct onedraw* r, float cx, float cy, float radius, draw_color color)
{
    private_draw_disc(r, vec2_set(cx, cy), radius, 0.f, fill_solid, color, 0);
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_draw_disc_gradient(struct onedraw* r, float cx, float cy, float radius, draw_color outter_color, draw_color inner_color)
{
    private_draw_disc(r, vec2_set(cx, cy), radius, 0.f, fill_gradient, outter_color, inner_color);
}

//----------------------------------------------------------------------------------------------------------------------------
void private_draw_oriented_box(struct onedraw* r, vec2 p0, vec2 p1, float width, float roundness, float thickness,
                               enum primitive_fillmode fillmode, draw_color primary_color, draw_color secondary_color)
{
    if (vec2_similar(p0, p1, HALF_PIXEL))
        return;

    if (buffers_are_full(r, 7))
    {
        od_log(r, "buffers for primitive are full, expect graphical artefacts");
        return;
    }

    thickness *= .5f;
    float roundness_thickness = (fillmode == fill_hollow) ? thickness : roundness;
    aabb bb = aabb_from_rounded_obb(p0, p1, width, roundness_thickness + draw_cmd_aabb_bump(r));

    gpu_draw_command cmd = gpu_draw_command_make(r->commands.float_data.num_elements, 0, LAST_CLIP_INDEX, fillmode, primitive_oriented_box);
    DB_PUSH(&r->commands.list, gpu_draw_command, cmd);
    DB_PUSH(&r->commands.colors, draw_color, primary_color);
    DB_PUSH(&r->commands.float_data, float, p0.x);
    DB_PUSH(&r->commands.float_data, float, p0.y);
    DB_PUSH(&r->commands.float_data, float, p1.x);
    DB_PUSH(&r->commands.float_data, float, p1.y);
    DB_PUSH(&r->commands.float_data, float, width);
    DB_PUSH(&r->commands.float_data, float, roundness_thickness);

    if (fillmode == fill_gradient)
    {
        DB_PUSH(&r->commands.float_data, float,bitcast_u32_to_float(secondary_color));
    }

    quantized_aabb aabb = quantized_aabb_make(bb.min.x, bb.min.y, bb.max.x, bb.max.y);
    merge_quantized_aabb(r->commands.group_aabb, &aabb);
    DB_PUSH(&r->commands.aabb, quantized_aabb, aabb);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_oriented_box(struct onedraw* r, float ax, float ay, float bx, float by, float width, float roundness, draw_color color)
{
    private_draw_oriented_box(r, vec2_set(ax, ay), vec2_set(bx, by), width, roundness, 0.f, fill_solid, color, 0);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_oriented_rect(struct onedraw* r, float ax, float ay, float bx, float by, float width, float roundness, float thickness, draw_color color)
{
    private_draw_oriented_box(r, vec2_set(ax, ay), vec2_set(bx, by), width, roundness, thickness, fill_hollow, color, 0);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_line(struct onedraw* r, float ax, float ay, float bx, float by, float width, draw_color srgb_color)
{
    private_draw_oriented_box(r, vec2_set(ax, ay), vec2_set(bx, by), width, 0.f, 0.f, fill_solid, srgb_color, 0);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_capsule(struct onedraw* r, float ax, float ay, float bx, float by, float radius, draw_color srgb_color)
{
    // capsule uses a specific sdf (see rasterizer shader) more efficient that oriented box
    private_draw_oriented_box(r, vec2_set(ax, ay), vec2_set(bx, by), 0.f, radius, 0.f, fill_solid, srgb_color, 0);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_capsule_gradient(struct onedraw* r, float ax, float ay, float bx, float by, float radius, draw_color primary_color, draw_color secondary_color)
{
    private_draw_oriented_box(r, vec2_set(ax, ay), vec2_set(bx, by), 0.f, radius, 0.f, fill_gradient, primary_color, secondary_color);
}

//----------------------------------------------------------------------------------------------------------------------------
void private_draw_ellipse(struct onedraw* r, vec2 p0, vec2 p1, float width, float thickness, enum primitive_fillmode fillmode, draw_color srgb_color)
{
    if (vec2_similar(p0, p1, HALF_PIXEL))
        return;

    if (width <= HALF_PIXEL)
    {
        private_draw_oriented_box(r, p0, p1, 0.f, 0.f, 0.f, fill_solid, srgb_color, 0);
        return;
    }

    if (buffers_are_full(r, 7))
    {
        od_log(r, "buffers for primitive are full, expect graphical artefacts");
        return;
    }

    thickness = float_max(thickness * .5f, 0.f);
    aabb bb = aabb_from_rounded_obb(p0, p1, width, draw_cmd_aabb_bump(r) + thickness);

    gpu_draw_command cmd = gpu_draw_command_make(r->commands.float_data.num_elements, 0, LAST_CLIP_INDEX, fillmode, primitive_ellipse);
    DB_PUSH(&r->commands.list, gpu_draw_command, cmd);
    DB_PUSH(&r->commands.colors, draw_color, srgb_color);
    DB_PUSH(&r->commands.float_data, float, p0.x);
    DB_PUSH(&r->commands.float_data, float, p0.y);
    DB_PUSH(&r->commands.float_data, float, p1.x);
    DB_PUSH(&r->commands.float_data, float, p1.y);
    DB_PUSH(&r->commands.float_data, float, width);

    if (fillmode == fill_hollow)
    {
        DB_PUSH(&r->commands.float_data, float, thickness);
    }

    quantized_aabb aabb = quantized_aabb_make(bb.min.x, bb.min.y, bb.max.x, bb.max.y);
    merge_quantized_aabb(r->commands.group_aabb, &aabb);
    DB_PUSH(&r->commands.aabb, quantized_aabb, aabb);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_ellipse(struct onedraw* r, float ax, float ay, float bx, float by, float width, draw_color srgb_color)
{
    private_draw_ellipse(r, vec2_set(ax, ay), vec2_set(bx, by), width, 0.f, fill_solid, srgb_color);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_ellipse_ring(struct onedraw* r, float ax, float ay, float bx, float by, float width, float thickness, draw_color srgb_color)
{
    private_draw_ellipse(r, vec2_set(ax, ay), vec2_set(bx, by), width, thickness, fill_hollow, srgb_color);
}

//----------------------------------------------------------------------------------------------------------------------------
void private_draw_triangle(struct onedraw* r, const vec2* v, float roundness, float thickness, enum primitive_fillmode fillmode, draw_color srgb_color)
{
    // exclude invalid triangle
    if (vec2_similar(v[0], v[1], HALF_PIXEL) || vec2_similar(v[2], v[1], HALF_PIXEL) || 
        vec2_similar(v[0], v[2], HALF_PIXEL)) return;

    if (buffers_are_full(r, 7))
    {
        od_log(r, "buffers for primitive are full, expect graphical artefacts");
        return;
    }

    thickness *= .5f;
    float roundness_thickness = (fillmode != fill_hollow) ? roundness : thickness;

    aabb bb = aabb_from_triangle(v[0], v[1], v[2]);
    aabb_grow(&bb, vec2_splat(roundness_thickness + draw_cmd_aabb_bump(r)));

    gpu_draw_command cmd = gpu_draw_command_make(r->commands.float_data.num_elements, 0, LAST_CLIP_INDEX, fillmode, primitive_triangle);
    DB_PUSH(&r->commands.list, gpu_draw_command, cmd);
    DB_PUSH(&r->commands.colors, draw_color, srgb_color);
    for(uint32_t i=0; i<3; ++i)
    {
        DB_PUSH(&r->commands.float_data, float, v[i].x);
        DB_PUSH(&r->commands.float_data, float, v[i].y);
    }
    DB_PUSH(&r->commands.float_data, float, roundness_thickness);

    quantized_aabb aabb = quantized_aabb_make(bb.min.x, bb.min.y, bb.max.x, bb.max.y);
    merge_quantized_aabb(r->commands.group_aabb, &aabb);
    DB_PUSH(&r->commands.aabb, quantized_aabb, aabb);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_triangle(struct onedraw* r, const float* vertices, float roundness, draw_color srgb_color)
{
    private_draw_triangle(r, (const vec2*) vertices, roundness, 0.f, fill_solid, srgb_color);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_triangle_ring(struct onedraw* r, const float* vertices, float roundness, float thickness, draw_color srgb_color)
{
    private_draw_triangle(r, (const vec2*) vertices, roundness, thickness, fill_hollow, srgb_color);
}

//----------------------------------------------------------------------------------------------------------------------------
void private_draw_pie(struct onedraw* r, vec2 center, vec2 direction, float radius, float aperture, float thickness, enum primitive_fillmode fillmode, draw_color srgb_color)
{
    if (aperture <= FLT_EPSILON)
        return;

    if (buffers_are_full(r, 8))
    {
        od_log(r, "buffers for primitive are full, expect graphical artefacts");
        return;
    }
    
    aperture = float_clamp(aperture, 0.f, VEC2_PI);
    thickness = float_max(thickness * .5f, 0.f);
    aabb bb = aabb_from_circle(center, radius);
    aabb_grow(&bb, vec2_splat(thickness + draw_cmd_aabb_bump(r)));

    gpu_draw_command cmd = gpu_draw_command_make(r->commands.float_data.num_elements, 0, LAST_CLIP_INDEX, fillmode, primitive_pie);
    DB_PUSH(&r->commands.list, gpu_draw_command, cmd);
    DB_PUSH(&r->commands.colors, draw_color, srgb_color);
    DB_PUSH(&r->commands.float_data, float, center.x);
    DB_PUSH(&r->commands.float_data, float, center.y);
    DB_PUSH(&r->commands.float_data, float, radius);
    DB_PUSH(&r->commands.float_data, float, direction.x);
    DB_PUSH(&r->commands.float_data, float, direction.y);
    DB_PUSH(&r->commands.float_data, float, sinf(aperture));
    DB_PUSH(&r->commands.float_data, float, cosf(aperture));

    if (fillmode == fill_hollow)
    {
        DB_PUSH(&r->commands.float_data, float, thickness);
    }

    quantized_aabb quantized_bb = quantized_aabb_make(bb.min.x, bb.min.y, bb.max.x, bb.max.y);
    merge_quantized_aabb(r->commands.group_aabb, &quantized_bb);
    DB_PUSH(&r->commands.aabb, quantized_aabb, quantized_bb);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_sector(struct onedraw* r, float cx, float cy, float radius, float start_angle, float sweep_angle, draw_color srgb_color)
{
    vec2 center = {cx, cy};
    float aperture = sweep_angle * .5f;
    vec2 direction = vec2_direction(start_angle + aperture);
    private_draw_pie(r, center, direction, radius, fabs(aperture), 0.f, fill_solid, srgb_color);
}

//----------------------------------------------------------------------------------------------------------------------------
void od_draw_sector_ring(struct onedraw* r, float cx, float cy, float radius, float start_angle, float sweep_angle, float thickness, draw_color srgb_color)
{
    vec2 center = {cx, cy};
    float aperture = sweep_angle * .5f;
    vec2 direction = vec2_direction(start_angle + aperture);
    private_draw_pie(r, center, direction, radius, fabs(aperture), thickness, fill_hollow, srgb_color);
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_draw_arc(struct onedraw* r, float cx, float cy, float dx, float dy, float aperture, float radius, float thickness, draw_color srgb_color)
{
    if (buffers_are_full(r, 8))
    {
        od_log(r, "buffers for primitive are full, expect graphical artefacts");
        return;
    }

    vec2 center = {cx, cy};
    vec2 direction = {dx, dy};

    aperture = float_clamp(aperture, 0.f, VEC2_PI);
    thickness = float_max(thickness, 0.f);
    aabb bb = aabb_from_circle(center, radius);
    aabb_grow(&bb, vec2_splat(thickness + draw_cmd_aabb_bump(r)));

    gpu_draw_command cmd = gpu_draw_command_make(r->commands.float_data.num_elements, 0, LAST_CLIP_INDEX, fill_solid, primitive_arc);
    DB_PUSH(&r->commands.list, gpu_draw_command, cmd);
    DB_PUSH(&r->commands.colors, draw_color, srgb_color);
    DB_PUSH(&r->commands.float_data, float, center.x);
    DB_PUSH(&r->commands.float_data, float, center.y);
    DB_PUSH(&r->commands.float_data, float, radius);
    DB_PUSH(&r->commands.float_data, float, direction.x);
    DB_PUSH(&r->commands.float_data, float, direction.y);
    DB_PUSH(&r->commands.float_data, float, sinf(aperture));
    DB_PUSH(&r->commands.float_data, float, cosf(aperture));
    DB_PUSH(&r->commands.float_data, float, thickness);

    quantized_aabb quantized_bb = quantized_aabb_make(bb.min.x, bb.min.y, bb.max.x, bb.max.y);
    merge_quantized_aabb(r->commands.group_aabb, &quantized_bb);
    DB_PUSH(&r->commands.aabb, quantized_aabb, quantized_bb);
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_draw_box(struct onedraw* r, float x0, float y0, float x1, float y1, float radius, draw_color srgb_color)
{
    if ((r->commands.float_data.num_elements + 5 >= r->commands.float_data.num_elements_max) ||
        (r->commands.colors.num_elements >= r->commands.colors.num_elements_max) ||
        (r->commands.list.num_elements >= r->commands.list.num_elements_max) ||
        (r->commands.aabb.num_elements >= r->commands.aabb.num_elements_max))
    {
        od_log(r, "buffers for primitive are full, expect graphical artefacts");
        return;
    }

    if (x0>x1) 
        float_swap(&x0, &x1);
    if (y0>y1) 
        float_swap(&y0, &y1);

    aabb box = {.min = {x0, y0}, .max = {x1, y1}};
    vec2 center = vec2_scale(vec2_add(box.min, box.max), .5f);
    vec2 half_extents = vec2_scale(vec2_sub(box.max, box.min), .5f);

    gpu_draw_command cmd = gpu_draw_command_make(r->commands.float_data.num_elements, 0, LAST_CLIP_INDEX, fill_solid, primitive_aabox);
    DB_PUSH(&r->commands.list, gpu_draw_command, cmd);
    DB_PUSH(&r->commands.colors, draw_color, srgb_color);
    DB_PUSH(&r->commands.float_data, float, center.x);
    DB_PUSH(&r->commands.float_data, float, center.y);
    DB_PUSH(&r->commands.float_data, float, half_extents.x);
    DB_PUSH(&r->commands.float_data, float, half_extents.y);
    DB_PUSH(&r->commands.float_data, float, radius);

    aabb_grow(&box, vec2_splat(draw_cmd_aabb_bump(r)));
    quantized_aabb quantized_box = quantized_aabb_make(box.min.x, box.min.y, box.max.x, box.max.y);
    merge_quantized_aabb(r->commands.group_aabb, &quantized_box);
    DB_PUSH(&r->commands.aabb, quantized_aabb, quantized_box);
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_draw_blurred_box(struct onedraw* r, float cx, float cy, float width, float height, float roundness, draw_color srgb_color)
{
    if ((r->commands.float_data.num_elements + 5 >= r->commands.float_data.num_elements_max) ||
        (r->commands.colors.num_elements >= r->commands.colors.num_elements_max) ||
        (r->commands.list.num_elements >= r->commands.list.num_elements_max) ||
        (r->commands.aabb.num_elements >= r->commands.aabb.num_elements_max))
    {
        od_log(r, "buffers for primitive are full, expect graphical artefacts");
        return;
    }

    float half_width = width * .5f;
    float half_height = height * .5f;

    gpu_draw_command cmd = gpu_draw_command_make(r->commands.float_data.num_elements, 0, LAST_CLIP_INDEX, fill_solid, primitive_blurred_box);
    DB_PUSH(&r->commands.list, gpu_draw_command, cmd);
    DB_PUSH(&r->commands.colors, draw_color, srgb_color);
    DB_PUSH(&r->commands.float_data, float, cx);
    DB_PUSH(&r->commands.float_data, float, cy);
    DB_PUSH(&r->commands.float_data, float, half_width);
    DB_PUSH(&r->commands.float_data, float, half_height);
    DB_PUSH(&r->commands.float_data, float, roundness);

    aabb box =
    {
        .min = {cx - half_width - roundness, cy - half_height - roundness},
        .max = {cx + half_width + roundness, cy + half_height + roundness}
    };

    quantized_aabb quantized_box = quantized_aabb_make(box.min.x, box.min.y, box.max.x, box.max.y);
    merge_quantized_aabb(r->commands.group_aabb, &quantized_box);
    DB_PUSH(&r->commands.aabb, quantized_aabb, quantized_box);
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_draw_char(struct onedraw* r, float x, float y, char c, draw_color srgb_color)
{
    if (c < r->font.desc.first_glyph || c > (r->font.desc.first_glyph + r->font.desc.num_glyphs))
        return;

    if ((r->commands.float_data.num_elements + 2 >= r->commands.float_data.num_elements_max) ||
        (r->commands.colors.num_elements >= r->commands.colors.num_elements_max) ||
        (r->commands.list.num_elements >= r->commands.list.num_elements_max) ||
        (r->commands.aabb.num_elements >= r->commands.aabb.num_elements_max))
    {
        od_log(r, "buffers for primitive are full, expect graphical artefacts");
        return;
    }

    uint32_t glyph_index = c - r->font.desc.first_glyph;
    const od_glyph* glyph = &r->font.desc.glyphs[glyph_index];
    x += glyph->bearing_x;
    y += glyph->bearing_y + r->font.desc.font_height;
    float glyph_width = (float)(glyph->x1 - glyph->x0);
    float glyph_height = (float)(glyph->y1 - glyph->y0);

    gpu_draw_command cmd = gpu_draw_command_make(r->commands.float_data.num_elements, (uint8_t) glyph_index, LAST_CLIP_INDEX, fill_solid, primitive_char);
    DB_PUSH(&r->commands.list, gpu_draw_command, cmd);
    DB_PUSH(&r->commands.colors, draw_color, srgb_color);
    DB_PUSH(&r->commands.float_data, float, x);
    DB_PUSH(&r->commands.float_data, float, y);

    quantized_aabb aabb = quantized_aabb_make(x, y, x + glyph_width, y + glyph_height);
    merge_quantized_aabb(r->commands.group_aabb, &aabb);
    DB_PUSH(&r->commands.aabb, quantized_aabb, aabb);
}

//-----------------------------------------------------------------------------------------------------------------------------
void od_draw_text(struct onedraw* r, float x, float y, const char* text, draw_color srgb_color)
{
    float left = x;
    for(const char *c = text; *c != 0; c++)
    {
        if (*c == '\n')
        {
            y += r->font.desc.font_height;
            x = left;
        }
        else if (*c >= r->font.desc.first_glyph && *c <= (r->font.desc.first_glyph + r->font.desc.num_glyphs))
        {
            od_draw_char(r, x, y, *c, srgb_color);
            uint32_t glyph_index = *c - r->font.desc.first_glyph;
            x += r->font.desc.glyphs[glyph_index].advance_x;
        }
        else
            x += r->font.desc.glyphs['_'- r->font.desc.first_glyph].advance_x * .65f;
    }
}

//-----------------------------------------------------------------------------------------------------------------------------
// void od_draw_quad(struct onedraw* r, float x0, float y0, float x1, float y1, od_quad_uv uv, uint32_t slice_index, draw_color srgb_color)
// {

// }

//-----------------------------------------------------------------------------------------------------------------------------
// void od_draw_oriented_quad(struct onedraw* r, float cx, float cy, float width, float height, float angle, od_quad_uv uv, uint32_t slice_index, draw_color srgb_color)
// {

// }

//-----------------------------------------------------------------------------------------------------------------------------
// Breaks the bezier quadratic curve into multiple capsules, using De Casteljau’s algorithm and colinear detection
uint32_t od_draw_quadratic_bezier(struct onedraw* r, const float* control_points, float width, draw_color srgb_color)
{
    quadratic_bezier stack[TESSELATION_STACK_MAX];
    uint32_t stack_index = 0;

    const float radius = width * .5f;
    uint32_t num_capsules = 0;

    stack[stack_index++] = (quadratic_bezier)
    {
        .c0 = {control_points[0], control_points[1]},
        .c1 = {control_points[2], control_points[3]},
        .c2 = {control_points[4], control_points[5]},
    };

    while (stack_index != 0)
    {
        quadratic_bezier c = stack[--stack_index];

        // splits proportionally to segment lengths
        float d0 = vec2_distance(c.c0, c.c1);
        float d1 = vec2_distance(c.c1, c.c2);
        float split = d0 / (d0 + d1);

        vec2 left = vec2_lerp( c.c0, c.c1, split);
        vec2 right = vec2_lerp(c.c1, c.c2, split);
        vec2 middle = vec2_lerp(left, right, split);
        
        if (is_colinear(c.c0, c.c2, middle, COLINEAR_THRESHOLD))
        {
            od_draw_capsule(r, c.c0.x, c.c0.y, c.c2.x, c.c2.y, radius, srgb_color);
            num_capsules++;
        }
        else
        {
            if (stack_index + 2 <= TESSELATION_STACK_MAX)
            {
                stack[stack_index++] = (quadratic_bezier)
                {
                    .c0 = c.c0,
                    .c1 = left,
                    .c2 = middle,
                };

                stack[stack_index++] = (quadratic_bezier)
                {
                    .c0 = middle,
                    .c1 = right,
                    .c2 = c.c2,
                };
            }
            else
                return UINT32_MAX;
        }
    }
    return num_capsules;
}

//----------------------------------------------------------------------------------------------------------------------------
uint32_t od_draw_cubic_bezier(struct onedraw* r, const float* control_points, float width, draw_color srgb_color)
{
    cubic_bezier stack[TESSELATION_STACK_MAX];
    uint32_t stack_index = 0;

    const float radius = width * .5f;
    uint32_t num_capsules = 0;

    stack[stack_index++] = (cubic_bezier)
    {
        .c0 = {control_points[0], control_points[1]},
        .c1 = {control_points[2], control_points[3]},
        .c2 = {control_points[4], control_points[5]},
        .c3 = {control_points[6], control_points[7]}
    };

    while (stack_index != 0)
    {
        cubic_bezier c = stack[--stack_index];

        // the halfway point along the control polygon roughly corresponds to halfway along the curve arc length
        float d0 = vec2_distance(c.c0, c.c1);
        float d1 = vec2_distance(c.c1, c.c2);
        float d2 = vec2_distance(c.c2, c.c3);
        float total = d0 + d1 + d2;
        float split = (d0 + 0.5f * d1) / total;

        vec2 c01 = vec2_lerp(c.c0, c.c1, split);
        vec2 c12 = vec2_lerp(c.c1, c.c2, split);
        vec2 c23 = vec2_lerp(c.c2, c.c3, split);
        vec2 c01c12 = vec2_lerp(c01, c12, split);
        vec2 c12c23 = vec2_lerp(c12, c23, split);
        vec2 middle = vec2_lerp(c01c12, c12c23, split);

        if (is_colinear(c.c0, c.c3, middle, COLINEAR_THRESHOLD))
        {
            od_draw_capsule(r, c.c0.x, c.c0.y, c.c2.x, c.c2.y, radius, srgb_color);
            num_capsules++;
        }
        else
        {
            if (stack_index + 2 <= TESSELATION_STACK_MAX)
            {
                stack[stack_index++] = (cubic_bezier)
                {
                    .c0 = c.c0,
                    .c1 = c01,
                    .c2 = c01c12,
                    .c3 = middle
                };

                stack[stack_index++] = (cubic_bezier)
                {
                    .c0 = middle,
                    .c1 = c12c23,
                    .c2 = c23,
                    .c3 = c.c3,
                };
            }
            else
                return UINT32_MAX;
        }
    }

    return num_capsules;
}
