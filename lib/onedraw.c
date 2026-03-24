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
#define FRAME_COUNT (3)
#define TILE_SIZE (16)
#define MAX_NODES_COUNT (1U<<22)
#define MAX_COMMANDS (1U<<16)

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

// ---------------------------------------------------------------------------------------------------------------------------
// Private structures
// ---------------------------------------------------------------------------------------------------------------------------

typedef struct {float x, y, z, w;} vec4;
typedef uint32_t quantized_aabb;
typedef enum sdf_operator {op_additive, op_subtractive} sdf_operator;

typedef struct dynamic_buffer
{
    WGPUBuffer buffers[FRAME_COUNT];
    size_t element_size;
    size_t num_elements;
    size_t num_elements_max;
    uint8_t* cpu_buffer;
} dynamic_buffer;

struct onedraw
{
    WGPUDevice device;
    WGPUQueue queue;
    WGPUCommandBuffer command_buffer;

    struct
    {
        dynamic_buffer draw_args;
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
        WGPUBuffer heads; 
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
        vec4 clear_color;
        uint32_t width;
        uint32_t height;
        float aa_width;
        sdf_operator group_op;
        float outline_width;
        bool srgb_backbuffer;
        bool clear_backbuffer;
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
        WGPUBindGroup rasterizer_group;
        WGPUBindGroup binning_group;
        WGPUBindGroup dynamic_group[FRAME_COUNT];
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

// must be in sync with common.wgsl
typedef struct gpu_draw_args
{
    uint32_t num_commands;
    uint32_t num_tile_width;
    uint32_t num_tile_height;
    uint32_t max_nodes;
    vec2 screen_div;
    float aa_width;
    vec4 clear_color;
    uint32_t srgb_backbuffer;
} gpu_draw_args;

typedef struct gpu_char
{
    vec2 uv_topleft;
    vec2 uv_bottomright;
    float width;
    float height;
} gpu_char;

typedef struct indirect_params
{
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
} indirect_params;


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
void dynamic_buffer_init(struct onedraw* r, dynamic_buffer* b, size_t element_size, size_t num_elements_max)
{
    b->element_size = element_size;
    b->num_elements = 0;
    b->num_elements_max = num_elements_max;

    size_t alloc_size = element_size * num_elements_max;
    for(uint32_t i=0; i<FRAME_COUNT; ++i)
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

    r->binding.rasterizer_group =  wgpuDeviceCreateBindGroup(r->device, &(WGPUBindGroupDescriptor)
    {
        .label = WGPU_STRING_VIEW("rasterizer_group"),
        .layout = r->binding.rasterizer_layout,
        .entryCount = ARRAY_SIZE(rasterizer_entries),
        .entries = rasterizer_entries
    });
    assert_msg(r->binding.rasterizer_group != NULL, "cannot create rasterizer binding group");

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

    r->binding.binning_group =  wgpuDeviceCreateBindGroup(r->device, &(WGPUBindGroupDescriptor)
    {
        .label = WGPU_STRING_VIEW("binning_group"),
        .layout = r->binding.binning_layout,
        .entryCount = ARRAY_SIZE(binning_entries),
        .entries = binning_entries
    });
    assert_msg(r->binding.binning_group != NULL, "cannot create rasterizer binding group");

    // for(uint32_t i=0; i<FRAME_COUNT; ++i)
    // {
    //     WGPUBindGroupEntry frame_entries[] = 
    //     {
    //         {.binding = 0, .buffer = r->commands.draw_args.buffers[i], .size = WGPU_WHOLE_SIZE},
    //         {.binding = 1, .buffer = r->tiles.indices, .size = WGPU_WHOLE_SIZE},
    //         {.binding = 2, .buffer = r->tiles.counters, .size = WGPU_WHOLE_SIZE},
    //         {.binding = 3, .buffer = r->tiles.heads, .size = WGPU_WHOLE_SIZE},
    //         {.binding = 4, .buffer = r->tiles.indirect_draw_params, .size = WGPU_WHOLE_SIZE}
    //     };
    // }
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

    WGPUColorTargetState color_target = 
    {
        .format = surface_format,
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
            .topology = WGPUPrimitiveTopology_TriangleList,
            .frontFace = WGPUFrontFace_CCW,
            .cullMode = WGPUCullMode_None
        },

        .fragment = &fragment,
        .multisample = {.count = 1, .mask = ~0u}
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

    WGPUComputePipelineDescriptor compute_pipeline_desc = 
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
        .maxAnisotropy = 1.f
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
        .mappedAtCreation = true,
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
        .label = WGPU_STRING_VIEW("gpu_font")
    });

    assert_msg(r->font.glyphs != NULL, "can create font description gpu buffer");
    void* mapping = wgpuBufferGetMappedRange(r->font.glyphs, 0, gpu_font_size);
    memcpy(mapping, cpu_buffer, gpu_font_size);
    wgpuBufferUnmap(r->font.glyphs);
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
        .maxAnisotropy = 1.f
    };

    r->atlas.sampler = wgpuDeviceCreateSampler(r->device, &sampler_desc);
    assert_msg(r->atlas.sampler != NULL, "cannot create atlas sampler");
}

//-----------------------------------------------------------------------------------------------------------------------------
void allocate_dynamic_buffers(struct onedraw* r)
{
    dynamic_buffer_init(r, &r->commands.draw_args, sizeof(gpu_draw_args), 1);
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
    r->rasterizer.srgb_backbuffer = def->srgb_backbuffer;
    r->rasterizer.aa_width = VEC2_SQR2;
    r->rasterizer.clear_backbuffer = true;
    r->rasterizer.clear_color = (vec4) {.x = 0.f, .y = 0.f, .z = 0.f, .w = 1.f};
    r->queue = wgpuDeviceGetQueue(r->device);
    assert_msg(r->queue != NULL, "cannot create queue from device");

    // resource creation
    r->tiles.counters = wgpuDeviceCreateBuffer(r->device, &(WGPUBufferDescriptor)
    {
        .label = WGPU_STRING_VIEW("tile_counters"),
        .mappedAtCreation = false,
        .size = sizeof(uint32_t) * 2,
        .usage = WGPUBufferUsage_Storage
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
        .size = sizeof(indirect_params),
        .usage = WGPUBufferUsage_Storage
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