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

// ---------------------------------------------------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------------------------------------------------

#define STRING_BUFFER_SIZE (512U)
#define VEC2_SQR2 (1.41421356237f)
#define HALF_PIXEL (.5f)
#define VEC2_PI (3.14159265f)
#define TESSELATION_STACK_MAX (1024U)
#define COLINEAR_THRESHOLD (.1f)

// ---------------------------------------------------------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------------------------------------------------------

#define assert_msg(expr, msg) assert((expr) && (msg))

// ---------------------------------------------------------------------------------------------------------------------------
// Private structures
// ---------------------------------------------------------------------------------------------------------------------------

struct onedraw
{
    WGPUDevice device;



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

//----------------------------------------------------------------------------------------------------------------------------
struct onedraw* od_init(onedraw_def* def)
{
    assert_msg(def->preallocated_buffer != NULL, "forgot to allocate memory?");
    assert_msg(((uintptr_t)def->preallocated_buffer)%sizeof(uintptr_t) == 0, "preallocated_buffer must be aligned on sizeof(uintptr_t)");
    assert(def->device != NULL);

    struct onedraw* r = (struct onedraw*) def->preallocated_buffer;

    r->custom_log = def->log_func;
    r->device = def->device;
    
    // od_build_pso(r);
    // od_build_font(r);
    // od_build_depthstencil_state(r);
    // od_resize(r, def->viewport_width, def->viewport_height);

    // if (def->atlas.width != 0)
    //     od_create_atlas(r, def->atlas.width, def->atlas.height, def->atlas.num_slices);

    return r;
}

//----------------------------------------------------------------------------------------------------------------------------
void od_terminate(struct onedraw* r)
{

}