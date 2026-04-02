// ---------------------------------------------------------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------------------------------------------------------

struct draw_command 
{
    data_index : u32,
    flags      : u32 // extra (8) | clip_index (8) | fillmode (8) | type (8)
};

struct tile_node
{
    next          : u32,
    command_index : u32 // command index + command type
};

struct counters
{
    num_nodes : atomic<u32>,
    num_tiles : atomic<u32>
};

struct draw_args 
{
    clear_color: vec4<f32>,
    screen_div: vec2<f32>,
    num_commands: u32,
    num_tile_width: u32,
    num_tile_height: u32,
    max_nodes: u32,
    aa_width: f32,
    options: u32,
};

struct glyph
{
    uv_topleft : vec2<f32>,
    uv_bottomright : vec2<f32>,
    width : f32,
    height : f32
};

struct clip_rect
{
    min_x : f32,
    min_y : f32,
    max_x : f32,
    max_y : f32
};

struct aabb
{
    min : vec2<f32>,
    max : vec2<f32>
};

struct obb
{
    axis_i : vec2<f32>,
    axis_j : vec2<f32>,
    center : vec2<f32>,
    extents : vec2<f32>
}

struct indirect_params 
{
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32
};


// ---------------------------------------------------------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------------------------------------------------------

fn get_extra(cmd: draw_command) -> u32  {return cmd.flags & 0xFFu;}
fn get_clip(cmd: draw_command) -> u32 {return (cmd.flags >> 8u) & 0xFFu;}
fn get_fillmode(cmd: draw_command) -> u32 {return (cmd.flags >> 16u) & 0xFFu;}
fn get_type(cmd: draw_command) -> u32 {return (cmd.flags >> 24u) & 0xFFu;}
fn get_command_index(n : tile_node) -> u32 {return n.command_index & 0xFFFFu;}
fn get_command_type(n : tile_node) -> u32 {return (n.command_index >> 16u) & 0xFFu;}


/*

//-----------------------------------------------------------------------------
// based on https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-23-high-speed-screen-particles
// we also use a specific blend equation
half4 accumulate_color(half4 color, half4 backbuffer)
{
    half4 output;
    output.rgb = (color.rgb * color.a) + (backbuffer.rgb * (1.f - color.a));
    output.a = backbuffer.a * (1.f - color.a);
    return output;
}

*/

fn accumulate_color(color: vec4<f32>, backbuffer: vec4<f32>) -> vec4<f32> 
{
    let rgb = mix(backbuffer.rgb, color.rgb, color.a);
    return vec4<f32>(rgb, 1.0);
}

fn linear_to_srgb_channel(c: f32) -> f32 
{
    if (c <= 0.0031308) 
    {
        return c * 12.92;
    } else 
    {
        return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    }
}

fn linear_to_srgb(linear_color: vec4<f32>) -> vec4<f32> 
{
    return vec4<f32>(
        linear_to_srgb_channel(linear_color.r),
        linear_to_srgb_channel(linear_color.g),
        linear_to_srgb_channel(linear_color.b),
        linear_color.a
    );
}

fn srgb_to_linear(srgb: vec4<f32>) -> vec4<f32> 
{
    let rgb = srgb.rgb;
    let linear_rgb = select(
        pow((rgb + 0.055) / 1.055, vec3<f32>(2.4)),
        rgb / 12.92,
        rgb <= vec3<f32>(0.04045)
    );
    return vec4<f32>(linear_rgb, srgb.a);
}

fn saturate(x: f32) -> f32 
{
    return clamp(x, 0.0, 1.0);
}

fn linearstep(edge0: f32, edge1: f32, x: f32) -> f32 
{
    return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}

fn skew(v: vec2<f32>) -> vec2<f32> 
{
    return vec2<f32>(-v.y, v.x);
}

fn distance_squared(a: vec2<f32>, b: vec2<f32>) -> f32 
{
    let d = a - b;
    return dot(d, d);
}

fn length_squared(v: vec2<f32>) -> f32 
{
    return dot(v, v);
}

fn square(x: f32) -> f32 
{
    return x * x;
}

// ---------------------------------------------------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------------------------------------------------

const TILE_SIZE: f32 = 16.0;
const PRIMITIVE_CHAR: u32 = 0u;
const PRIMITIVE_AABOX: u32 = 1u;
const PRIMITIVE_ORIENTED_BOX: u32 = 2u;
const PRIMITIVE_DISC: u32 = 3u;
const PRIMITIVE_TRIANGLE: u32 = 4u;
const PRIMITIVE_ELLIPSE: u32 = 5u;
const PRIMITIVE_PIE: u32 = 6u;
const PRIMITIVE_ARC: u32 = 7u;
const PRIMITIVE_BLURRED_BOX: u32 = 8u;
const PRIMITIVE_QUAD: u32 = 9u;
const PRIMITIVE_ORIENTED_QUAD: u32 = 10u;
const BEGIN_GROUP: u32 = 32u;
const END_GROUP: u32 = 33u;
const INVALID_INDEX:u32 = 0xFFFFFFFFu;
const OP_OVERWRITE:u32 = 0u;
const OP_BLEND:u32 = 1u;
const FILL_SOLID:u32 = 0u;
const FILL_OUTLINE:u32 = 1u;
const FILL_HOLLOW:u32 = 2u;
const FILL_GRADIENT:u32 = 3u;
const OPTION_SRGB_BACKBUFFER:u32 = (1u<<0);
const OPTION_DEBUG_BINNING:u32 = (1u<<1);

