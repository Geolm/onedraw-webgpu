// ---------------------------------------------------------------------------------------------------------------------------
// Common structures and bindgroups
// ---------------------------------------------------------------------------------------------------------------------------

struct draw_command 
{
    data_index : u32,
    flags      : u32 // extra (8) | clip_index (8) | fillmode (8) | type (8)
};

fn get_extra(cmd: draw_command) -> u32  {return cmd.flags & 0xFFu;}
fn get_clip(cmd: draw_command) -> u32 {return (cmd.flags >> 8u) & 0xFFu;}
fn get_fillmode(cmd: draw_command) -> u32 {return (cmd.flags >> 16u) & 0xFFu;}
fn get_type(cmd: draw_command) -> u32 {return (cmd.flags >> 24u) & 0xFFu;}

struct tile_node
{
    next          : u32,
    command_index : u32 // command index + command type
};

fn get_command_index(n : tile_node) -> u32 {return n.command_index & 0xFFFFu;}
fn get_command_type(n : tile_node) -> u32 {return (n.command_index >> 16u) & 0xFFu;}

struct counters
{
    num_nodes : atomic<u32>,
    num_tiles : atomic<u32>
};

struct draw_args 
{
    num_commands: u32,
    num_tile_width: u32,
    num_tile_height: u32,
    max_nodes: u32,
    screen_div: vec2<f32>,
    aa_width: f32
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

// group 0 : static or updated on gpu
@group(0) @binding(0)
var<storage, read_write> g_tile_nodes : array<tile_node>;

@group(0) @binding(1)
var<storage, read_write> g_tile_indices : array<u32>;

@group(0) @binding(2)
var<storage, read_write> g_counters : counters;

@group(0) @binding(3)
var<storage, read_write> g_glyphs : array<glyph>;

// group 1 : updated each frame
@group(1) @binding(0)
var<uniform> g_draw_args: draw_args;

@group(1) @binding(1)
var<storage, read> g_commands : array<draw_command>;

@group(1) @binding(2)
var<storage, read> g_quantized_aabb : array<u32>;

@group(1) @binding(3)
var<storage> g_draw_data: array<f32>;

@group(1) @binding(4)
var<storage> g_clips: array<clip_rect>;

@group(1) @binding(5)
var<storage> g_colors: array<u32>;


// ---------------------------------------------------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------------------------------------------------

const TILE_SIZE: f32 = 16.0;


