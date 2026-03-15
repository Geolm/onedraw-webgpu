// ---------------------------------------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------------------------------------

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

fn linear_to_srgb_channel(c: f32) -> f32 
{
    if (c <= 0.0031308) 
    {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(color: vec4<f32>) -> vec4<f32> 
{
    return vec4<f32>
    (
        linear_to_srgb_channel(color.r),
        linear_to_srgb_channel(color.g),
        linear_to_srgb_channel(color.b),
        color.a
    );
}

// ---------------------------------------------------------------------------------------------------------------------------
// Signed distance functions
// ---------------------------------------------------------------------------------------------------------------------------

fn erf_approx(x: f32) -> f32 
{
    return sign(x) * sqrt(1.0 - exp2(-1.787776 * x * x));
}

fn sd_disc(position: vec2<f32>, center: vec2<f32>, radius: f32) -> f32 
{
    return length(center - position) - radius;
}

fn sd_aabox(position: vec2<f32>, box_center: vec2<f32>, half_extents: vec2<f32>, radius: f32) -> f32 
{
    var p = abs(position - box_center) - half_extents + vec2<f32>(radius);
    return length(max(p, vec2<f32>(0.0))) +
           min(max(p.x, p.y), 0.0) - radius;
}

fn sd_oriented_box(position: vec2<f32>, a: vec2<f32>, b: vec2<f32>, width: f32) -> f32 
{
    let l = length(b - a);
    let d = (b - a) / l;
    var q = position - (a + b) * 0.5;

    let rot = mat2x2<f32>(d.x, -d.y, d.y,  d.x);

    q = rot * q;
    q = abs(q) - vec2<f32>(l, width) * 0.5;

    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

fn sd_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 
{
    let pa = p - a;
    let ba = b - a;
    let h = saturate(dot(pa, ba) / dot(ba, ba));
    return length(pa - ba * h);
}

fn smooth_minimum(a_in: f32, b_in: f32, k: f32) -> vec2<f32> 
{
    var a = a_in;
    var b = max(b_in, 0.0);

    if (k > 0.0) 
    {
        let h = max(k - abs(a - b), 0.0) / k;
        let m = h * h * h * 0.5;
        let s = m * k * (1.0 / 3.0);

        if (a < b) 
        {
            return vec2<f32>(a - s, 0.0);
        } 
        else 
        {
            return vec2<f32>(b - s, 1.0 - linearstep(-k, 0.0, b - a));
        }
    }

    return vec2<f32>(min(a, b), select(1.0, 0.0, a < b));
}

// ---------------------------------------------------------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------------------------------------------------------

struct vs_out 
{
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) tile_index: u32
};

@vertex
fn tile_vs(@builtin(instance_index) instance_id: u32, @builtin(vertex_index) vertex_id: u32) -> vs_out
{
    var out: vs_out;

    let tile_index = g_tile_indices[instance_id];
    let tile_x = tile_index % g_draw_args.num_tile_width;
    let tile_y = tile_index / g_draw_args.num_tile_width;

    var screen_pos = vec2<f32>(f32(vertex_id & 1u), f32(vertex_id >> 1u));

    screen_pos += vec2<f32>(f32(tile_x), f32(tile_y));
    screen_pos *= TILE_SIZE;

    var clipspace = screen_pos * g_draw_args.screen_div;
    clipspace = clipspace * 2.0 - vec2<f32>(1.0);
    clipspace.y = -clipspace.y;

    out.pos = vec4<f32>(clipspace, 0.0, 1.0);
    out.tile_index = tile_index;

    return out;
}

// ---------------------------------------------------------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------------------------------------------------------

@fragment
fn tile_fs(in: vs_out) -> @location(0) vec4<f32> 
{

    var output = vec4<f32>(0.0);

    return output;
}