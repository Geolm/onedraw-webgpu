// ---------------------------------------------------------------------------------------------------------------------------
// Bindgroups
// ---------------------------------------------------------------------------------------------------------------------------

// group 0 : static or updated on gpu
@group(0) @binding(0)
var<storage, read> g_tile_nodes : array<tile_node>;

@group(0) @binding(1)
var<storage, read> g_tile_indices : array<u32>;

@group(0) @binding(2)
var<storage, read> g_glyphs : array<glyph>;

@group(0) @binding(3)
var<storage, read> g_tile_heads : array<u32>;

@group(0) @binding(4) 
var g_atlas: texture_2d_array<f32>;

@group(0) @binding(5) 
var g_atlas_sampler: sampler;

@group(0) @binding(6) 
var g_font: texture_2d<f32>;

@group(0) @binding(7) 
var g_font_sampler: sampler;

// group 1 : updated each frame
@group(1) @binding(0)
var<storage, read> g_draw_args: draw_args;

@group(1) @binding(1)
var<storage, read> g_commands : array<draw_command>;

@group(1) @binding(2)
var<storage, read> g_quantized_aabb : array<u32>;

@group(1) @binding(3)
var<storage, read> g_draw_data: array<f32>;

@group(1) @binding(4)
var<storage, read> g_clips: array<clip_rect>;

@group(1) @binding(5)
var<storage, read> g_colors: array<u32>;

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

fn sd_triangle(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 
{
    let e0 = p1 - p0;
    let e1 = p2 - p1;
    let e2 = p0 - p2;

    let v0 = p - p0;
    let v1 = p - p1;
    let v2 = p - p2;

    let pq0 = v0 - e0 * saturate(dot(v0, e0) / dot(e0, e0));
    let pq1 = v1 - e1 * saturate(dot(v1, e1) / dot(e1, e1));
    let pq2 = v2 - e2 * saturate(dot(v2, e2) / dot(e2, e2));
    
    let s = e0.x * e2.y - e0.y * e2.x;
    
    let d = min(min(vec2<f32>(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
                    vec2<f32>(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
                    vec2<f32>(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));

    return -sqrt(d.x) * sign(d.y);
}

fn sd_gaussian_box(position_in: vec2<f32>, box_center: vec2<f32>, box_size: vec2<f32>, radius: f32) -> vec2<f32> 
{
    let p = position_in - box_center;
    let d = abs(p) - box_size;
    let sd = length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0) - radius;
    
    let blur_radius = radius * 0.5;

    let u = erf_approx((p.x + box_size.x) / blur_radius) - erf_approx((p.x - box_size.x) / blur_radius);
    let v = erf_approx((p.y + box_size.y) / blur_radius) - erf_approx((p.y - box_size.y) / blur_radius);
    
    return vec2<f32>(sd, u * v / 4.0);
}

fn sd_ellipse(p: vec2<f32>, e: vec2<f32>) -> f32 
{
    let pAbs = abs(p);
    let ei = 1.0 / e;
    let e2 = e * e;
    let ve = ei * vec2<f32>(e2.x - e2.y, e2.y - e2.x);
    
    var t = vec2<f32>(0.70710678118, 0.70710678118);

    // Iterative refinement to find the nearest point on the ellipse
    for (var i: i32 = 0; i < 3; i++) 
    {
        let v = ve * t * t * t;
        let u = normalize(pAbs - v) * length(t * e - v);
        let w = ei * (v + u);
        t = normalize(clamp(w, vec2<f32>(0.0), vec2<f32>(1.0)));
    }
    
    let nearestAbs = t * e;
    let dist = length(pAbs - nearestAbs);
    return select(dist, -dist, dot(pAbs, pAbs) < dot(nearestAbs, nearestAbs));
}

fn sd_oriented_ellipse(position: vec2<f32>, a: vec2<f32>, b: vec2<f32>, width: f32) -> f32 
{
    let height = length(b - a);
    let axis = (b - a) / height;
    let pos_translated = position - (a + b) * 0.5;

    let rot = mat2x2<f32>(vec2<f32>(axis.x, axis.y),vec2<f32>(-axis.y, axis.x));
    let pos_boxspace = pos_translated * rot;

    return sd_ellipse(pos_boxspace, vec2<f32>(height * 0.5, width * 0.5));
}

fn sd_oriented_pie(position_in: vec2<f32>, center: vec2<f32>, direction_in: vec2<f32>, aperture: vec2<f32>, radius: f32) -> f32 
{
    let dir = -skew(direction_in);
    let p_rel = position_in - center;
    
    let rot = mat2x2<f32>(vec2<f32>(dir.x, dir.y),vec2<f32>(-dir.y, dir.x));
    let p = p_rel * rot;
    
    let px_abs = abs(p.x);
    let l = length(p) - radius;
    let m = length(vec2<f32>(px_abs, p.y) - aperture * clamp(dot(vec2<f32>(px_abs, p.y), aperture), 0.0, radius));
    return max(l, m * sign(aperture.y * px_abs - aperture.x * p.y));
}

fn sd_oriented_arc(position_in: vec2<f32>, center: vec2<f32>, direction_in: vec2<f32>, aperture: vec2<f32>, radius: f32, thickness: f32) -> f32 
{
    let dir = -skew(direction_in);
    var p_rel = position_in - center;
    let rot = mat2x2<f32>(vec2<f32>(dir.x, dir.y),vec2<f32>(-dir.y, dir.x));
    var p = p_rel * rot;

    p.x = abs(p.x);
    
    // Rotate by aperture matrix
    p = vec2<f32>(-aperture.y * p.x - aperture.x * p.y,aperture.x * p.x - aperture.y * p.y);
    
    let half_thick = thickness * 0.5;
    let d1 = abs(length(p) - radius) - half_thick;
    let d2 = length(vec2<f32>(p.x, max(0.0, abs(radius - p.y) - half_thick))) * sign(p.x);
    
    return max(d1, d2);
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
    var output:vec4<f32> = g_draw_args.clear_color;

    if ((g_draw_args.options & OPTION_DEBUG_BINNING) != 0u)
    {
        output = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    }
 
    var node_idx:u32 = g_tile_heads[in.tile_index];
    if (node_idx == INVALID_INDEX)
    {
        return output;
    }

    var previous_distance: f32 = 1e8;
    var previous_color: vec4<f32> = vec4<f32>(0.0);
    var group_smoothness: f32 = 0.0;
    var group_op: u32 = OP_OVERWRITE;
    var grouping: bool = false;
    var outline_width: f32 = 0.0;

    while (node_idx != INVALID_INDEX)
    {
        let node = g_tile_nodes[node_idx];
        let cmd_idx = get_command_index(node);
        let cmd = g_commands[cmd_idx];
        
        // Extract info using your helpers
        let cmd_type = get_type(cmd);
        let fillmode = get_fillmode(cmd);
        let extra    = get_extra(cmd);
        let clip_idx = get_clip(cmd);
        let data_idx = cmd.data_index;
        let clip       = g_clips[clip_idx];

        var cmd_color: vec4<f32> = srgb_to_linear(unpack4x8unorm(g_colors[cmd_idx]));

        let clipped = (in.pos.x < clip.min_x || in.pos.y < clip.min_y || 
                                in.pos.x > clip.max_x || in.pos.y > clip.max_y);

        if (!clipped)
        {
            var distance: f32 = 10.0;

            if (cmd_type == BEGIN_GROUP)
            {
                previous_color = vec4<f32>(0.0);
                previous_distance = 1e8;
                group_smoothness = g_draw_data[data_idx + 0u];
                grouping = true;
                group_op = extra;
                outline_width = g_draw_data[data_idx + 1u];
            }
            else
            {
                switch (cmd_type)
                {
                    case PRIMITIVE_DISC:
                    {
                        let center = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
                        let radius = g_draw_data[data_idx + 2u];
                        distance = sd_disc(in.pos.xy, center, radius);
                        
                        if (fillmode == FILL_HOLLOW) 
                        {
                            distance = abs(distance) - g_draw_data[data_idx + 3u];
                        } 
                        else if (fillmode == FILL_GRADIENT) 
                        {
                            let packed_color = bitcast<u32>(g_draw_data[data_idx + 3u]);
                            let inner_color = srgb_to_linear(unpack4x8unorm(packed_color));
                            cmd_color = mix(inner_color, cmd_color, linearstep(-radius, 0.0, distance));
                        }
                    }
                    case PRIMITIVE_ORIENTED_BOX:
                    {
                        let p0 = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
                        let p1 = vec2<f32>(g_draw_data[data_idx + 2u], g_draw_data[data_idx + 3u]);
                        let width = g_draw_data[data_idx + 4u];

                        if (width == 0.0) 
                        {
                            distance = sd_segment(in.pos.xy, p0, p1);
                        } else 
                        {
                            distance = sd_oriented_box(in.pos.xy, p0, p1, width);
                        }

                        if (fillmode == FILL_HOLLOW) 
                        {
                            distance = abs(distance);
                        } 
                        else if (fillmode == FILL_GRADIENT) 
                        {
                            let packed_color = bitcast<u32>(g_draw_data[data_idx + 6u]);
                            let inner_color = srgb_to_linear(unpack4x8unorm(packed_color));
                            let pa = in.pos.xy - p0;
                            let ba = p1 - p0;
                            let h = saturate(dot(pa, ba) / dot(ba, ba));
                            cmd_color = mix(inner_color, cmd_color, h);
                        }
                        distance -= g_draw_data[data_idx + 5u];
                    }
                    case PRIMITIVE_ELLIPSE:
                    {
                        let p0 = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
                        let p1 = vec2<f32>(g_draw_data[data_idx + 2u], g_draw_data[data_idx + 3u]);
                        let width = g_draw_data[data_idx + 4u];

                        distance = sd_oriented_ellipse(in.pos.xy, p0, p1, width);

                        if (fillmode == FILL_HOLLOW) 
                        {
                            distance = abs(distance) - g_draw_data[data_idx + 5u];;
                        }
                    }
                    case PRIMITIVE_AABOX:
                    {
                        let center = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
                        let half_ext = vec2<f32>(g_draw_data[data_idx + 2u], g_draw_data[data_idx + 3u]);
                        distance = sd_aabox(in.pos.xy, center, half_ext, g_draw_data[data_idx + 4u]);
                    }
                    case PRIMITIVE_TRIANGLE: 
                    {
                        let p0 = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
                        let p1 = vec2<f32>(g_draw_data[data_idx + 2u], g_draw_data[data_idx + 3u]);
                        let p2 = vec2<f32>(g_draw_data[data_idx + 4u], g_draw_data[data_idx + 5u]);
                        distance = sd_triangle(in.pos.xy, p0, p1, p2);
                        
                        if (fillmode == FILL_HOLLOW) 
                        {
                            distance = abs(distance);
                        }
                        distance -= g_draw_data[data_idx + 6u];
                    }
                    case PRIMITIVE_BLURRED_BOX: 
                    {
                        let center = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
                        let size   = vec2<f32>(g_draw_data[data_idx + 2u], g_draw_data[data_idx + 3u]);
                        let roundness = g_draw_data[data_idx + 4u];
                        
                        let res = sd_gaussian_box(in.pos.xy, center, size, roundness);
                        distance = res.x;
                        cmd_color.a *= res.y; // apply the gaussian alpha
                    }
                    case PRIMITIVE_CHAR: 
                    {
                        let glyph_idx = extra;
                        let top_left = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
                        let g = g_glyphs[glyph_idx];
                        let t = (in.pos.xy - top_left) / vec2<f32>(g.width, g.height);
                        if (all(t >= vec2<f32>(0.0)) && all(t <= vec2<f32>(1.0))) 
                        {
                            distance = (1.0 - textureSample(g_font, g_font_sampler, mix(g.uv_topleft, g.uv_bottomright, t)).r) * g_draw_args.aa_width;
                        }
                    }
                    case PRIMITIVE_QUAD: 
                    {
                        let top_left     = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
                        let bottom_right = vec2<f32>(g_draw_data[data_idx + 2u], g_draw_data[data_idx + 3u]);
                        let uv_topleft   = vec2<f32>(g_draw_data[data_idx + 4u], g_draw_data[data_idx + 5u]);
                        let uv_bottomright = vec2<f32>(g_draw_data[data_idx + 6u], g_draw_data[data_idx + 7u]);
                        
                        // Calculate normalized local coordinates (0.0 to 1.0)
                        let t = (in.pos.xy - top_left) / (bottom_right - top_left);

                        if (all(t >= vec2<f32>(0.0)) && all(t <= vec2<f32>(1.0))) 
                        {
                            let uv = mix(uv_topleft, uv_bottomright, t);
                            let tex_color = textureSample(g_atlas, g_atlas_sampler, uv, i32(extra));
                            cmd_color = cmd_color * tex_color;
                            distance = 0.0;
                        }
                    }
                    case PRIMITIVE_ORIENTED_QUAD: 
                    {
                        let center       = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
                        let dimensions   = vec2<f32>(g_draw_data[data_idx + 2u], g_draw_data[data_idx + 3u]);
                        let axis         = vec2<f32>(g_draw_data[data_idx + 4u], g_draw_data[data_idx + 5u]);
                        let uv_topleft   = vec2<f32>(g_draw_data[data_idx + 6u], g_draw_data[data_idx + 7u]);
                        let uv_bottomright = vec2<f32>(g_draw_data[data_idx + 8u], g_draw_data[data_idx + 9u]);

                        let relative = in.pos.xy - center;
                        
                        var t = vec2<f32>(dot(axis, relative), dot(skew(axis), relative));
                        
                        t = t * dimensions + 0.5;

                        if (all(t >= vec2<f32>(0.0)) && all(t <= vec2<f32>(1.0)))
                        {
                            let uv = mix(uv_topleft, uv_bottomright, t);
                            let tex_color = textureSample(g_atlas, g_atlas_sampler, uv, i32(extra));
                            cmd_color = cmd_color * tex_color;
                            distance = 0.0;
                        }
                    }
                    case PRIMITIVE_ARC:
                    {
                        let center= vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
                        let radius = g_draw_data[data_idx + 2u];
                        let direction= vec2<f32>(g_draw_data[data_idx + 3u], g_draw_data[data_idx + 4u]);
                        let aperture= vec2<f32>(g_draw_data[data_idx + 5u], g_draw_data[data_idx + 6u]);
                        let thickness = g_draw_data[data_idx + 7u];
                        distance = sd_oriented_arc(in.pos.xy, center, direction, aperture, radius, thickness);
                    }
                    case PRIMITIVE_PIE:
                    {
                        let center= vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
                        let radius = g_draw_data[data_idx + 2u];
                        let direction= vec2<f32>(g_draw_data[data_idx + 3u], g_draw_data[data_idx + 4u]);
                        let aperture= vec2<f32>(g_draw_data[data_idx + 5u], g_draw_data[data_idx + 6u]);
                        distance = sd_oriented_pie(in.pos.xy, center, direction, aperture, radius);

                        if (fillmode == FILL_HOLLOW) 
                        {
                            distance = abs(distance) - g_draw_data[data_idx + 7u];
                        }
                    }
                    // TODO : other cases
                    default: { distance = 1e8; }
                }

                var final_color: vec4<f32>;
                if (cmd_type == END_GROUP)
                {
                    grouping = false;
                    final_color = previous_color;
                    distance = previous_distance;
                    group_op = OP_OVERWRITE;
                }
                else
                {
                    final_color = cmd_color;
                }

                if (grouping)
                {
                    let smooth_factor = select(g_draw_args.aa_width, group_smoothness, group_op == OP_BLEND);
                    let res = smooth_minimum(distance, previous_distance, smooth_factor);
                    previous_distance = res.x;
                    previous_color = mix(final_color, previous_color, res.y);
                }
                else
                {
                    var alpha_factor: f32;
                    if (outline_width > 0.0 && cmd_type == END_GROUP)
                    {
                        if (distance > g_draw_args.aa_width) 
                        {
                            final_color = vec4<f32>(cmd_color.rgb, final_color.a); 
                        } 
                        else 
                        {
                            final_color = vec4<f32>(mix(cmd_color.rgb, final_color.rgb, linearstep(g_draw_args.aa_width, 0.0, distance)), final_color.a);
                        }
                        alpha_factor = linearstep(g_draw_args.aa_width * 2.0 + outline_width, g_draw_args.aa_width + outline_width, distance);
                        outline_width = 0.0;
                    }
                    else
                    {
                        alpha_factor = linearstep(g_draw_args.aa_width, 0.0, distance);
                    }

                    final_color.a *= alpha_factor;
                    output = accumulate_color(final_color, output);
                }
            }
        }
        node_idx = node.next;
    }

    if ((g_draw_args.options & OPTION_SRGB_BACKBUFFER) == 0u)
    {
        output = linear_to_srgb(output);
    }

    return output;
}