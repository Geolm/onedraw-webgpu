// ---------------------------------------------------------------------------------------------------------------------------
// Bindgroups
// ---------------------------------------------------------------------------------------------------------------------------

// group 0 : static or updated on gpu
@group(0) @binding(0)
var<storage, read_write> g_tile_nodes : array<tile_node>;

@group(0) @binding(1)
var<storage, read_write> g_tile_indices : array<u32>;

@group(0) @binding(2)
var<storage, read_write> g_counters : counters;

@group(0) @binding(3)
var<storage, read_write> g_tile_heads : array<u32>;

@group(0) @binding(4)
var<storage, read_write> g_indirect_draw : indirect_params;

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
// Helpers
// ---------------------------------------------------------------------------------------------------------------------------

fn aabb_grow(box: aabb, amount: vec2<f32>) -> aabb 
{
    return aabb(box.min - amount, box.max + amount);
}

fn aabb_get_extents(box: aabb) -> vec2<f32> {return box.max - box.min;}

fn compute_obb(p0: vec2<f32>, p1: vec2<f32>, width: f32) -> obb 
{
    var result: obb;

    result.center = (p0 + p1) * 0.5;
    result.axis_j = p1 - result.center;
    result.extents.y = length(result.axis_j);
    result.axis_j = result.axis_j / result.extents.y;
    result.axis_i = skew(result.axis_j);
    result.extents.x = width * 0.5;

    return result;
}

fn obb_transform(obox: obb, point: vec2<f32>) -> vec2<f32> 
{
    let p = point - obox.center;
    return vec2<f32>(
        abs(dot(obox.axis_i, p)),
        abs(dot(obox.axis_j, p))
    );
}

fn edge_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 
{
    return (b.y - a.y) * (p.x - a.x) - (b.x - a.x) * (p.y - a.y);
}

fn edge_separation(e0: vec2<f32>, e1: vec2<f32>, refp: vec2<f32>, box_corners: array<vec2<f32>,4>) -> bool 
{
    let ref_dist = edge_distance(refp, e0, e1);

    var all_opposite = true;
    if (ref_dist > 0.0) 
    {
        for (var i: u32 = 0; i < 4; i = i + 1) 
        {
            if (edge_distance(box_corners[i], e0, e1) >= 0.0) 
            {
                all_opposite = false;
            }
        }
    } else if (ref_dist < 0.0) 
    {
        for (var i: u32 = 0; i < 4; i = i + 1) 
        {
            if (edge_distance(box_corners[i], e0, e1) <= 0.0) 
            {
                all_opposite = false;
            }
        }
    } else 
    {
        all_opposite = false;
    }

    return all_opposite;
}

// ---------------------------------------------------------------------------------------------------------------------------
// Intersection functions
// ---------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------
fn intersection_aabb_ray(box: aabb, origin: vec2<f32>, direction: vec2<f32>) -> bool 
{
    var tmin: f32 = 0.0;
    var tmax: f32 = 1e10;

    for (var i: u32 = 0; i < 2; i = i + 1) 
    {
        let inv_dir = 1.0 / direction[i];
        var t1 = (box.min[i] - origin[i]) * inv_dir;
        var t2 = (box.max[i] - origin[i]) * inv_dir;

        if (t1 > t2) 
        {
            let temp = t1;
            t1 = t2;
            t2 = temp;
        }

        tmin = max(tmin, t1);
        tmax = min(tmax, t2);

        if (tmin > tmax) 
        {
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------
fn intersection_aabb_disc(box: aabb, center: vec2<f32>, radius: f32) -> bool 
{
    let nearest_point = clamp(center, box.min, box.max);
    return distance_squared(nearest_point, center) < square(radius);
}

// ---------------------------------------------------------------------------------------------------------------------------
fn intersection_aabb_circle(box: aabb, center: vec2<f32>, radius: f32, half_width: f32) -> bool 
{
    if (!intersection_aabb_disc(box, center, radius + half_width))
    {
        return false;
    }

    let candidate0 = abs(center - box.min);
    let candidate1 = abs(center - box.max);
    let furthest_point = max(candidate0, candidate1);

    return length_squared(furthest_point) > square(radius - half_width);
}

// ---------------------------------------------------------------------------------------------------------------------------
fn intersection_aabb_obb(box: aabb, p0: vec2<f32>, p1: vec2<f32>, width: f32) -> bool 
{
    let dir = p1 - p0;
    let center = (p0 + p1) * 0.5;
    let height = length(dir);

    let axis_j = dir / height;
    let axis_i = vec2<f32>(-axis_j.y, axis_j.x);

    let half_i = width * 0.5;
    let half_j = height * 0.5;

    let aabb_extent = abs(axis_i * half_i) + abs(axis_j * half_j);
    let obb_min = center - aabb_extent;
    let obb_max = center + aabb_extent;

    if (all(obb_max < box.min) || all(box.max < obb_min)) 
    {
        return false;
    }

    let aabb_center = (box.min + box.max) * 0.5;
    let aabb_half = (box.max - box.min) * 0.5;

    var d = abs(dot(axis_i, center - aabb_center));
    var r = aabb_half.x * abs(axis_i.x) + aabb_half.y * abs(axis_i.y);
    if (d > (half_i + r)) 
    {
        return false;
    }

    d = abs(dot(axis_j, center - aabb_center));
    r = aabb_half.x * abs(axis_j.x) + aabb_half.y * abs(axis_j.y);
    if (d > (half_j + r)) 
    {
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------
fn intersection_aabb_triangle(box: aabb, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> bool 
{
    let pmin = min(min(p0, p1), p2);
    let pmax = max(max(p0, p1), p2);
    if (any(pmax < box.min) || any(box.max < pmin)) 
    {
        return false;
    }

    // Box corners
    let v0 = box.min;
    let v1 = box.max;
    let v2 = vec2<f32>(box.min.x, box.max.y);
    let v3 = vec2<f32>(box.max.x, box.min.y);
    let box_corners: array<vec2<f32>,4> = array<vec2<f32>,4>(v0, v1, v2, v3);

    // Edge separation tests
    if (edge_separation(p0, p1, p2, box_corners)) { return false; }
    if (edge_separation(p1, p2, p0, box_corners)) { return false; }
    if (edge_separation(p2, p0, p1, box_corners)) { return false; }

    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------
fn intersection_aabb_pie(box: aabb, center: vec2<f32>, direction: vec2<f32>, aperture: vec2<f32>, radius: f32) -> bool 
{
    if (!intersection_aabb_disc(box, center, radius)) 
    {
        return false;
    }

    let aabb_vertices: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
        box.min,
        box.max,
        vec2<f32>(box.min.x, box.max.y),
        vec2<f32>(box.max.x, box.min.y)
    );

    for (var i: u32 = 0; i < 4; i = i + 1) 
    {
        let center_vertex = normalize(aabb_vertices[i] - center);
        if (dot(center_vertex, direction) > aperture.y) 
        {
            return true;
        }
    }

    return intersection_aabb_ray(box, center, direction);
}

// ---------------------------------------------------------------------------------------------------------------------------
fn intersection_aabb_arc(box: aabb, center: vec2<f32>, direction: vec2<f32>, aperture: vec2<f32>, radius: f32, thickness: f32) -> bool 
{
    let half_thickness = thickness * 0.5;
    if (!intersection_aabb_circle(box, center, radius, half_thickness)) 
    {
        return false;
    }
    return intersection_aabb_pie(box, center, direction, aperture, radius + half_thickness);
}

// ---------------------------------------------------------------------------------------------------------------------------
fn point_in_triangle(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, point: vec2<f32>) -> bool 
{
    let d1 = edge_distance(point, p0, p1);
    let d2 = edge_distance(point, p1, p2);
    let d3 = edge_distance(point, p2, p0);

    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

    return !(has_neg && has_pos);
}

// ---------------------------------------------------------------------------------------------------------------------------
fn point_in_pie(center: vec2<f32>, direction: vec2<f32>, radius: f32, cos_aperture: f32, point: vec2<f32>) -> bool 
{
    if (distance_squared(center, point) > square(radius)) 
    {
        return false;
    }
    let to_point = normalize(point - center);
    return dot(to_point, direction) > cos_aperture;
}

// ---------------------------------------------------------------------------------------------------------------------------
fn intersection_ellipse_circle(p0: vec2<f32>, p1: vec2<f32>, width: f32, center: vec2<f32>, radius: f32) -> bool 
{
    let obox = compute_obb(p0, p1, width);
    let transformed_center = obb_transform(obox, center) / obox.extents;
    let scaled_radius = radius / min(obox.extents.x, obox.extents.y);
    let squared_distance = dot(transformed_center, transformed_center);
    return squared_distance <= square(1.0 + scaled_radius);
}

// ---------------------------------------------------------------------------------------------------------------------------
fn is_aabb_inside_ellipse(p0: vec2<f32>, p1: vec2<f32>, width: f32, box: aabb) -> bool 
{
    let aabb_vertices: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
        box.min,
        box.max,
        vec2<f32>(box.min.x, box.max.y),
        vec2<f32>(box.max.x, box.min.y)
    );

    let obox = compute_obb(p0, p1, width);

    for (var i: u32 = 0; i < 4; i = i + 1) 
    {
        let vertex_ellipse_space = obb_transform(obox, aabb_vertices[i]);
        let distance = square(vertex_ellipse_space.x) / square(obox.extents.x) +
                            square(vertex_ellipse_space.y) / square(obox.extents.y);
        if (distance > 1.0) 
        {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------
fn is_aabb_inside_triangle(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, box: aabb) -> bool 
{
    let aabb_vertices: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
        box.min,
        box.max,
        vec2<f32>(box.min.x, box.max.y),
        vec2<f32>(box.max.x, box.min.y)
    );

    for (var i: u32 = 0; i < 4; i = i + 1) 
    {
        let d0 = edge_distance(p0, p1, aabb_vertices[i]);
        let d1 = edge_distance(p1, p2, aabb_vertices[i]);
        let d2 = edge_distance(p2, p0, aabb_vertices[i]);

        let has_neg = (d0 < 0.0) || (d1 < 0.0) || (d2 < 0.0);
        let has_pos = (d0 > 0.0) || (d1 > 0.0) || (d2 > 0.0);

        if (has_neg && has_pos) 
        {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------
fn is_aabb_inside_obb(p0: vec2<f32>, p1: vec2<f32>, width: f32, box: aabb) -> bool 
{
    let aabb_vertices: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
        box.min,
        box.max,
        vec2<f32>(box.min.x, box.max.y),
        vec2<f32>(box.max.x, box.min.y)
    );

    let obox = compute_obb(p0, p1, width);

    for (var i: u32 = 0; i < 4; i = i + 1) 
    {
        let point = obb_transform(obox, aabb_vertices[i]);
        if (any(abs(point) > obox.extents)) 
        {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------
fn is_aabb_inside_pie(center: vec2<f32>, direction: vec2<f32>, aperture: vec2<f32>, radius: f32, box: aabb) -> bool 
{
    let aabb_vertices: array<vec2<f32>, 4> = array<vec2<f32>, 4>
    (
        box.min,
        box.max,
        vec2<f32>(box.min.x, box.max.y),
        vec2<f32>(box.max.x, box.min.y)
    );

    for (var i: u32 = 0; i < 4; i = i + 1) 
    {
        if (!point_in_pie(center, direction, radius, aperture.y, aabb_vertices[i])) 
        {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------
// Tile Binning Helpers
// ---------------------------------------------------------------------------------------------------------------------------

fn clip_tile(tile: aabb, clip: clip_rect) -> bool
{
    return (tile.max.x < clip.min_x || tile.max.y < clip.min_y || 
            tile.min.x > clip.max_x || tile.min.y > clip.max_y);
}

fn intersection_tile_command(tile_aabb: aabb, cmd: draw_command, aabb_margin: f32) -> bool
{
    let tile_enlarge_aabb = aabb_grow(tile_aabb, vec2<f32>(aabb_margin));
    let is_hollow = (get_fillmode(cmd) != 0u); 
    let cmd_type = get_type(cmd);
    let data_idx = cmd.data_index;
    var intersection = false;

    switch (cmd_type)
    {
        case PRIMITIVE_ORIENTED_BOX:
        {
            let p0 = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
            let p1 = vec2<f32>(g_draw_data[data_idx + 2u], g_draw_data[data_idx + 3u]);
            let width = g_draw_data[data_idx + 4u];
            let tile_rounded = aabb_grow(tile_enlarge_aabb, vec2<f32>(g_draw_data[data_idx + 5u]));
            intersection = intersection_aabb_obb(tile_rounded, p0, p1, width);

            if (intersection && is_hollow && is_aabb_inside_obb(p0, p1, width, tile_rounded))
            {
                intersection = false;
            }
        }
        case PRIMITIVE_ELLIPSE:
        {
            let p0 = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
            let p1 = vec2<f32>(g_draw_data[data_idx + 2u], g_draw_data[data_idx + 3u]);
            let width = g_draw_data[data_idx + 4u];
            let tile_center = (tile_aabb.min + tile_aabb.max) * 0.5;
            let margin = select(0.0, g_draw_data[data_idx + 5u], is_hollow);
            let tile_smooth = aabb_grow(tile_enlarge_aabb, vec2<f32>(margin));
            intersection = intersection_ellipse_circle(p0, p1, width, tile_center, length(aabb_get_extents(tile_smooth) * 0.5));

            if (intersection && is_hollow && is_aabb_inside_ellipse(p0, p1, width, tile_smooth))
            {
                intersection = false;
            }
        }
        case PRIMITIVE_ARC:
        {
            let center = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
            let radius = g_draw_data[data_idx + 2u];
            let direction = vec2<f32>(g_draw_data[data_idx + 3u], g_draw_data[data_idx + 4u]);
            let aperture = vec2<f32>(g_draw_data[data_idx + 5u], g_draw_data[data_idx + 6u]);
            let thickness = g_draw_data[data_idx + 7u];
            intersection = intersection_aabb_arc(tile_enlarge_aabb, center, direction, aperture, radius, thickness);
        }
        case PRIMITIVE_PIE:
        {
            let center = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
            let radius = g_draw_data[data_idx + 2u];
            let direction = vec2<f32>(g_draw_data[data_idx + 3u], g_draw_data[data_idx + 4u]);
            let aperture = vec2<f32>(g_draw_data[data_idx + 5u], g_draw_data[data_idx + 6u]);
            let margin = select(0.0, g_draw_data[data_idx + 7u], is_hollow);
            let tile_smooth = aabb_grow(tile_enlarge_aabb, vec2<f32>(margin));
            intersection = intersection_aabb_pie(tile_smooth, center, direction, aperture, radius);

            if (intersection && is_hollow && is_aabb_inside_pie(center, direction, aperture, radius, tile_smooth))
            {
                intersection = false;
            }
        }
        case PRIMITIVE_DISC:
        {
            let center = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
            var radius = g_draw_data[data_idx + 2u];
            if (is_hollow)
            {
                let half_width = g_draw_data[data_idx + 3u] + aabb_margin;
                intersection = intersection_aabb_circle(tile_aabb, center, radius, half_width);
            }
            else
            {
                radius += aabb_margin;
                intersection = intersection_aabb_disc(tile_aabb, center, radius);
            }
        }
        case PRIMITIVE_TRIANGLE:
        {
            let p0 = vec2<f32>(g_draw_data[data_idx + 0u], g_draw_data[data_idx + 1u]);
            let p1 = vec2<f32>(g_draw_data[data_idx + 2u], g_draw_data[data_idx + 3u]);
            let p2 = vec2<f32>(g_draw_data[data_idx + 4u], g_draw_data[data_idx + 5u]);
            let tile_rounded = aabb_grow(tile_enlarge_aabb, vec2<f32>(g_draw_data[data_idx + 6u]));
            intersection = intersection_aabb_triangle(tile_rounded, p0, p1, p2);

            if (intersection && is_hollow && is_aabb_inside_triangle(p0, p1, p2, tile_rounded))
            {
                intersection = false;
            }
        }
        default:
        {
            intersection = true;
        }
    }
    return intersection;
}

// ---------------------------------------------------------------------------------------------------------------------------
// Compute Shaders
// ---------------------------------------------------------------------------------------------------------------------------

@compute @workgroup_size(64)
fn clear_heads(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let i = global_id.x;
    let num_tiles = g_draw_args.num_tile_width * g_draw_args.num_tile_height;

    if (i < num_tiles)
    {
        g_tile_heads[i] = 0xffffffffu;
    }
}

@compute @workgroup_size(16, 16, 1)
fn tile_bin(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let tile_xy = global_id.xy;

    if (tile_xy.x >= g_draw_args.num_tile_width || tile_xy.y >= g_draw_args.num_tile_height)
    {
        return;
    }

    let tile_index:u32 = tile_xy.y * g_draw_args.num_tile_width + tile_xy.x;
    
    var current_head = INVALID_INDEX; 

    let tile_min = vec2<f32>(tile_xy) * TILE_SIZE;
    let tile_max = (vec2<f32>(tile_xy) + 1.0) * TILE_SIZE;
    let tile_aabb = aabb(tile_min, tile_max);

    var aabb_margin: f32 = 0.0;

    // REVERSE TRAVERSAL: Start from the last command and go to the first.
    // This ensures that command 0 becomes the final 'head' of the linked list.
    for (var i: i32 = i32(g_draw_args.num_commands) - 1; i >= 0; i--)
    {
        let cmd_idx = u32(i);
        let cmd = g_commands[cmd_idx];
        let cmd_type = get_type(cmd);

        // 1. Coarse AABB check (Quantized)
        let q_aabb = g_quantized_aabb[cmd_idx];
        let min_x = (q_aabb >> 0u) & 0xFFu;
        let min_y = (q_aabb >> 8u) & 0xFFu;
        let max_x = (q_aabb >> 16u) & 0xFFu;
        let max_y = (q_aabb >> 24u) & 0xFFu;

        if (tile_xy.x < min_x || tile_xy.y < min_y || tile_xy.x > max_x || tile_xy.y > max_y)
        {
            continue;
        }

        // 2. Clip Rect check
        let clip_idx = get_clip(cmd);
        let clip = g_clips[clip_idx];
        if (clip_tile(tile_aabb, clip))
        {
            continue;
        }

        // 3. Fine-grained intersection test
        let to_be_added = intersection_tile_command(tile_aabb, cmd, g_draw_args.aa_width + aabb_margin);

        // we traverse in reverse order, so the end comes first
        if (cmd_type == BEGIN_GROUP) 
        {
            aabb_margin = 0.0;
        }
        else if (cmd_type == END_GROUP)
        {
            aabb_margin = g_draw_data[cmd.data_index];
        }

        if (to_be_added)
        {
            let node_idx = atomicAdd(&g_counters.num_nodes, 1u);

            if (node_idx < g_draw_args.max_nodes)
            {
                let packed_info = (cmd_type << 16u) | (cmd_idx & 0xFFFFu);

                g_tile_nodes[node_idx] = tile_node(current_head, packed_info);

                current_head = node_idx;
            }
        }
    }

    // Write final head to global buffer
    g_tile_heads[tile_index] = current_head;

    // If tile is not empty, register it for processing
    if (current_head != INVALID_INDEX)
    {
        let pos = atomicAdd(&g_counters.num_tiles, 1u);
        g_tile_indices[pos] = tile_index;
    }
}

@compute @workgroup_size(1, 1, 1)
fn write_indirect_args()
{
    g_indirect_draw.vertex_count   = 4u;
    g_indirect_draw.instance_count = atomicLoad(&g_counters.num_tiles);
    g_indirect_draw.first_vertex   = 0u;
    g_indirect_draw.first_instance = 0u;
}
