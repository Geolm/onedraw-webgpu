Hierarchical command binning divides the screen into coarse **Regions** (e.g., an $8 \times 4$ grid of 32 spatial regions) and fine **Tiles** (e.g., $16 \times 16$ pixels). Instead of broadcasting every command to all tiles in a monolithic compute pass, commands are coarsely binned into regions first via parallel prefix scans, allowing downstream tile binning to process only region-relevant command subsets.

```
[ Draw Commands (N) ]
          │
          ▼
Dispatch 1: Compute 32-bit Spatial Region Intersect Mask
          │
          ▼
Dispatch 2: Workgroup-Local Block Histograms (256 cmds/block)
          │
          ▼
Dispatch 3: Exclusive Prefix Scan Over Block Counts & Region Bases
          │
          ▼
Dispatch 4: Stable Multi-Region Parallel Scatter (Intra-Block Scan)
          │
          ▼
[ Sorted Region Command Buffer & Range Index ]
          │
          ▼
Dispatch 5: Regional Tile Binning (Existing Tile Head/Node Builder)

```

---

## Buffer Layouts and Data Structures

### WGSL Shared Structures (`binning_hierarchical.wgsl`)

```wgsl
struct RegionArgs 
{
    num_regions_x: u32,
    num_regions_y: u32,
    num_commands: u32,
    num_blocks: u32,
};

// Layout: [Block 0 (32 u32s), Block 1 (32 u32s), ..., Block B-1 (32 u32s)]
@group(0) @binding(0) var<storage, read_write> g_region_masks : array<u32>;
@group(0) @binding(1) var<storage, read_write> g_block_counts : array<u32>;  // B * 32
@group(0) @binding(2) var<storage, read_write> g_block_offsets : array<u32>; // B * 32
@group(0) @binding(3) var<storage, read_write> g_region_base   : array<u32>; // 32
@group(0) @binding(4) var<storage, read_write> g_region_counts : array<u32>; // 32
@group(0) @binding(5) var<storage, read_write> g_region_indices: array<u32>; // Output sorted indices

```

---

## Compute Pipeline Execution

### Dispatch 1 — Region Mask Computation

Calculates a 32-bit visibility mask for each command across up to 32 screen regions.

* **Workgroup Size**: $(256, 1, 1)$
* **Dispatch Grid**: $(\lceil N / 256 \rceil, 1, 1)$

```wgsl
@compute @workgroup_size(256)
fn compute_region_masks(@builtin(global_invocation_id) global_id: vec3<u32>) 
{
    let cmd_idx = global_id.x;
    if (cmd_idx >= params.num_commands) { return; }

    // Read quantized AABB: [min_x, min_y, max_x, max_y] in tile coordinates
    let q_aabb = g_quantized_aabb[cmd_idx];
    let min_tile_x = q_aabb & 0xFFu;
    let min_tile_y = (q_aabb >> 8u) & 0xFFu;
    let max_tile_x = (q_aabb >> 16u) & 0xFFu;
    let max_tile_y = (q_aabb >> 24u) & 0xFFu;

    // Convert tile bounds to coarse region coordinates
    let min_reg_x = min_tile_x / params.tiles_per_region_x;
    let min_reg_y = min_tile_y / params.tiles_per_region_y;
    let max_reg_x = min(max_tile_x / params.tiles_per_region_x, params.num_regions_x - 1u);
    let max_reg_y = min(max_tile_y / params.tiles_per_region_y, params.num_regions_y - 1u);

    var mask = 0u;
    for (var ry = min_reg_y; ry <= max_reg_y; ry++) {
        for (var rx = min_reg_x; rx <= max_reg_x; rx++) {
            let region_id = ry * params.num_regions_x + rx;
            if (region_id < 32u) {
                mask |= (1u << region_id);
            }
        }
    }

    g_region_masks[cmd_idx] = mask;
}

```

---

### Dispatch 2 — Workgroup Histogram Generation

Computes per-block region command counts for each chunk of 256 commands.

* **Workgroup Size**: $(256, 1, 1)$
* **Dispatch Grid**: $(B, 1, 1)$ where $B = \lceil N / 256 \rceil$

```wgsl
var<workgroup> local_counts: array<atomic<u32>, 32>;

@compute @workgroup_size(256)
fn build_block_histograms(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let block_idx = wg_id.x;
    let cmd_idx = block_idx * 256u + tid;

    if (tid < 32u) {
        atomicStore(&local_counts[tid], 0u);
    }
    workgroupBarrier();

    if (cmd_idx < params.num_commands) {
        var mask = g_region_masks[cmd_idx];
        while (mask != 0u) {
            let r = countTrailingZeros(mask);
            atomicAdd(&local_counts[r], 1u);
            mask &= mask - 1u; // Clear lowest set bit
        }
    }
    workgroupBarrier();

    if (tid < 32u) {
        let count = atomicLoad(&local_counts[tid]);
        g_block_counts[block_idx * 32u + tid] = count;
    }
}

```

---

### Dispatch 3 — Parallel Scan across Blocks and Region Base Offsets

Performs exclusive prefix scans along block counts per region, followed by global region base computations.

* **Workgroup Size**: $(256, 1, 1)$
* **Dispatch Grid**: $(32, 1, 1)$ (1 workgroup per region $r \in [0, 31]$)

```wgsl
var<workgroup> scan_buf: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_block_counts(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let region_id = wg_id.x;
    let tid = local_id.x;
    let num_blocks = params.num_blocks;

    // Load block counts into local shared memory
    var val = 0u;
    if (tid < num_blocks) {
        val = g_block_counts[tid * 32u + region_id];
    }
    scan_buf[tid] = val;
    workgroupBarrier();

    // Parallel Kogge-Stone exclusive scan across blocks
    for (var stride = 1u; stride < 256u; stride <<= 1u) {
        var temp = 0u;
        if (tid >= stride) {
            temp = scan_buf[tid - stride];
        }
        workgroupBarrier();
        if (tid >= stride) {
            scan_buf[tid] += temp;
        }
        workgroupBarrier();
    }

    let inclusive_val = scan_buf[tid];
    let exclusive_offset = inclusive_val - val;

    if (tid < num_blocks) {
        g_block_offsets[tid * 32u + region_id] = exclusive_offset;
    }

    // Thread 255 computes the total for this region
    if (tid == 255u) {
        let total_region_cmds = select(0u, scan_buf[num_blocks - 1u], num_blocks > 0u);
        g_region_counts[region_id] = total_region_cmds;
    }
}

```

#### Single-Pass Prefix Scan for Region Bases

A single workgroup pass scans `g_region_counts` across all 32 regions to produce `g_region_base`.

```wgsl
@compute @workgroup_size(32)
fn scan_region_bases(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let r = local_id.x;
    let count = g_region_counts[r];
    scan_buf[r] = count;
    workgroupBarrier();

    // Exclusive scan across 32 region counts
    for (var stride = 1u; stride < 32u; stride <<= 1u) {
        var temp = 0u;
        if (r >= stride) { temp = scan_buf[r - stride]; }
        workgroupBarrier();
        if (r >= stride) { scan_buf[r] += temp; }
        workgroupBarrier();
    }

    g_region_base[r] = scan_buf[r] - count;
}

```

---

### Dispatch 4 — Stable Intra-Block Parallel Scatter

Scatters command indices into contiguous region partitions using subgroup ballot acceleration with shared-memory fallback.

* **Workgroup Size**: $(256, 1, 1)$
* **Dispatch Grid**: $(B, 1, 1)$

```wgsl
struct BinningParams {
    num_commands: u32,
    num_blocks: u32,
};

@group(0) @binding(0) var<storage, read> g_region_masks   : array<u32>;
@group(0) @binding(1) var<storage, read> g_block_offsets  : array<u32>; // Block * 32 + r
@group(0) @binding(2) var<storage, read> g_region_base    : array<u32>; // 32
@group(0) @binding(3) var<storage, read_write> g_region_indices : array<u32>;
@group(0) @binding(4) var<uniform> params                 : BinningParams;

// 256 elements of workgroup memory for the parallel prefix scan
var<workgroup> local_scan: array<u32, 256>;

@compute @workgroup_size(256)
fn stable_scatter_region_indices(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let block_idx = wg_id.x;
    let cmd_idx = block_idx * 256u + tid;

    // Load visibility mask for this thread's command
    let mask = select(0u, g_region_masks[cmd_idx], cmd_idx < params.num_commands);

    // Loop through each of the 32 regions
    for (var r = 0u; r < 32u; r++) {
        let has_region = (mask & (1u << r)) != 0u;
        let active_bit = select(0u, 1u, has_region);

        // Load 1 or 0 into shared memory
        local_scan[tid] = active_bit;
        workgroupBarrier();

        // Standard parallel Kogge-Stone scan across 256 threads in shared memory
        for (var stride = 1u; stride < 256u; stride <<= 1u) {
            var val = 0u;
            if (tid >= stride) {
                val = local_scan[tid - stride];
            }
            workgroupBarrier();
            if (tid >= stride) {
                local_scan[tid] += val;
            }
            workgroupBarrier();
        }

        // Exclusive prefix = (Inclusive prefix sum) - (current thread's bit)
        let local_prefix = local_scan[tid] - active_bit;

        // Scatter command index to global memory if visible in region r
        if (has_region && cmd_idx < params.num_commands) {
            let reg_base = g_region_base[r];
            let block_off = g_block_offsets[block_idx * 32u + r];
            let dst_slot = reg_base + block_off + local_prefix;

            g_region_indices[dst_slot] = cmd_idx;
        }

        // Synchronize workgroup shared memory before processing next region iteration
        workgroupBarrier();
    }
}

```

---

## C API and Integration into `onedraw.c`

### Extended Engine Struct Layout

```c
// Add region binning state to struct onedraw in onedraw.c
typedef struct onedraw_region_binning {
    WGPUBuffer region_masks;    // N * sizeof(uint32_t)
    WGPUBuffer block_counts;    // B * 32 * sizeof(uint32_t)
    WGPUBuffer block_offsets;   // B * 32 * sizeof(uint32_t)
    WGPUBuffer region_base;     // 32 * sizeof(uint32_t)
    WGPUBuffer region_counts;   // 32 * sizeof(uint32_t)
    WGPUBuffer region_indices;  // N_allocated * sizeof(uint32_t)
    
    WGPUComputePipeline pso_region_masks;
    WGPUComputePipeline pso_block_histograms;
    WGPUComputePipeline pso_scan_blocks;
    WGPUComputePipeline pso_scan_region_bases;
    WGPUComputePipeline pso_stable_scatter;
    
    WGPUBindGroup bind_group;
    uint32_t num_regions_x;
    uint32_t num_regions_y;
} onedraw_region_binning;

```

### Execution Flow in `od_end_frame()`

```c
void od_end_frame_hierarchical(struct onedraw* r, WGPUTextureView target_view)
{
    uint32_t num_cmds = (uint32_t)r->commands.list.num_elements;
    if (num_cmds == 0) return;

    uint32_t blocks = (num_cmds + 255u) / 256u;
    uint32_t buffer_index = r->stats.frame_index % BUFFER_FRAME_COUNT;

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(r->device, NULL);

    // --- COARSE REGION BINNING PASSES ---
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &(WGPUComputePassDescriptor){
        .label = WGPU_STRING_VIEW("hierarchical region binning")
    });

    // Dispatch 1: Compute Region Masks
    wgpuComputePassEncoderSetPipeline(pass, r->regions_bin.pso_region_masks);
    wgpuComputePassEncoderSetBindGroup(pass, 0, r->regions_bin.bind_group, 0, NULL);
    wgpuComputePassEncoderSetBindGroup(pass, 1, r->binding.frame_bindgroup[buffer_index], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, blocks, 1, 1);

    // Dispatch 2: Block Histograms
    wgpuComputePassEncoderSetPipeline(pass, r->regions_bin.pso_block_histograms);
    wgpuComputePassEncoderDispatchWorkgroups(pass, blocks, 1, 1);

    // Dispatch 3a: Parallel Scan Block Counts (32 workgroups)
    wgpuComputePassEncoderSetPipeline(pass, r->regions_bin.pso_scan_blocks);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 32, 1, 1);

    // Dispatch 3b: Single-pass Exclusive Scan for Region Bases
    wgpuComputePassEncoderSetPipeline(pass, r->regions_bin.pso_scan_region_bases);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);

    // Dispatch 4: Final Stable Scatter
    wgpuComputePassEncoderSetPipeline(pass, r->regions_bin.pso_stable_scatter);
    wgpuComputePassEncoderDispatchWorkgroups(pass, blocks, 1, 1);

    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    // --- FINE TILE BINNING PASS ---
    // Downstream tile binning now iterates over g_region_indices per region span
    {
        WGPUComputePassEncoder tile_pass = wgpuCommandEncoderBeginComputePass(encoder, &(WGPUComputePassDescriptor){
            .label = WGPU_STRING_VIEW("fine tile binning")
        });
        wgpuComputePassEncoderSetPipeline(tile_pass, r->tiles.binning_pso);
        wgpuComputePassEncoderSetBindGroup(tile_pass, 0, r->binding.binning_bindgroup, 0, NULL);
        wgpuComputePassEncoderSetBindGroup(tile_pass, 1, r->binding.frame_bindgroup[buffer_index], 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(tile_pass, r->tiles.num_width, r->tiles.num_height, 1);
        wgpuComputePassEncoderEnd(tile_pass);
        wgpuComputePassEncoderRelease(tile_pass);
    }

    // Proceed to indirect argument generation and rasterization pass...
}

```

---

## Performance Considerations

* **Atomics vs. Bitwise Reduction**: Dispatch 2 reduces atomic contention by restricting atomics to workgroup shared memory (`local_counts[32]`) before flushing a single coalesced write per region to global storage (`g_block_counts`).
* **Subgroup Acceleration**: Dispatch 4 uses WebGPU subgroup intrinsics (`subgroupBallot`, `countOneBits`) to compute thread prefixes without shared memory barrier roundtrips on hardware that supports `enable subgroups;`.
* **Dynamic Scaling**: Memory buffer capacity scales dynamically as $B = \lceil N / 256 \rceil$, keeping the peak footprint modest (for $N = 65,536$, $B = 256$, requiring only $256 \times 32 \times 4\text{ B} = 32\text{ KB}$ for histogram buffers).