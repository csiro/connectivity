use crate::routing::Tree;
use anyhow::{anyhow, bail, ensure, Context, Result};
use cubecl::{bytes::Bytes, client::Client, prelude::*, Device};
use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Numeric adjacency shared by both searches; all edge costs are rounded on the CPU.
pub(crate) struct CsrGraph<'a> {
    pub offsets: &'a [usize],
    pub edges: &'a [(usize, u32, u32)],
}

pub(crate) struct GpuRouter {
    client: Client,
    backend: &'static str,
    threads: u32,
}

impl GpuRouter {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn metal() -> Result<Self> {
        Self::initialize("Metal", 4, crate::metal::device)
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    pub(crate) fn cuda() -> Result<Self> {
        // Device zero is relative to CUDA_VISIBLE_DEVICES, including Slurm allocations.
        // One full NVIDIA warp avoids carrying Metal's four-thread launch to CUDA.
        let router = Self::initialize("CUDA", 32, || {
            Device::cuda(0).context("CUDA GPU is unavailable; use device=\"cpu\"")
        })?;
        // A context alone does not check NVRTC or CUDA headers. Compile and run the
        // actual routing kernel once before validate_device accepts this backend.
        router
            .search_batch(&[CsrGraph {
                offsets: &[0, 0],
                edges: &[],
            }])
            .map_err(|error| {
                anyhow!(
                    "CUDA kernel initialization failed: {error:#}; check that CUDA_PATH points \
                     to your CUDA Toolkit, NVRTC libraries are available, and the NVIDIA driver \
                     is compatible; use device=\"cpu\""
                )
            })?;
        Ok(router)
    }

    fn initialize(
        backend: &'static str,
        threads: u32,
        device: impl FnOnce() -> Result<Device>,
    ) -> Result<Self> {
        catch_unwind(AssertUnwindSafe(|| {
            Ok(Self {
                client: device()?.client(),
                backend,
                threads,
            })
        }))
        .map_err(|panic| {
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown device error");
            anyhow!("{backend} GPU initialization failed: {message}; use device=\"cpu\"")
        })?
    }

    /// Two work items per graph run adjusted and intact Dijkstra independently.
    /// Each search preserves CPU heap ordering; batches parallelize across focal cells.
    pub(crate) fn search_batch(&self, graphs: &[CsrGraph<'_>]) -> Result<Vec<(Tree, Tree)>> {
        let backend = self.backend;
        catch_unwind(AssertUnwindSafe(|| {
            if graphs.is_empty() {
                return Ok(Vec::new());
            }

            let mut sizes = [0usize; 4];
            for graph in graphs {
                ensure!(graph.offsets.len() >= 2, "{backend} graph has no source node");
                ensure!(
                    graph.offsets[0] == 0
                        && graph.offsets.last() == Some(&graph.edges.len())
                        && graph.offsets.windows(2).all(|w| w[0] <= w[1]),
                    "{backend} graph has invalid adjacency offsets"
                );
                let nodes = graph.offsets.len() - 1;
                ensure!(
                    graph.edges.iter().all(|e| e.0 < nodes),
                    "{backend} graph has an invalid destination node"
                );
                let lengths = [
                    graph.offsets.len(),
                    graph.edges.len().checked_mul(3)
                        .with_context(|| format!("{backend} graph is too large"))?,
                    nodes.checked_mul(8)
                        .and_then(|n| n.checked_add(2))
                        .with_context(|| format!("{backend} graph is too large"))?,
                    graph.edges.len().checked_add(1)
                        .and_then(|n| n.checked_mul(4))
                        .with_context(|| format!("{backend} graph is too large"))?,
                ];
                for (size, length) in sizes.iter_mut().zip(lengths) {
                    *size = size.checked_add(length).with_context(|| format!("{backend} batch is too large"))?;
                    ensure!(
                        *size <= u32::MAX as usize,
                        "{backend} batch exceeds 32-bit indexing"
                    );
                }
            }
            ensure!(
                graphs.len() <= u32::MAX as usize / 6,
                "{backend} batch is too large"
            );
            let input_len = (graphs.len() * 6)
                .checked_add(sizes[0])
                .and_then(|n| n.checked_add(sizes[1]))
                .with_context(|| format!("{backend} batch is too large"))?;
            ensure!(input_len <= u32::MAX as usize, "{backend} batch exceeds 32-bit indexing");
            for length in [input_len, sizes[2], sizes[3], graphs.len() * 2] {
                ensure!(
                    length as u64 * size_of::<u32>() as u64
                        <= self.client.properties().memory.max_page_size,
                    "{backend} batch exceeds the device buffer limit"
                );
            }

            // Metadata holds node count, offset/edge starts, output/heap starts and heap capacity.
            // Each search owns four node arrays (parent, edge, cost, order) plus order length.
            // One packed input avoids separately synchronized adjacency uploads.
            let mut input = Vec::with_capacity(input_len);
            input.resize(graphs.len() * 6, 0u32);
            let mut state_start = 0usize;
            let mut heap_start = 0usize;
            for (i, graph) in graphs.iter().enumerate() {
                let nodes = graph.offsets.len() - 1;
                let offset_start = input.len();
                input[i * 6..i * 6 + 6].copy_from_slice(&[
                    nodes as u32,
                    offset_start as u32,
                    (offset_start + graph.offsets.len()) as u32,
                    state_start as u32,
                    heap_start as u32,
                    (graph.edges.len() + 1) as u32,
                ]);
                input.extend(graph.offsets.iter().map(|&v| v as u32));
                for &(next, adjusted, intact) in graph.edges {
                    input.extend_from_slice(&[next as u32, adjusted, intact]);
                }
                state_start += nodes * 8 + 2;
                heap_start += (graph.edges.len() + 1) * 4;
            }
            let input_handle = self.client.create(Bytes::from_elems(input));
            let state_handle = self.client.empty(sizes[2] * size_of::<u32>());
            let heap_handle = self.client.empty(sizes[3] * size_of::<u32>());
            let status_handle = self.client.empty(graphs.len() * 2 * size_of::<u32>());
            let searches = graphs.len() as u32 * 2;
            // Each backend selects its launch size; searches own disjoint scratch slices.
            let threads = self.threads;

            // Input validation above and disjoint per-search slices cover every unchecked access.
            unsafe {
                shortest_paths::launch_unchecked(
                    &self.client,
                    CubeCount::Static(searches.div_ceil(threads), 1, 1),
                    CubeDim::new_1d(threads),
                    BufferArg::from_raw_parts(input_handle, input_len),
                    BufferArg::from_raw_parts(state_handle.clone(), sizes[2]),
                    BufferArg::from_raw_parts(heap_handle, sizes[3]),
                    BufferArg::from_raw_parts(status_handle.clone(), graphs.len() * 2),
                );
            }

            // Read both outputs together so the server only waits for the batch once.
            let results = self.client.read(vec![status_handle, state_handle])
                .with_context(|| format!("{backend} result read failed"))?;
            for (search, &status) in u32::from_bytes(&results[0]).iter().enumerate() {
                match status {
                    0 => {}
                    1 => bail!(
                        "Path cost overflow in {backend} graph {}; check max_cost and resistance scaling!",
                        search / 2
                    ),
                    _ => bail!(
                        "{backend} shortest-path scratch capacity exceeded for graph {}",
                        search / 2
                    ),
                }
            }
            let state = u32::from_bytes(&results[1]);
            let mut trees = Vec::with_capacity(graphs.len());
            let mut state_start = 0;
            for graph in graphs {
                let nodes = graph.offsets.len() - 1;
                let read_tree = |weight| -> Result<Tree> {
                    let start = state_start + weight * (4 * nodes + 1);
                    let values = &state[start..start + 4 * nodes + 1];
                    let count = values[4 * nodes] as usize;
                    ensure!(count <= nodes, "{backend} returned an invalid settlement count");
                    Ok(Tree {
                        parent: values[..nodes].iter().map(|&v| {
                            if v == u32::MAX { usize::MAX } else { v as usize }
                        }).collect(),
                        edge: values[nodes..2 * nodes].iter()
                            .map(|&v| v as usize).collect(),
                        cost: values[2 * nodes..3 * nodes].to_vec(),
                        order: values[3 * nodes..3 * nodes + count].iter()
                            .map(|&v| v as usize).collect(),
                    })
                };
                trees.push((read_tree(0)?, read_tree(1)?));
                state_start += nodes * 8 + 2;
            }
            Ok(trees)
        }))
        .map_err(|panic| {
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown runtime error");
            anyhow!("{backend} GPU execution failed: {message}; use device=\"cpu\"")
        })?
    }
}

#[cube(launch_unchecked)]
fn shortest_paths(input: &[u32], state: &mut [u32], heap: &mut [u32], status: &mut [u32]) {
    let search = ABSOLUTE_POS;
    if search >= status.len() {
        terminate!();
    }
    let meta = (search / 2) * 6;
    let weight = search % 2;
    let nodes = input[meta] as usize;
    let offset_start = input[meta + 1] as usize;
    let edge_start = input[meta + 2] as usize;
    let start = input[meta + 3] as usize + weight * (4 * nodes + 1);
    let capacity = input[meta + 5] as usize;
    let heap_start = input[meta + 4] as usize + weight * capacity * 2;
    status[search] = 0;
    for i in 0..nodes {
        state[start + i] = u32::MAX;
        state[start + nodes + i] = 0;
        state[start + 2 * nodes + i] = 0;
    }
    state[start] = 0;
    state[start + 4 * nodes] = 0;
    heap[heap_start] = 0;
    heap[heap_start + 1] = 0;
    let mut length = 1usize;
    let mut settled = 0usize;

    while length > 0 {
        let index = heap[heap_start] as usize;
        let cost = heap[heap_start + 1];
        length -= 1;
        if length > 0 {
            let last_index = heap[heap_start + 2 * length];
            let last_cost = heap[heap_start + 2 * length + 1];
            let mut pos = 0usize;
            let mut child = 1usize;
            // Match BinaryHeap's sift-down-to-bottom, including choosing the right child on ties.
            while child + 1 < length {
                if heap[heap_start + 2 * child + 1] >= heap[heap_start + 2 * (child + 1) + 1] {
                    child += 1;
                }
                heap[heap_start + 2 * pos] = heap[heap_start + 2 * child];
                heap[heap_start + 2 * pos + 1] = heap[heap_start + 2 * child + 1];
                pos = child;
                child = 2 * pos + 1;
            }
            if child < length {
                heap[heap_start + 2 * pos] = heap[heap_start + 2 * child];
                heap[heap_start + 2 * pos + 1] = heap[heap_start + 2 * child + 1];
                pos = child;
            }
            while pos > 0 {
                let parent = (pos - 1) / 2;
                if last_cost >= heap[heap_start + 2 * parent + 1] {
                    break;
                }
                heap[heap_start + 2 * pos] = heap[heap_start + 2 * parent];
                heap[heap_start + 2 * pos + 1] = heap[heap_start + 2 * parent + 1];
                pos = parent;
            }
            heap[heap_start + 2 * pos] = last_index;
            heap[heap_start + 2 * pos + 1] = last_cost;
        }
        if cost <= state[start + 2 * nodes + index] {
            if settled >= nodes {
                status[search] = 2;
                terminate!();
            }
            state[start + 3 * nodes + settled] = index as u32;
            settled += 1;
            let begin = input[offset_start + index] as usize;
            let end = input[offset_start + index + 1] as usize;
            for edge in begin..end {
                let next = input[edge_start + 3 * edge] as usize;
                let edge_cost = input[edge_start + 3 * edge + weight + 1];
                if cost > u32::MAX - edge_cost {
                    status[search] = 1;
                    terminate!();
                }
                let new_cost = cost + edge_cost;
                if state[start + next] == u32::MAX || state[start + 2 * nodes + next] > new_cost {
                    state[start + next] = index as u32;
                    state[start + nodes + next] = edge as u32;
                    state[start + 2 * nodes + next] = new_cost;
                    if length >= capacity {
                        status[search] = 2;
                        terminate!();
                    }
                    let mut pos = length;
                    length += 1;
                    // Equal costs do not sift up, preserving the CPU's strict improvement/tie rules.
                    while pos > 0 {
                        let parent = (pos - 1) / 2;
                        if new_cost >= heap[heap_start + 2 * parent + 1] {
                            break;
                        }
                        heap[heap_start + 2 * pos] = heap[heap_start + 2 * parent];
                        heap[heap_start + 2 * pos + 1] = heap[heap_start + 2 * parent + 1];
                        pos = parent;
                    }
                    heap[heap_start + 2 * pos] = next as u32;
                    heap[heap_start + 2 * pos + 1] = new_cost;
                }
            }
        }
    }
    state[start + 4 * nodes] = settled as u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routes_batches_with_ties_and_isolated_nodes(router: &GpuRouter) {
        assert!(router.search_batch(&[]).unwrap().is_empty());
        let offsets = [0, 3, 5, 6, 7, 7];
        let edges = [
            (1, 5, 2),
            (2, 0, 2),
            (3, 0, 4),
            (4, 0, 1),
            (2, 0, 1),
            (1, 0, 1),
            (4, 0, 1),
        ];
        let inputs = [
            CsrGraph {
                offsets: &offsets,
                edges: &edges,
            },
            CsrGraph {
                offsets: &[0, 0],
                edges: &[],
            },
            CsrGraph {
                offsets: &[0, 1, 1, 1],
                edges: &[(1, u32::MAX, 123)],
            },
        ];
        let trees = router.search_batch(&inputs).unwrap();
        let (adjusted, intact) = &trees[0];
        assert_eq!(adjusted.order, [0, 2, 3, 1, 4]);
        assert_eq!(adjusted.parent, [0, 2, 0, 0, 3]);
        assert_eq!(adjusted.edge, [0, 5, 1, 2, 6]);
        assert_eq!(adjusted.cost, [0; 5]);
        assert_eq!(intact.order, [0, 1, 2, 4, 3]);
        assert_eq!(intact.parent, [0, 0, 0, 0, 1]);
        assert_eq!(intact.edge, [0, 0, 1, 2, 3]);
        assert_eq!(intact.cost, [0, 2, 2, 4, 3]);
        assert_eq!(trees[1].0.order, [0]);
        assert_eq!(trees[1].1.parent, [0]);
        assert_eq!(trees[2].0.cost, [0, u32::MAX, 0]);
        assert_eq!(trees[2].0.parent, [0, 0, usize::MAX]);
        assert_eq!(trees[2].1.cost, [0, 123, 0]);
    }

    fn reports_overflow_and_invalid_graphs(router: &GpuRouter) {
        let overflow = CsrGraph {
            offsets: &[0, 1, 2, 2],
            edges: &[(1, u32::MAX, 1), (2, 1, 1)],
        };
        let error = router.search_batch(&[overflow]).err().unwrap();
        assert!(error.to_string().contains("Path cost overflow"));
        for invalid in [
            CsrGraph {
                offsets: &[],
                edges: &[],
            },
            CsrGraph {
                offsets: &[0, 1],
                edges: &[],
            },
            CsrGraph {
                offsets: &[0, 1],
                edges: &[(1, 1, 1)],
            },
        ] {
            assert!(router.search_batch(&[invalid]).is_err());
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal 3 GPU"]
    fn metal_routes_batches_and_reports_errors() {
        let router = GpuRouter::metal().unwrap();
        routes_batches_with_ties_and_isolated_nodes(&router);
        reports_overflow_and_invalid_graphs(&router);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    #[ignore = "requires an NVIDIA GPU and CUDA runtime compiler"]
    fn cuda_routes_batches_and_reports_errors() {
        let router = GpuRouter::cuda().unwrap();
        routes_batches_with_ties_and_isolated_nodes(&router);
        reports_overflow_and_invalid_graphs(&router);
    }
}
