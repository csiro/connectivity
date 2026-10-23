use anyhow::Result;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Array3, Axis};
use std::collections::HashMap;
// local modules
use crate::affine;
use crate::graph;
use crate::metrics;
use crate::overview;
use crate::routing;
use crate::utils;
use crate::window;
use affine::Affine;
use graph::Graph;
use overview::Resampling;
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "cuda", any(target_os = "linux", target_os = "windows"))
))]
use rayon::slice::ParallelSliceMut;
use routing::{Device, GraphDijkstraExt};
use window::{FocalWindow, WindowMode};

/// Core connectivity function
/// # Returns
/// A 2D array of connectivity values for each cell at the native resolution.
pub fn conn(
    cond_array: &Option<Array2<f32>>,
    trav_array: Option<&Array2<f32>>,
    trans_arrays: Option<&[Array3<f32>]>,
    transform_map: &HashMap<i32, Affine>,
    mask_array: &Array2<bool>,
    levels: &Vec<usize>,
    lambdas: &[f32],
    is_geo: bool,
    max_cost: f32,
    window_size: i32,
    outer_window: i32,
    offsets: (usize, usize),
    window_mode: WindowMode,
    pa_to_pa: bool,
    device: Device,
) -> Result<Array2<f32>> {
    // If transgrids are provided run BERI, otherwise connectedness.
    let run_beri = trans_arrays
        .map(|arrays| !arrays.is_empty())
        .unwrap_or(false);

    // Check condition dictionay was not empty and run the code for level 1 (original resolution)
    if let Some(cond_base) = cond_array {
        let (nrows, ncols) = (cond_base.shape()[0], cond_base.shape()[1]);
        let num_levels = levels.len();

        // Generate overviews
        let cond_map =
            overview::make_overview(cond_base, levels, offsets, Resampling::Average, None)
                .expect("Failed average resampling.");
        let weight_method = if is_geo {
            Resampling::Area
        } else {
            Resampling::Count
        };
        let base_transform = transform_map
            .get(&1)
            .expect("Missing base-level transform (key=1).");
        let cell_weights = overview::make_overview(
            cond_base,
            levels,
            offsets,
            weight_method,
            Some(base_transform),
        )
        .expect("Failed to build cell weights.");
        // Ensure cell-counts has the same keys and dimension as condition array map;
        utils::check_dims(&cell_weights, &cond_map)?;

        // PARC target-gating: build a PA-membership overview from the mask. Each cell holds the
        // fraction of its condition-valid sub-cells that fall on a protected area. The base
        // indicator is NaN where condition is NaN so the average shares the cell-weights valid
        // set, making `num_cells * pa` equal the protected area at every level. When disabled,
        // no gating is applied and the output is bit-identical to the original PARC path.
        let pa_map: Option<HashMap<i32, Array2<f32>>> = if pa_to_pa {
            let pa_indicator = Array2::<f32>::from_shape_fn((nrows, ncols), |(i, j)| {
                if cond_base[[i, j]].is_nan() {
                    f32::NAN
                } else if !mask_array[[i, j]] {
                    1.0
                } else {
                    0.0
                }
            });
            let map =
                overview::make_overview(&pa_indicator, levels, offsets, Resampling::Average, None)
                    .expect("Failed to build PA-membership overview.");
            utils::check_dims(&map, &cond_map)?;
            Some(map)
        } else {
            None
        };

        // Optional traversal overview (resistance-derived, averaged like condition). Built
        // only when a resistance raster is supplied; when None the graph weight falls back to
        // condition, so the output is bit-identical to a resistance-free run.
        let trav_map: Option<HashMap<i32, Array2<f32>>> = match trav_array {
            Some(arr) => {
                let map = overview::make_overview(arr, levels, offsets, Resampling::Average, None)
                    .expect("Failed average resampling for traversal.");
                utils::check_dims(&map, &cond_map)?;
                Some(map)
            }
            None => None,
        };

        // Generate overviews for all scenarios
        let trans_maps: Vec<HashMap<i32, Array3<f32>>> = if let Some(arrays) = trans_arrays {
            arrays
                .iter()
                .map(|arr| {
                    overview::make_overview_3d(arr, levels, offsets, Resampling::Average)
                        .expect("Failed average resampling.")
                })
                .collect()
        } else {
            Vec::new() // Empty vec if None
        };

        // Initialize output with NaNs so masked cells keep the previous output semantics.
        let mut outarray = Array2::<f32>::from_elem((nrows, ncols), f32::NAN);
        let mut sorted_levels: Vec<i32> = cond_map.keys().copied().collect();
        sorted_levels.sort_unstable();
        let max_level = sorted_levels.last().copied().unwrap_or(1);

        let build_graph = |i: usize, j: usize| {
            // Keep windows indexed in ascending level order.
            let ij_values: Array1<f32> = utils::get_current(&trans_maps, i, j);
            let mut windows = Vec::with_capacity(num_levels);
            for &level in &sorted_levels {
                let win = FocalWindow::from_data(
                    i as i32,
                    j as i32,
                    level,
                    max_level,
                    window_size,
                    outer_window,
                    offsets,
                    &cond_map,
                    &trans_maps,
                    &ij_values,
                    &cell_weights,
                    pa_map.as_ref(),
                    trav_map.as_ref(),
                    window_mode,
                );
                windows.push((level, win));
            }
            Graph::from_data(
                i as i32,
                j as i32,
                max_cost,
                &windows,
                transform_map,
                is_geo,
                offsets,
                window_mode,
                pa_to_pa,
            )
        };
        let score = |cell_paths: &[graph::EdgeData]| {
            if lambdas.is_empty() {
                0.0
            } else {
                let sum: f32 = lambdas
                    .iter()
                    .map(|&lambda| {
                        if run_beri {
                            metrics::beri_score(cell_paths, lambda)
                        } else {
                            metrics::connectedness(cell_paths, lambda)
                        }
                    })
                    .sum();
                sum / lambdas.len() as f32
            }
        };

        match device {
            Device::Cpu => {
                // CPU workers consume one graph at a time without batch allocations.
                outarray
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row_result)| {
                        for (j, value) in row_result.iter_mut().enumerate() {
                            if !mask_array[[i, j]] {
                                *value = score(&build_graph(i, j).paths());
                            }
                        }
                    });
            }
            #[cfg(any(
                all(feature = "metal", target_os = "macos"),
                all(feature = "cuda", any(target_os = "linux", target_os = "windows"))
            ))]
            Device::Gpu(_) => {
                // Cross row boundaries so narrow rasters can fill multiple GPU workgroups.
                // Bound retained graphs per worker; large windows flush sooner by edge count.
                const BATCH_CELLS: usize = 256;
                const BATCH_EDGES: usize = 262_144;
                outarray
                    .as_slice_mut()
                    .expect("New output array is contiguous")
                    .par_chunks_mut(BATCH_CELLS)
                    .enumerate()
                    .try_for_each(|(chunk, output)| -> Result<()> {
                        let mut graphs = Vec::with_capacity(output.len());
                        let mut positions = Vec::with_capacity(output.len());
                        let mut edges = 0usize;
                        let flush = |graphs: &mut Vec<Graph>,
                                     positions: &mut Vec<usize>,
                                     output: &mut [f32]|
                         -> Result<()> {
                            if !graphs.is_empty() {
                                for (&pos, paths) in positions.iter().zip(device.paths(graphs)?) {
                                    output[pos] = score(&paths);
                                }
                                graphs.clear();
                                positions.clear();
                            }
                            Ok(())
                        };

                        for pos in 0..output.len() {
                            let cell = chunk * BATCH_CELLS + pos;
                            let (i, j) = (cell / ncols, cell % ncols);
                            if mask_array[[i, j]] {
                                continue;
                            }
                            let graph = build_graph(i, j);
                            let graph_edges = graph.data.len();
                            if edges.saturating_add(graph_edges) > BATCH_EDGES {
                                flush(&mut graphs, &mut positions, output)?;
                                edges = 0;
                            }
                            graphs.push(graph);
                            positions.push(pos);
                            edges += graph_edges;
                            // A graph larger than the budget runs alone.
                            if edges >= BATCH_EDGES {
                                flush(&mut graphs, &mut positions, output)?;
                                edges = 0;
                            }
                        }
                        flush(&mut graphs, &mut positions, output)
                    })?;
            }
        }

        Ok(outarray)
    } else {
        anyhow::bail!("No base level (key=1) in condition array");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal GPU"]
    fn metal_matches_cpu() {
        gpu_matches_cpu("metal");
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    #[ignore = "requires an NVIDIA GPU and CUDA toolkit"]
    fn cuda_matches_cpu() {
        gpu_matches_cpu("cuda");
    }

    #[cfg(any(
        all(feature = "metal", target_os = "macos"),
        all(feature = "cuda", any(target_os = "linux", target_os = "windows"))
    ))]
    fn gpu_matches_cpu(name: &str) {
        let device = Device::parse(name).unwrap();
        for (rows, cols) in [(9, 39), (39, 9)] {
            let cond = Some(Array2::from_shape_fn((rows, cols), |(i, j)| {
                if (i * 7 + j * 11) % 29 == 0 {
                    f32::NAN
                } else {
                    ((i * 7 + j * 13) % 10 + 1) as f32 / 10.0
                }
            }));
            let mask = Array2::from_shape_fn((rows, cols), |(i, j)| {
                cond.as_ref().unwrap()[[i, j]].is_nan() || (i * 3 + j * 5) % 7 == 0
            });
            let traversal = Array2::from_shape_fn((rows, cols), |(i, j)| {
                if (i + j * 3) % 13 == 0 {
                    f32::NAN
                } else {
                    ((i * 5 + j * 3) % 11) as f32 / 10.0
                }
            });
            let transitions = vec![Array3::from_shape_fn((rows, cols, 2), |(i, j, k)| {
                if (i * 3 + j * 5 + k) % 31 == 0 {
                    f32::NAN
                } else {
                    ((i * 7 + j * 3 + k * 2) % 23) as f32 / 23.0
                }
            })];
            let levels = vec![1, 2, 4];
            let transforms: HashMap<i32, Affine> = levels
                .iter()
                .map(|&level| {
                    let s = 1000.0 * level as f64;
                    (
                        level as i32,
                        Affine::from((s, 0.0, 500_000.0, 0.0, -s, 6_000_000.0)),
                    )
                })
                .collect();
            // Flat batches cross rows and include a partial tail, masked holes and directed fringe links.
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                for mode in [WindowMode::Circular, WindowMode::Square] {
                    for beri in [false, true] {
                        for pa_to_pa in [false, true] {
                            let run = |device| {
                                pool.install(|| {
                                    conn(
                                        &cond,
                                        Some(&traversal),
                                        beri.then_some(transitions.as_slice()),
                                        &transforms,
                                        &mask,
                                        &levels,
                                        &[2.0, 20.0],
                                        false,
                                        20.0,
                                        3,
                                        5,
                                        (3, 5),
                                        mode,
                                        pa_to_pa,
                                        device,
                                    )
                                    .unwrap()
                                })
                            };
                            let cpu = run(Device::Cpu);
                            let gpu = run(device);
                            assert!(cpu.iter().zip(&gpu).all(|(a, b)| a.to_bits() == b.to_bits()),
                                "{name} differs from CPU: {rows}x{cols}, {mode:?}, beri={beri}, pa_to_pa={pa_to_pa}, threads={threads}");
                        }
                    }
                }
            }
        }
    }

    // Repeated runs must give bit-identical output.
    #[test]
    fn repeats_exactly() {
        let cond = Some(Array2::from_shape_fn((32, 32), |(i, j)| {
            ((i * 7 + j * 13) % 10 + 1) as f32 / 10.0
        }));
        let mask = Array2::from_elem((32, 32), false);
        let levels = vec![1, 2, 4];
        let transforms: HashMap<i32, Affine> = levels
            .iter()
            .map(|&l| {
                let s = 1000.0 * l as f64;
                (l as i32, Affine::from((s, 0.0, 0.0, 0.0, -s, 32_000.0)))
            })
            .collect();
        let run = || {
            conn(
                &cond,
                None,
                None,
                &transforms,
                &mask,
                &levels,
                &[2.0, 20.0],
                false,
                2.0,
                3,
                5,
                (0, 0),
                WindowMode::Circular,
                false,
                Device::Cpu,
            )
            .unwrap()
        };

        let first = run();
        assert!(first.iter().any(|&v| v > 0.0));
        for _ in 0..3 {
            let next = run();
            assert!(first
                .iter()
                .zip(&next)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn multilevel_regression() {
        let cond = Some(Array2::from_shape_fn((13, 14), |(i, j)| {
            if (i * 7 + j * 11) % 29 == 0 {
                f32::NAN
            } else {
                ((i * 7 + j * 13) % 10 + 1) as f32 / 10.0
            }
        }));
        let condition = cond.as_ref().unwrap();
        let mask = Array2::from_shape_fn((13, 14), |(i, j)| {
            !condition[[i, j]].is_finite() || (i * 3 + j * 5) % 7 == 0
        });
        let traversal = Array2::from_shape_fn((13, 14), |(i, j)| {
            if (i + j * 3) % 13 == 0 {
                f32::NAN
            } else {
                ((i * 5 + j * 3) % 11) as f32 / 10.0
            }
        });
        let transitions: Vec<Array3<f32>> = (0..2)
            .map(|scenario| {
                Array3::from_shape_fn((13, 14, 2), |(i, j, k)| {
                    if (i * 3 + j * 5 + scenario + k) % 31 == 0 {
                        f32::NAN
                    } else {
                        ((i * 7 + j * 3 + scenario * 11 + k * 2) % 23) as f32 / 23.0
                    }
                })
            })
            .collect();
        let levels = vec![1, 2, 4];
        let offsets = (3, 5);
        let transforms: HashMap<i32, Affine> = levels
            .iter()
            .map(|&level| {
                let scale = 1000.0 * level as f64;
                let row0 = offsets.0 / level * level;
                let col0 = offsets.1 / level * level;
                (
                    level as i32,
                    Affine::from((
                        scale,
                        0.0,
                        500_000.0 + col0 as f64 * 1000.0,
                        0.0,
                        -scale,
                        6_000_000.0 - row0 as f64 * 1000.0,
                    )),
                )
            })
            .collect();
        let sample_cells = [(0, 1), (3, 5), (6, 8), (12, 13)];

        // Frozen CPU outputs from before the dense-array rewrite, including fringe links,
        // traversal fallback and missing similarities. Allow platform float-math variation.
        for (mode, beri, pa_to_pa, expected) in [
            (
                WindowMode::Circular,
                false,
                false,
                [0.035036724, 0.060928639, 0.057472181, 0.023413222],
            ),
            (
                WindowMode::Circular,
                false,
                true,
                [0.031013189, 0.051940750, 0.050136760, 0.017623197],
            ),
            (
                WindowMode::Circular,
                true,
                false,
                [0.287769228, 0.330879152, 0.336351663, 0.270322204],
            ),
            (
                WindowMode::Circular,
                true,
                true,
                [0.287769228, 0.330879152, 0.336351663, 0.270322204],
            ),
            (
                WindowMode::Square,
                false,
                false,
                [0.036365397, 0.062699854, 0.057887629, 0.023465034],
            ),
            (
                WindowMode::Square,
                false,
                true,
                [0.032288730, 0.053552385, 0.050358951, 0.017685590],
            ),
            (
                WindowMode::Square,
                true,
                false,
                [0.288273066, 0.334522188, 0.339534730, 0.269480884],
            ),
            (
                WindowMode::Square,
                true,
                true,
                [0.288273066, 0.334522188, 0.339534730, 0.269480884],
            ),
        ] {
            let output = conn(
                &cond,
                Some(&traversal),
                beri.then_some(transitions.as_slice()),
                &transforms,
                &mask,
                &levels,
                &[2.0, 20.0],
                false,
                20.0,
                3,
                5,
                offsets,
                mode,
                pa_to_pa,
                Device::Cpu,
            )
            .unwrap();

            for (cell, expected) in sample_cells.iter().zip(expected) {
                let actual = output[*cell];
                assert!(actual.is_finite());
                assert!(
                    (actual - expected).abs() <= 1.0e-6,
                    "{mode:?}, beri={beri}, pa_to_pa={pa_to_pa}, cell={cell:?}: {actual} != {expected}"
                );
            }
            for ((i, j), &masked) in mask.indexed_iter() {
                if masked {
                    assert!(output[[i, j]].is_nan());
                }
            }
        }
    }
}
