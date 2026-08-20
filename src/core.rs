use anyhow::Result;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Array3, Axis};
use pathfinding::prelude::build_path;
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
use graph::{EdgeData, Graph};
use overview::Resampling;
use routing::{GraphDijkstraExt, Path};
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
            let map = overview::make_overview(
                &pa_indicator, 
                levels, 
                offsets, 
                Resampling::Average, 
                None
            )
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
                let map =
                    overview::make_overview(arr, levels, offsets, Resampling::Average, None)
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

        // Parallel iteration over rows in the thread pool
        outarray
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row_result)| {
                for j in 0..ncols {
                    // Skip masked areas
                    if mask_array[[i, j]] {
                        continue;
                    }

                    // Get the transgrid values for ij cell for the current climate
                    let ij_values: Array1<f32> = utils::get_current(&trans_maps, i, j);
                    // Pre-allocate window hashmap
                    let mut windows: HashMap<i32, FocalWindow> = HashMap::with_capacity(num_levels);
                    // Build window for each level for the cell ij
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
                        windows.insert(level, win);
                    }

                    // Build a Graph for the cell ij using multi-res windows
                    let the_graph = Graph::from_data(
                        i as i32,
                        j as i32,
                        max_cost,
                        &windows,
                        transform_map,
                        is_geo,
                        offsets,
                        window_mode,
                        pa_to_pa,
                    );
                    // Calculate all reachable paths using weighted distance by conditon; altered condition
                    let nodes_altered = the_graph.dijkstra(Path::Adjusted);
                    // Using unweighted distance, i.e. intact condition case for the denominator
                    let nodes_intact = the_graph.dijkstra(Path::Intact);

                    let mut cell_paths: Vec<EdgeData> = Vec::with_capacity(nodes_altered.len());

                    for &k in nodes_altered.keys() {
                        // Calcaulate optimal path for each reachable path
                        let optim_path = build_path(&k, &nodes_altered);
                        // Get the intact distance from source; divided by 100 to cancel out from path adjacency
                        let dist_intact: f32 =
                            nodes_intact.get(&k).expect("Node not found!").1 as f32 / 100.0;
                        // Get the path info for each target segment/node
                        cell_paths.push(routing::path_distance(
                            &the_graph,
                            &optim_path,
                            dist_intact,
                        ));
                    }

                    // Calculate BERI or Connectedness
                    row_result[j] = if lambdas.is_empty() {
                        0.0
                    } else {
                        let sum: f32 = lambdas
                            .iter()
                            .map(|&lambda| {
                                if run_beri {
                                    metrics::beri_score(&cell_paths, lambda)
                                } else {
                                    metrics::connectedness(&cell_paths, lambda)
                                }
                            })
                            .sum();

                        sum / lambdas.len() as f32
                    };
                }
            });

        Ok(outarray)
    } else {
        anyhow::bail!("No base level (key=1) in condition array");
    }
}
