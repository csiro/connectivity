use ndarray::{Array3, Array2, Array1};
use std::collections::HashMap;
use pathfinding::prelude::build_path;
use rayon::prelude::*;
use anyhow::Result;
// local modules
use crate::window;
use crate::utils;
use crate::metrics;
use crate::affine;
use crate::graph;
use crate::routing;
use crate::overview;
use affine::Affine;
use window::FocalWindow;
use graph::{Graph, EdgeData};
use routing::{Path, GraphDijkstraExt};
use overview::Resampling;


/// Core connectivity function
/// # Returns
/// A 2D array of connectivity values for each cell at the native resolution.
pub fn conn(
    cond_array: &Option<Array2<f32>>,
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
        let cond_map = overview::make_overview(cond_base, levels, offsets, Resampling::Average).expect("Failed average resampling.");
        let cell_weights = overview::make_overview(cond_base, levels, offsets, Resampling::Count)
            .expect("Failed to build valid-count weights.");
        // Ensure cell-counts has the same keys and dimension as condition array map;
        utils::check_dims(&cell_weights, &cond_map)?;

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
            Vec::new()  // Empty vec if None
        };

        // Initialize output with zeros
        let mut outarray = Array2::<f32>::zeros((nrows, ncols));

        // Parallel iteration over rows in the thread pool
        let out_vec: Vec<(usize, Vec<f32>)> = (0..nrows)
            .into_par_iter()
            .map(|i| {
                let mut row_result = vec![f32::NAN; ncols];
            
                for j in 0..ncols {
                    // Skip masked areas
                    if mask_array[[i, j]] {
                        continue;
                    }

                    // Get the transgrid values for ij cell for the current climate
                    let ij_values: Array1<f32> = utils::get_current(&trans_maps, i, j);
                    // Pre-allocate window hashmap
                    let mut windows: HashMap::<i32, FocalWindow> = HashMap::with_capacity(num_levels);
                    // Build window for each level for the cell ij
                    for &level in cond_map.keys() {
                        let win = FocalWindow::from_data(
                            i as i32,
                            j as i32,
                            level,
                            window_size,
                            outer_window,
                            offsets,
                            &cond_map,
                            &trans_maps,
                            &ij_values,
                            &cell_weights,
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
                    );
                    // Calculate all reachable paths using weighted distance by conditon; altered condition
                    let nodes_altered  = the_graph.dijkstra(Path::Adjusted);
                    // Using unweighted distance, i.e. intact condition case for the denominator
                    let nodes_intact = the_graph.dijkstra(Path::Intact); 
                    
                    let mut cell_paths: Vec<EdgeData> = Vec::with_capacity(nodes_altered.len());

                    // HashMap iteration order is non-deterministic; sort targets so repeated
                    // tile runs produce bit-stable accumulation at shared boundaries.
                    let mut targets: Vec<u32> = nodes_altered.keys().copied().collect();
                    targets.sort_unstable();

                    for k in targets {
                        // Calcaulate optimal path for each reachable path
                        let optim_path = build_path(&k, &nodes_altered);
                        // Get the intact distance from source; divided by 100 to cancel out from path adjacency
                        let dist_intact: f32 = nodes_intact.get(&k).expect("Node not found!").1 as f32 / 100.0;
                        // Get the path info for each target segment/node
                        cell_paths.push(routing::path_distance(&the_graph, &optim_path, dist_intact));
                    }

                    // Calculate BERI or Connectedness
                    row_result[j] = if lambdas.is_empty() {
                        0.0
                    } else {
                        let sum: f32 = lambdas.iter().map(|&lambda| {
                            if run_beri {
                                metrics::beri_score(&cell_paths, lambda)
                            } else {
                                metrics::connectedness(&cell_paths, lambda)
                            }
                        }).sum();
                        
                        sum / lambdas.len() as f32
                    };
                }

                (i, row_result)
            })
            .collect();

        // Write back results into outarray
        for (i, row) in out_vec {
            for (j, val) in row.into_iter().enumerate() {
                outarray[[i, j]] = val;
            }
        }

        Ok(outarray)
    } else {
        anyhow::bail!("No base level (key=1) in condition array");
    }  
}
