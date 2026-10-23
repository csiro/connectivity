//! Frozen 95554da CPU path (window, graph, routing and metrics), kept as the test reference.
//!
//! `conn` is the baseline `core::conn`, split into `Prepared` (the per-call overviews) and its
//! per-cell steps so the canonical reference can reuse the legacy construction unchanged.
//! Distances and geometry come from `Affine::xy`, `distances::distance_km` and the window
//! helpers, never from production templates or tables.
pub mod builder;
pub mod canonical;
pub mod graph;
pub mod metrics;
pub mod routing;
pub mod window;

use crate::affine::Affine;
use crate::overview::{self, Resampling};
use crate::utils;
use crate::window::WindowMode;
use anyhow::Result;
use graph::{EdgeData, Graph};
use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2, Array3, Axis};
use routing::GraphDijkstraExt;
use std::collections::HashMap;
use window::FocalWindow;

/// Get the transgrid values for ij cell (baseline `utils::get_current`)
pub fn get_current(trans_maps: &[HashMap<i32, Array3<f32>>], i: usize, j: usize) -> Array1<f32> {
    // Guard against empty trans_maps
    if trans_maps.is_empty() {
        return Array1::zeros(0);
    }

    let trans_array = &trans_maps[0];
    if let Some(array3) = trans_array.get(&1) {
        if i < array3.shape()[0] && j < array3.shape()[1] {
            return array3.slice(s![i, j, ..]).to_owned(); // returns Array1<f32>
        }
    }
    // Return empty array if anything fails
    Array1::zeros(0)
}

/// Overviews and settings shared by every focal cell of one call (baseline `conn` set-up).
pub struct Prepared<'a> {
    pub cond_map: HashMap<i32, Array2<f32>>,
    pub cell_weights: HashMap<i32, Array2<f32>>,
    pub pa_map: Option<HashMap<i32, Array2<f32>>>,
    pub trav_map: Option<HashMap<i32, Array2<f32>>>,
    pub trans_maps: Vec<HashMap<i32, Array3<f32>>>,
    pub sorted_levels: Vec<i32>,
    pub max_level: i32,
    pub run_beri: bool,
    pub transform_map: &'a HashMap<i32, Affine>,
    pub lambdas: &'a [f32],
    pub is_geo: bool,
    pub max_cost: f32,
    pub window_size: i32,
    pub outer_window: i32,
    pub offsets: (usize, usize),
    pub window_mode: WindowMode,
    pub pa_to_pa: bool,
}

impl<'a> Prepared<'a> {
    pub fn new(
        cond_base: &Array2<f32>,
        trav_array: Option<&Array2<f32>>,
        trans_arrays: Option<&[Array3<f32>]>,
        transform_map: &'a HashMap<i32, Affine>,
        mask_array: &Array2<bool>,
        levels: &Vec<usize>,
        lambdas: &'a [f32],
        is_geo: bool,
        max_cost: f32,
        window_size: i32,
        outer_window: i32,
        offsets: (usize, usize),
        window_mode: WindowMode,
        pa_to_pa: bool,
    ) -> Result<Self> {
        // If transgrids are provided run BERI, otherwise connectedness.
        let run_beri = trans_arrays
            .map(|arrays| !arrays.is_empty())
            .unwrap_or(false);
        let (nrows, ncols) = (cond_base.shape()[0], cond_base.shape()[1]);

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
        utils::check_dims(&cell_weights, &cond_map)?;

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
                None,
            )
            .expect("Failed to build PA-membership overview.");
            utils::check_dims(&map, &cond_map)?;
            Some(map)
        } else {
            None
        };

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

        let mut sorted_levels: Vec<i32> = cond_map.keys().copied().collect();
        sorted_levels.sort_unstable();
        let max_level = sorted_levels.last().copied().unwrap_or(1);

        Ok(Self {
            cond_map,
            cell_weights,
            pa_map,
            trav_map,
            trans_maps,
            sorted_levels,
            max_level,
            run_beri,
            transform_map,
            lambdas,
            is_geo,
            max_cost,
            window_size,
            outer_window,
            offsets,
            window_mode,
            pa_to_pa,
        })
    }

    /// Windows for every level of cell ij, in ascending level order.
    pub fn windows(&self, i: usize, j: usize) -> Vec<(i32, FocalWindow)> {
        // Get the transgrid values for ij cell for the current climate
        let ij_values: Array1<f32> = get_current(&self.trans_maps, i, j);
        // Keep windows indexed in ascending level order
        let mut windows = Vec::with_capacity(self.sorted_levels.len());
        // Build window for each level for the cell ij
        for &level in &self.sorted_levels {
            let win = FocalWindow::from_data(
                i as i32,
                j as i32,
                level,
                self.max_level,
                self.window_size,
                self.outer_window,
                self.offsets,
                &self.cond_map,
                &self.trans_maps,
                &ij_values,
                &self.cell_weights,
                self.pa_map.as_ref(),
                self.trav_map.as_ref(),
                self.window_mode,
            );
            windows.push((level, win));
        }
        windows
    }

    /// The legacy graph of cell ij.
    pub fn graph(&self, i: usize, j: usize, windows: &[(i32, FocalWindow)]) -> Graph {
        Graph::from_data(
            i as i32,
            j as i32,
            self.max_cost,
            windows,
            self.transform_map,
            self.is_geo,
            self.offsets,
            self.window_mode,
            self.pa_to_pa,
        )
    }

    /// BERI or connectedness of one cell's paths, averaged over lambdas.
    pub fn score(&self, cell_paths: &[EdgeData]) -> f32 {
        if self.lambdas.is_empty() {
            0.0
        } else {
            let sum: f32 = self
                .lambdas
                .iter()
                .map(|&lambda| {
                    if self.run_beri {
                        metrics::beri_score(cell_paths, lambda)
                    } else {
                        metrics::connectedness(cell_paths, lambda)
                    }
                })
                .sum();

            sum / self.lambdas.len() as f32
        }
    }

    /// Legacy value of cell ij.
    pub fn cell(&self, i: usize, j: usize) -> f32 {
        let windows = self.windows(i, j);
        let the_graph = self.graph(i, j, &windows);
        let cell_paths = the_graph.paths();
        self.score(&cell_paths)
    }
}

/// Baseline `core::conn`.
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
    if let Some(cond_base) = cond_array {
        let prepared = Prepared::new(
            cond_base,
            trav_array,
            trans_arrays,
            transform_map,
            mask_array,
            levels,
            lambdas,
            is_geo,
            max_cost,
            window_size,
            outer_window,
            offsets,
            window_mode,
            pa_to_pa,
        )?;
        Ok(map_cells(cond_base, mask_array, |i, j| prepared.cell(i, j)))
    } else {
        anyhow::bail!("No base level (key=1) in condition array");
    }
}

/// Evaluate every unmasked cell in parallel over rows; masked cells stay NaN.
pub fn map_cells(
    cond_base: &Array2<f32>,
    mask_array: &Array2<bool>,
    cell: impl Fn(usize, usize) -> f32 + Sync,
) -> Array2<f32> {
    let mut outarray = Array2::<f32>::from_elem(cond_base.dim(), f32::NAN);
    outarray
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row_result)| {
            for j in 0..row_result.len() {
                if mask_array[[i, j]] {
                    continue;
                }
                row_result[j] = cell(i, j);
            }
        });
    outarray
}

#[cfg(test)]
mod tests {
    use super::canonical::{edge_order, Inject};
    use super::routing::Routed;
    use crate::testkit::{diff, Fixture};
    use rustc_hash::FxHashMap;

    /// Reached nodes with (adjusted, intact) integer path costs, keyed by legacy node id.
    fn costs(routed: &Routed) -> FxHashMap<u64, (u32, u32)> {
        let nodes = &routed.adjacency.nodes;
        routed
            .altered
            .order
            .iter()
            .map(|&i| (nodes[i], (routed.altered.cost[i], routed.intact.cost[i])))
            .collect()
    }

    // Reordering never changes the edge set, reachability or shortest-path costs
    #[test]
    fn canonical_invariants() {
        for seed in 0..60 {
            let fixture = Fixture::random(seed);
            let prepared = fixture.prepared();
            for ((i, j), &masked) in fixture.mask.indexed_iter() {
                if masked {
                    continue;
                }
                let windows = prepared.windows(i, j);
                let graph = prepared.graph(i, j, &windows);
                let keys = edge_order(&graph, fixture.offsets);
                let legacy = graph.route(graph.to_adjacency());
                let canonical = graph.route(graph.to_adjacency_in(&keys));
                assert_eq!(costs(&legacy), costs(&canonical), "seed {seed}, cell ({i}, {j})");
                assert_eq!(legacy.paths.len(), canonical.paths.len());
            }
        }
    }

    // Injecting the legacy distances themselves leaves the canonical output unchanged
    #[test]
    fn identity_injection() {
        for seed in 0..20 {
            let fixture = Fixture::random(seed);
            let prepared = fixture.prepared();
            let mut legacy_dist: FxHashMap<(u64, u64), f32> = FxHashMap::default();
            for ((i, j), &masked) in fixture.mask.indexed_iter() {
                if !masked {
                    let windows = prepared.windows(i, j);
                    for (&key, edge) in &prepared.graph(i, j, &windows).data {
                        legacy_dist.insert(key, edge.geo_dist);
                    }
                }
            }
            let inject = |u: u64, v: u64, _: bool| legacy_dist.get(&(u, v)).copied();
            let inject: Inject = &inject;
            let plain = fixture.canonical(None);
            let injected = fixture.canonical(Some(inject));
            assert!(crate::testkit::same_bits(&plain, &injected), "seed {seed}");
        }
    }

    // Ordering-only effect on fixtures; reported, masks must agree
    #[test]
    fn ordering_effect() {
        let mut total = 0;
        for seed in 0..60 {
            let fixture = Fixture::random(seed);
            let d = diff(&fixture.canonical(None), &fixture.legacy());
            assert_eq!(d.mask_mismatch, 0);
            total += d.differing;
            if d.differing > 0 {
                println!("seed {seed}: {d:?}");
            }
        }
        println!("ordering-only differing cells over 60 fixtures: {total}");
    }
}

#[cfg(test)]
mod frozen {
    use crate::testkit::Fixture;
    use crate::window::WindowMode;

    // Prints Legacy and Canonical values at the multilevel sample cells
    #[test]
    #[ignore]
    fn print_multilevel() {
        let cells = [(0, 1), (3, 5), (6, 8), (12, 13)];
        for mode in [WindowMode::Circular, WindowMode::Square] {
            for beri in [false, true] {
                for pa_to_pa in [false, true] {
                    let f = Fixture::multilevel(mode, beri, pa_to_pa);
                    let (legacy, canonical) = (f.legacy(), f.canonical(None));
                    let l: Vec<String> = cells.iter().map(|&c| format!("{:.9}", legacy[c])).collect();
                    let c: Vec<String> = cells.iter().map(|&c| format!("{:.9}", canonical[c])).collect();
                    let d: Vec<String> = cells.iter().map(|&c| format!("{:+.2e}", canonical[c] - legacy[c])).collect();
                    let bits: Vec<String> = cells.iter().map(|&c| format!("{:#010x}", canonical[c].to_bits())).collect();
                    println!("{mode:?} beri={beri} pa={pa_to_pa}\n  legacy    {l:?}\n  canonical {c:?}\n  diff      {d:?}\n  bits      {bits:?}");
                    let diff = crate::testkit::diff(&canonical, &legacy);
                    println!("  whole fixture: {diff:?}");
                }
            }
        }
    }
}
