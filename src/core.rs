use anyhow::Result;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, Axis};
use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;
// local modules
use crate::affine;
use crate::builder::{Context, Scratch};
use crate::overview;
use crate::tables::{Tables, STATUSES};
use crate::template::{Templates, CACHE_BUDGET};
use crate::utils;
use crate::window;
use affine::Affine;
use overview::Resampling;
use window::WindowMode;

/// Call-level resource statistics.
#[derive(Debug, Default, Clone)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct Stats {
    pub scratch: Vec<usize>, // retained bytes of each scratch instance
    pub cache_bytes: usize,  // shared templates and distance tables
    pub table_bytes: usize,  // distance tables alone
    pub cached: Vec<usize>,  // template phases cached per level
    pub edges: [u64; STATUSES], // edges by distance-table status (hits, then fallback reasons)
    pub setup_s: f64,        // overviews, templates, contexts and scratch
    pub process_s: f64,      // focal-cell loop
}

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
    conn_stats(
        cond_array,
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
    )
    .map(|(out, _)| out)
}

/// `conn` that also reports scratch memory and setup/processing time.
pub fn conn_stats(
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
) -> Result<(Array2<f32>, Stats)> {
    let start = Instant::now();
    // Check condition dictionay was not empty and run the code for level 1 (original resolution)
    if let Some(cond_base) = cond_array {
        let ctx = prepare(
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

        // One scratch per pool thread, sized up front; a row borrows its thread's scratch, so
        // focal cells only allocate if a window outgrows the initial capacity.
        let threads = rayon::current_num_threads();
        let scratch: Vec<Mutex<Scratch>> =
            (0..threads).map(|_| Mutex::new(Scratch::new(&ctx))).collect();
        let setup_s = start.elapsed().as_secs_f64();

        // Initialize output with NaNs so masked cells keep the previous output semantics.
        let ncols = cond_base.ncols();
        let mut outarray = Array2::<f32>::from_elem(cond_base.dim(), f32::NAN);
        outarray
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row_result)| {
                let slot = rayon::current_thread_index().unwrap_or(0) % threads;
                // A panicking cell poisons its scratch, but every cell resets what it uses
                let mut scratch = scratch[slot].lock().unwrap_or_else(PoisonError::into_inner);
                for j in 0..ncols {
                    // Skip masked areas
                    if mask_array[[i, j]] {
                        continue;
                    }
                    row_result[j] = scratch.eval(&ctx, i, j);
                }
            });

        let stats = Stats {
            scratch: scratch
                .iter()
                .map(|s| s.lock().unwrap_or_else(PoisonError::into_inner).bytes())
                .collect(),
            cache_bytes: ctx.templates.bytes + ctx.tables.bytes,
            table_bytes: ctx.tables.bytes,
            cached: ctx.templates.cached.clone(),
            edges: scratch.iter().fold([0; STATUSES], |mut acc, s| {
                let s = s.lock().unwrap_or_else(PoisonError::into_inner);
                acc.iter_mut().zip(s.counts).for_each(|(a, c)| *a += c);
                acc
            }),
            setup_s,
            process_s: start.elapsed().as_secs_f64() - setup_s,
        };
        Ok((outarray, stats))
    } else {
        anyhow::bail!("No base level (key=1) in condition array");
    }
}

/// Overviews and settings for one call, indexed by ascending level.
pub fn prepare<'a>(
    cond_base: &Array2<f32>,
    trav_array: Option<&Array2<f32>>,
    trans_arrays: Option<&[Array3<f32>]>,
    transform_map: &HashMap<i32, Affine>,
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
) -> Result<Context<'a>> {
    // If transgrids are provided run BERI, otherwise connectedness.
    let run_beri = trans_arrays
        .map(|arrays| !arrays.is_empty())
        .unwrap_or(false);
    let (nrows, ncols) = (cond_base.shape()[0], cond_base.shape()[1]);

    // Generate overviews
    let mut cond_map =
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
    let mut cell_weights = overview::make_overview(
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
    let mut trans_maps: Vec<HashMap<i32, Array3<f32>>> = if let Some(arrays) = trans_arrays {
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

    // Level-indexed vectors replace per-cell hash lookups
    let mut sorted_levels: Vec<i32> = cond_map.keys().copied().collect();
    sorted_levels.sort_unstable();
    let by_level = |map: &mut HashMap<i32, Array2<f32>>| -> Vec<Array2<f32>> {
        sorted_levels.iter().map(|l| map.remove(l).unwrap()).collect()
    };
    let cond = by_level(&mut cond_map);
    let weight = by_level(&mut cell_weights);
    let pa = pa_map.map(|mut map| by_level(&mut map));
    let trav = trav_map.map(|mut map| by_level(&mut map));
    // Standard layout keeps each cell's bands contiguous (the base level copies the input's
    // layout, e.g. a band-major view); values are unchanged.
    let trans: Vec<Vec<Array3<f32>>> = trans_maps
        .iter_mut()
        .map(|map| {
            sorted_levels
                .iter()
                .map(|l| {
                    let arr = map.remove(l).unwrap();
                    if arr.is_standard_layout() {
                        arr
                    } else {
                        arr.as_standard_layout().into_owned()
                    }
                })
                .collect()
        })
        .collect();
    // Flat band slices need standard layout and the condition grid's shape at every level
    let contiguous = trans.iter().all(|levels| {
        levels.iter().zip(&cond).all(|(t, c)| {
            t.is_standard_layout() && (t.dim().0, t.dim().1) == c.dim()
        })
    });
    let fringe: Vec<bool> = (0..sorted_levels.len())
        .map(|l| sorted_levels.get(l + 1) == Some(&(sorted_levels[l] * 2)))
        .collect();
    let transforms: Vec<Option<Affine>> =
        sorted_levels.iter().map(|l| transform_map.get(l).copied()).collect();
    let base = sorted_levels.iter().position(|&l| l == 1);

    // Window and level checks of the legacy window builder, which ran for each processed cell
    let mut outer = outer_window;
    if mask_array.iter().any(|&masked| !masked) {
        // Ensure neighborhood sizes are odd
        if window_size % 2 == 0 {
            panic!("Window size must be odd-numbered");
        }
        // If outer_win is lower than win_size, use the regular win_size
        outer = if outer_window < window_size {
            window_size
        } else if outer_window % 2 == 0 {
            panic!("Outer window size must be odd-numbered");
        } else {
            outer_window
        };
        base.expect("Base condition level not found!");
        for transform in &transforms {
            transform.expect("Missing level in Affine set.");
        }
    }

    // Tile edges that cut the first or last cells of each level grid
    let clip = sorted_levels
        .iter()
        .map(|&level| {
            let (r0, c0) = (offsets.0 as i32, offsets.1 as i32);
            let (r1, c1) = (r0 + nrows as i32, c0 + ncols as i32);
            [r0 % level != 0, r1 % level != 0, c0 % level != 0, c1 % level != 0]
        })
        .collect();
    let effective: Vec<i32> = (0..sorted_levels.len())
        .map(|l| if l + 1 == sorted_levels.len() { outer } else { window_size })
        .collect();
    // Distance tables first, then window templates, within one shared cache budget
    let dims: Vec<(usize, usize)> = cond.iter().map(|a| a.dim()).collect();
    let tables = Tables::new(&sorted_levels, &transforms, &dims, offsets, &fringe, is_geo, CACHE_BUDGET);
    let templates = Templates::new(
        &sorted_levels,
        &effective,
        window_size,
        window_mode,
        mask_array,
        offsets,
        CACHE_BUDGET - tables.bytes,
        None,
    );

    Ok(Context {
        levels: sorted_levels,
        cond,
        weight,
        pa,
        trav,
        trans,
        contiguous,
        base,
        transforms,
        fringe,
        clip,
        templates,
        tables,
        base_shape: (nrows as i32, ncols as i32),
        offsets,
        mode: window_mode,
        is_geo,
        factor: max_cost,
        pa_to_pa,
        lambdas,
        run_beri,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
                &cond, None, None, &transforms, &mask, &levels, &[2.0, 20.0], false, 2.0, 3, 5,
                (0, 0), WindowMode::Circular, false,
            )
            .unwrap()
        };

        let first = run();
        assert!(first.iter().any(|&v| v > 0.0));
        for _ in 0..3 {
            let next = run();
            assert!(first.iter().zip(&next).all(|(a, b)| a.to_bits() == b.to_bits()));
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
            let output = crate::legacy::conn(
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

    // The production path at the frozen sample cells. Validated against the canonical reference
    // (`legacy::frozen::print_multilevel`), whose values equal the frozen ones exactly there.
    #[test]
    fn multilevel_values() {
        let cells = [(0, 1), (3, 5), (6, 8), (12, 13)];
        for (mode, beri, pa_to_pa, expected) in [
            (WindowMode::Circular, false, false, [0.035036724, 0.060928639, 0.057472181, 0.023413222]),
            (WindowMode::Circular, false, true, [0.031013189, 0.051940750, 0.050136760, 0.017623197]),
            (WindowMode::Circular, true, false, [0.287769228, 0.330879152, 0.336351663, 0.270322204]),
            (WindowMode::Circular, true, true, [0.287769228, 0.330879152, 0.336351663, 0.270322204]),
            (WindowMode::Square, false, false, [0.036365397, 0.062699854, 0.057887629, 0.023465034]),
            (WindowMode::Square, false, true, [0.032288730, 0.053552385, 0.050358951, 0.017685590]),
            (WindowMode::Square, true, false, [0.288273066, 0.334522188, 0.339534730, 0.269480884]),
            (WindowMode::Square, true, true, [0.288273066, 0.334522188, 0.339534730, 0.269480884]),
        ] {
            let f = crate::testkit::Fixture::multilevel(mode, beri, pa_to_pa);
            let output = conn(
                &Some(f.cond.clone()),
                f.trav.as_ref(),
                f.trans.as_deref(),
                &f.transforms,
                &f.mask,
                &f.levels,
                &f.lambdas,
                f.is_geo,
                f.max_cost,
                f.window_size,
                f.outer_window,
                f.offsets,
                f.mode,
                f.pa_to_pa,
            )
            .unwrap();
            for (cell, expected) in cells.iter().zip(expected) {
                let actual = output[*cell];
                assert!(
                    (actual - expected).abs() <= 1.0e-6,
                    "{mode:?}, beri={beri}, pa_to_pa={pa_to_pa}, cell={cell:?}: {actual} != {expected}"
                );
            }
        }
    }
}
