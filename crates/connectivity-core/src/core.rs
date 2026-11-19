use anyhow::{ensure, Result};
use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView2, Axis, CowArray, Ix2};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;
// local modules
use crate::affine;
use crate::builder::{Context, Scratch};
use crate::climate::Climate;
use crate::crossing::{self, Base};
#[cfg(test)]
use crate::gate::GateStats;
use crate::gate::Gate;
use crate::strict::Strict;
#[cfg(test)]
use crate::strict::StrictStats;
use crate::{Crossing, CrossingMode};
use crate::overview;
use crate::tables::{Tables, STATUSES};
use crate::template::{Templates, CACHE_BUDGET};
use crate::{Config, Inputs};
use affine::Affine;
use overview::Resampling;

/// Call-level resource statistics.
#[derive(Debug, Default, Clone)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct Stats {
    pub scratch: Vec<usize>, // retained bytes of each scratch instance
    pub cache_bytes: usize,  // retained shared cache: templates and distance tables
    pub cache_peak: usize,   // largest cache heap while building, temporaries included
    pub cache_fixed: usize,  // per-level descriptors, allocated whatever the budget
    pub table_bytes: usize,  // distance tables alone
    pub climate_bytes: usize, // BERI climate held by the call: coarse means and any layout copies
    pub cached: Vec<usize>,  // template phases cached per level
    pub linked: Vec<bool>,   // levels with cached links
    pub edges: [u64; STATUSES], // edges by distance-table status (hits, then fallback reasons)
    pub setup_s: f64,        // overviews, templates, contexts and scratch
    pub process_s: f64,      // focal-cell loop
    #[cfg(test)]
    pub gate: Option<GateStats>, // crossing mode 1 diagnostics
    #[cfg(test)]
    pub strict: Option<StrictStats>, // crossing mode 2 diagnostics
}

/// Core connectivity function (`crate::connectivity`)
/// # Returns
/// A 2D array of connectivity values for each cell at the native resolution.
pub fn conn(inputs: &Inputs, config: &Config) -> Result<Array2<f32>> {
    conn_stats(inputs, config).map(|(out, _)| out)
}

/// `conn` that also reports scratch memory and setup/processing time.
pub fn conn_stats(inputs: &Inputs, config: &Config) -> Result<(Array2<f32>, Stats)> {
    let start = Instant::now();
    let ctx = match &config.crossing {
        None => prepare(inputs, config)?,
        Some(cross) => prepare_cross(inputs, config, cross, true)?,
    };
    run(inputs, ctx, start)
}

/// `conn_stats` under a NoData crossing mode; `relax` selects relaxed contact in mode 2 (the
/// rule; tests compare it with plain contact).
#[cfg(test)]
pub fn conn_cross(inputs: &Inputs, config: &Config, cross: &Crossing, relax: bool) -> Result<(Array2<f32>, Stats)> {
    let start = Instant::now();
    let ctx = prepare_cross(inputs, config, cross, relax)?;
    run(inputs, ctx, start)
}

/// `prepare` under a NoData crossing mode. NoData in gaps up to `cross.width` km wide is
/// crossed at path weight `cross.cost` and carries no habitat area. Mode 1 admits only the focal
/// cell's group of touching land and crossable NoData; mode 2 splits coarse cells into pieces of
/// touching traversable cells, linked only where their base cells touch.
pub fn prepare_cross<'a>(inputs: &Inputs<'a>, config: &Config<'a>, cross: &Crossing, relax: bool) -> Result<Context<'a>> {
    validate(inputs, config)?;
    check_crossing(cross, &inputs.transforms[&1], config.is_geo)?;
    let cond = inputs.condition;
    let mut sorted: Vec<i32> = config.levels.iter().map(|&l| l as i32).collect();
    sorted.sort_unstable();
    sorted.dedup();
    let base_tr = inputs.transforms[&1];
    let (dy, dx) = crossing::cell_km(&base_tr, cond.nrows(), config.is_geo);
    let cells = crossing::crossable(&cond.mapv(f32::is_finite), cross.width as f64, dy, &dx)?;
    // Traversal values are those of land: a traversal input with finite values on NoData is
    // confined to land (copied), so they never make NoData passable or enter coarse means;
    // otherwise it is borrowed, and without one condition serves, as without a crossing mode.
    let traversal = inputs.traversal;
    let stray = |t: &Array2<f32>| t.indexed_iter().any(|(ij, v)| v.is_finite() && !cond[ij].is_finite());
    let confined = traversal.filter(|t| stray(t)).map(|t| {
        Array2::from_shape_fn(cond.dim(), |ij| if cond[ij].is_finite() { t[ij] } else { f32::NAN })
    });
    let trav = confined.as_ref().or(traversal);
    // Crossable NoData costs `cross.cost` in place of a traversal value: its share among each
    // cell's cells with a path weight (1 on crossable NoData, 0 on land with a traversal value,
    // NaN elsewhere) mixes the two weights (`crossing::mixed_weight`). Without crossable cells
    // no share is built and the weights are those of land, as without a crossing mode.
    let share = cells.iter().any(|&c| c).then(|| {
        Array2::from_shape_fn(cond.dim(), |ij| {
            if cells[ij] {
                1.0
            } else if cond[ij].is_finite() && trav.map_or(cond[ij], |t| t[ij]).is_finite() {
                0.0
            } else {
                f32::NAN
            }
        })
    });
    let areas = config.is_geo.then(|| overview::row_areas(&base_tr, cond.nrows()));
    let mask = config.pa_to_pa.then_some(inputs.mask);
    let trans = inputs.transgrids.unwrap_or_default();
    let base = Base::new(&cells, cond, trav.unwrap_or(cond), &sorted, config.offsets, areas, mask, trans);
    drop(cells);
    let (gate, strict) = match cross.mode {
        CrossingMode::Landmass => (Some(Gate::build(&base)), None),
        CrossingMode::Strict => {
            let mut st = Strict::build(&base)?;
            st.relax = relax;
            (None, Some(st))
        }
    };
    drop(base);
    // The context owns a confined traversal; it borrows a caller's
    let trav = match confined {
        Some(owned) => Some(CowArray::from(owned)),
        None => traversal.map(|t| CowArray::from(t.view())),
    };
    let mut ctx = prepare_with(inputs, config, true, trav, share)?;
    ctx.cost = cross.cost;
    ctx.gate = gate;
    ctx.strict = strict;
    Ok(ctx)
}

fn run(inputs: &Inputs, ctx: Context, start: Instant) -> Result<(Array2<f32>, Stats)> {
    let (cond_base, mask_array) = (inputs.condition, inputs.mask);

    // One scratch per pool thread, sized up front; a row borrows its thread's scratch, so
    // focal cells only allocate if a window outgrows the initial capacity.
    let threads = rayon::current_num_threads();
    let scratch: Vec<Mutex<Scratch>> =
        (0..threads).map(|_| Mutex::new(Scratch::new(&ctx))).collect();
    let setup_s = start.elapsed().as_secs_f64();

    // Initialize output with NaNs so masked cells keep the previous output semantics.
    let ncols = cond_base.ncols();
    let mut outarray = Array2::<f32>::from_elem(cond_base.dim(), f32::NAN);
    // A path-cost overflow on any worker stops the loop and is returned
    outarray
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row_result)| -> Result<()> {
            let slot = rayon::current_thread_index().unwrap_or(0) % threads;
            // A panicking cell poisons its scratch, but every cell resets what it uses
            let mut scratch = scratch[slot].lock().unwrap_or_else(PoisonError::into_inner);
            for j in 0..ncols {
                // Skip masked areas
                if mask_array[[i, j]] {
                    continue;
                }
                row_result[j] = scratch.eval(&ctx, i, j)?;
            }
            Ok(())
        })?;

    let stats = Stats {
        scratch: scratch
            .iter()
            .map(|s| s.lock().unwrap_or_else(PoisonError::into_inner).bytes())
            .collect(),
        cache_bytes: ctx.templates.bytes + ctx.tables.bytes,
        cache_peak: ctx.templates.peak + ctx.tables.bytes,
        cache_fixed: ctx.templates.fixed + ctx.tables.fixed,
        table_bytes: ctx.tables.bytes,
        climate_bytes: ctx.climate.as_ref().map_or(0, |c| c.bytes),
        cached: ctx.templates.cached.to_vec(),
        linked: ctx.templates.linked.to_vec(),
        edges: scratch.iter().fold([0; STATUSES], |mut acc, s| {
            let s = s.lock().unwrap_or_else(PoisonError::into_inner);
            acc.iter_mut().zip(s.counts).for_each(|(a, c)| *a += c);
            acc
        }),
        setup_s,
        process_s: start.elapsed().as_secs_f64() - setup_s,
        #[cfg(test)]
        gate: ctx.gate.as_ref().map(|g| g.stats.clone()),
        #[cfg(test)]
        strict: ctx.strict.as_ref().map(|s| s.stats.clone()),
    };
    Ok((outarray, stats))
}

/// Shapes, band counts, levels, transforms and window sizes of one call, checked before any
/// work so invalid inputs fail with an error rather than a panic or a wrong result.
pub fn validate(inputs: &Inputs, config: &Config) -> Result<()> {
    let shape = inputs.condition.dim();
    let mask = inputs.mask.dim();
    ensure!(mask == shape, "Dimension mismatch for mask: {mask:?} vs {shape:?}");
    if let Some(traversal) = inputs.traversal {
        let dim = traversal.dim();
        ensure!(dim == shape, "Dimension mismatch for key 1: {dim:?} vs {shape:?}");
    }
    // Every transgrid has condition's rows and columns and the first one's bands
    let grids = inputs.transgrids.unwrap_or_default();
    if let Some(first) = grids.first() {
        let expected = (shape.0, shape.1, first.dim().2);
        ensure!(expected.2 > 0, "Transgrids must have at least one band");
        for (k, grid) in grids.iter().enumerate() {
            let dim = grid.dim();
            ensure!(dim == expected, "Dimension mismatch for transgrid {k}: {dim:?} vs {expected:?}");
        }
    }

    for &level in config.levels {
        ensure!(level.is_power_of_two(), "Level {level} is not a power of 2");
    }
    let mut levels = config.levels.to_vec();
    levels.sort_unstable();
    levels.dedup();
    ensure!(levels.first() == Some(&1), "Base condition level not found!");
    ensure!(
        levels.windows(2).all(|pair| pair[1] == pair[0] * 2),
        "levels must form a continuous power-of-two sequence"
    );
    let transforms = inputs.transforms;
    ensure!(transforms.contains_key(&1), "Missing base-level transform (key=1).");
    for &level in &levels {
        ensure!(transforms.contains_key(&(level as i32)), "Missing level in Affine set.");
    }

    // Outer windows below window_size use window_size
    let (window, outer) = (config.window_size, config.outer_window);
    ensure!(window > 0, "Window size must be positive");
    ensure!(window % 2 == 1, "Window size must be odd-numbered");
    ensure!(outer < window || outer % 2 == 1, "Outer window size must be odd-numbered");
    if let Some(cross) = &config.crossing {
        check_crossing(cross, &transforms[&1], config.is_geo)?;
    }
    Ok(())
}

/// Crossing settings, and a base grid on which gap widths can be measured along rows and
/// columns (`crossing::cell_km`): finite cell sizes, perpendicular rows and columns, and north-up
/// on geographic grids.
fn check_crossing(cross: &Crossing, base: &Affine, is_geo: bool) -> Result<()> {
    ensure!(
        cross.width.is_finite() && cross.width >= 0.0,
        "Crossing width must be a finite number of kilometres >= 0, got {}",
        cross.width
    );
    ensure!(
        cross.cost.is_finite() && cross.cost >= 1.0,
        "Crossing cost must be a finite number >= 1, got {}",
        cross.cost
    );
    let Affine { x_scale: a, x_skew: b, y_skew: d, y_scale: e, .. } = *base;
    let (width, height) = (a.hypot(d), b.hypot(e));
    ensure!(
        width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0,
        "NoData crossing needs finite, non-zero cell sizes in the base transform"
    );
    if b != 0.0 || d != 0.0 {
        ensure!(!is_geo, "NoData crossing needs a north-up geographic grid (transform terms b and d of 0), got b = {b}, d = {d}");
        ensure!(
            (a * b + d * e).abs() <= 1.0e-6 * width * height,
            "NoData crossing needs perpendicular rows and columns, but the base transform is sheared: (a, b, d, e) = ({a}, {b}, {d}, {e})"
        );
    }
    Ok(())
}

/// Overviews and settings for one call without a crossing mode (`Config::crossing` `None`, the
/// legacy NoData handling), indexed by ascending level.
pub fn prepare<'a>(inputs: &Inputs<'a>, config: &Config<'a>) -> Result<Context<'a>> {
    prepare_with(inputs, config, false, inputs.traversal.map(|t| CowArray::from(t.view())), None)
}

/// `prepare`; with `land_climate`, coarse cells' transgrid means use their land (finite
/// condition) cells only, as the crossing modes' parts do, so a cell holding one part gets that
/// part's climate and land that cannot be reached never changes it. `trav` is the traversal used
/// for path weights (a crossing mode can confine one to land), replacing `inputs.traversal`, and
/// `share` the base level of crossable NoData's share of path weights (crossing modes).
///
/// Ownership: the base level of condition, traversal and transgrids is borrowed from the caller
/// (a transgrid that is not in standard layout is copied once into it, so each cell's bands are
/// contiguous; see `climate`); the context owns only what it derives: coarse levels, weights, the
/// protected fraction, a confined traversal and the crossing share.
fn prepare_with<'a>(
    inputs: &Inputs<'a>,
    config: &Config<'a>,
    land_climate: bool,
    trav: Option<CowArray<'a, f32, Ix2>>,
    share: Option<Array2<f32>>,
) -> Result<Context<'a>> {
    validate(inputs, config)?;
    let Inputs {
        condition: cond_base,
        mask: mask_array,
        traversal: _,
        transgrids: trans_arrays,
        transforms: transform_map,
    } = *inputs;
    let Config {
        levels: _,
        lambdas,
        is_geo,
        max_cost,
        window_size,
        outer_window,
        offsets,
        window_mode,
        pa_to_pa,
        crossing: _,
    } = *config;
    let (nrows, ncols) = (cond_base.shape()[0], cond_base.shape()[1]);

    // Ascending distinct levels; every per-level vector below is indexed like them
    let mut sorted_levels: Vec<i32> = config.levels.iter().map(|&l| l as i32).collect();
    sorted_levels.sort_unstable();
    sorted_levels.dedup();
    let coarse = &sorted_levels[1..]; // `validate` requires level 1
    let overviews = |base: ArrayView2<f32>, method: Resampling, areas: Option<&[f32]>| -> Vec<Array2<f32>> {
        coarse.iter().map(|&l| overview::coarse_level(base, l as usize, offsets, method, areas)).collect()
    };

    // Condition: the caller's base level, then averages
    let cond: Vec<CowArray<f32, Ix2>> = std::iter::once(CowArray::from(cond_base.view()))
        .chain(overviews(cond_base.view(), Resampling::Average, None).into_iter().map(CowArray::from))
        .collect();
    // Habitat area weights: valid-cell counts, or cos-latitude areas on geographic grids
    let weight_method = if is_geo { Resampling::Area } else { Resampling::Count };
    let areas = overview::area_rows(weight_method, transform_map.get(&1), nrows)?;
    let weight: Vec<Array2<f32>> = std::iter::once(overview::base_weights(cond_base.view(), weight_method, areas.as_deref()))
        .chain(overviews(cond_base.view(), weight_method, areas.as_deref()))
        .collect();

    // PARC target-gating: build a PA-membership overview from the mask. Each cell holds the
    // fraction of its condition-valid sub-cells that fall on a protected area. The base
    // indicator is NaN where condition is NaN so the average shares the cell-weights valid
    // set, making `num_cells * pa` equal the protected area at every level. When disabled,
    // no gating is applied and the output is bit-identical to the original PARC path.
    let pa: Option<Vec<Array2<f32>>> = pa_to_pa.then(|| {
        let pa_indicator = Array2::<f32>::from_shape_fn((nrows, ncols), |(i, j)| {
            if cond_base[[i, j]].is_nan() {
                f32::NAN
            } else if !mask_array[[i, j]] {
                1.0
            } else {
                0.0
            }
        });
        let rest = overviews(pa_indicator.view(), Resampling::Average, None);
        std::iter::once(pa_indicator).chain(rest).collect()
    });

    // Optional traversal overview (resistance-derived, averaged like condition). Built
    // only when a resistance raster is supplied; when None the graph weight falls back to
    // condition, so the output is bit-identical to a resistance-free run.
    let trav: Option<Vec<CowArray<f32, Ix2>>> = trav.map(|base| {
        let rest = overviews(base.view(), Resampling::Average, None);
        std::iter::once(base).chain(rest.into_iter().map(CowArray::from)).collect()
    });
    // Crossing modes: crossable NoData's share of each cell's path weights, averaged as traversal
    let cross: Option<Vec<Array2<f32>>> = share.map(|base| {
        let rest = overviews(base.view(), Resampling::Average, None);
        std::iter::once(base).chain(rest).collect()
    });

    // BERI climate: every scenario's base level (borrowed in standard layout) and band means
    let dims: Vec<(usize, usize)> = cond.iter().map(|a| a.dim()).collect();
    let climate = match trans_arrays {
        Some(arrays) if !arrays.is_empty() => {
            Some(Climate::new(arrays, &sorted_levels, &dims, offsets, land_climate.then_some(cond_base))?)
        }
        _ => None,
    };
    let fringe: Vec<bool> = (0..sorted_levels.len())
        .map(|l| sorted_levels.get(l + 1) == Some(&(sorted_levels[l] * 2)))
        .collect();
    let transforms: Vec<Option<Affine>> =
        sorted_levels.iter().map(|l| transform_map.get(l).copied()).collect();

    // If outer_win is lower than win_size, use the regular win_size
    let outer = outer_window.max(window_size);

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
    // Distance tables first, then window templates, within one shared cache budget that bounds
    // the cache's heap while it is built and afterwards
    let reserved = Templates::fixed(sorted_levels.len());
    let budget = CACHE_BUDGET.saturating_sub(reserved);
    let tables = Tables::new(&sorted_levels, &transforms, &dims, offsets, &fringe, is_geo, budget);
    let templates = Templates::new(
        &sorted_levels,
        &effective,
        window_size,
        window_mode,
        mask_array,
        offsets,
        CACHE_BUDGET.saturating_sub(tables.bytes),
        None,
    );

    Ok(Context {
        levels: sorted_levels,
        cond,
        weight,
        pa,
        trav,
        cross,
        climate,
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
        cost: 1.0,
        pa_to_pa,
        lambdas,
        gate: None,
        strict: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::WindowMode;
    use ndarray::Array3;
    use std::collections::HashMap;

    // Repeated runs must give bit-identical output.
    #[test]
    fn repeats_exactly() {
        let cond = Array2::from_shape_fn((32, 32), |(i, j)| {
            ((i * 7 + j * 13) % 10 + 1) as f32 / 10.0
        });
        let mask = Array2::from_elem((32, 32), false);
        let levels = vec![1, 2, 4];
        let transforms: HashMap<i32, Affine> = levels
            .iter()
            .map(|&l| {
                let s = 1000.0 * l as f64;
                (l as i32, Affine::from((s, 0.0, 0.0, 0.0, -s, 32_000.0)))
            })
            .collect();
        let inputs = Inputs { condition: &cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
        let config = Config {
            levels: &levels, lambdas: &[2.0, 20.0], is_geo: false, max_cost: 2.0, window_size: 3,
            outer_window: 5, offsets: (0, 0), window_mode: WindowMode::Circular, pa_to_pa: false, crossing: None,
        };
        let run = || conn(&inputs, &config).unwrap();

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

    // Transgrids in any memory layout give the same bits: standard layout (borrowed), band-major
    // and Fortran order (each copied once into standard layout), without and with crossing modes
    #[test]
    fn transgrid_layouts() {
        use crate::{Crossing, CrossingMode};
        let modes = [None, Some(CrossingMode::Strict), Some(CrossingMode::Landmass)];
        for (mode, crossing) in [WindowMode::Circular, WindowMode::Square].into_iter().flat_map(|m| modes.map(|c| (m, c))) {
            let mut f = crate::testkit::Fixture::multilevel(mode, true, false);
            f.crossing = crossing.map(|mode| Crossing { mode, width: 1.5, cost: 3.5 });
            let expected = conn(&f.inputs(), &f.config()).unwrap();
            let grids = f.trans.clone().unwrap();
            let band_major: Vec<Array3<f32>> = grids
                .iter()
                .map(|g| g.view().permuted_axes([2, 0, 1]).as_standard_layout().into_owned().permuted_axes([1, 2, 0]))
                .collect();
            let fortran: Vec<Array3<f32>> =
                grids.iter().map(|g| g.view().reversed_axes().as_standard_layout().into_owned().reversed_axes()).collect();
            assert!(!band_major[0].is_standard_layout() && !fortran[0].is_standard_layout());
            for layout in [band_major, fortran] {
                f.trans = Some(layout);
                let out = conn(&f.inputs(), &f.config()).unwrap();
                assert!(crate::testkit::same_bits(&out, &expected), "{mode:?} {crossing:?}");
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
            let output = conn(&f.inputs(), &f.config()).unwrap();
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
