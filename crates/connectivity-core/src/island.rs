//! Island artefact and NoData crossing modes: diagnostics and regression checks.
//!
//! The `proto_*` and island tests print the tables of docs/island_bridging_2026-11-12.md and
//! docs/nodata_crossing_modes.md; they are ignored by default and meant for release builds:
//! `cargo test --release -p connectivity-core --lib island -- --ignored --nocapture
//! --test-threads=1` (figure data needs `ISLAND_OUT=<dir>`). The `check_*` tests at the end are
//! small, fast assertions run with the normal test suite.
use crate::affine::level_transforms;
use crate::builder::Scratch;
use crate::core::{conn, prepare};
use crate::graph::NONE;
use crate::window::WindowMode;
use crate::{Config, Inputs};
use ndarray::Array2;
use std::fmt::Write as _;
use std::path::PathBuf;

const RES: f64 = 250.0; // metres
const LAMBDAS: [f32; 3] = [2.0, 20.0, 200.0];
const MULTI: [usize; 5] = [1, 2, 4, 8, 16];

fn out_dir() -> PathBuf {
    let dir = PathBuf::from(std::env::var("ISLAND_OUT").expect("set ISLAND_OUT"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

struct Run<'a> {
    levels: &'a [usize],
    window: i32,
    outer: i32,
}

const DEFAULT: Run = Run { levels: &MULTI, window: 5, outer: 9 };
// Base resolution only, same reach as 16 x 9 = 144 base cells
const BASE: Run = Run { levels: &[1], window: 5, outer: 145 };

fn config<'a>(run: &'a Run) -> Config<'a> {
    Config {
        levels: run.levels,
        lambdas: &LAMBDAS,
        is_geo: false,
        max_cost: 2.0,
        window_size: run.window,
        outer_window: run.outer,
        offsets: (0, 0),
        window_mode: WindowMode::Circular,
        pa_to_pa: false,
        crossing: None,
    }
}

/// Connectedness of the cells where `focal` is true.
fn run(cond: &Array2<f32>, focal: &Array2<bool>, run: &Run) -> Array2<f32> {
    let mask = Array2::from_shape_fn(cond.dim(), |ij| !focal[ij] || cond[ij].is_nan());
    let rows = cond.nrows() as f64;
    let transforms = level_transforms((RES, 0.0, 0.0, 0.0, -RES, rows * RES), run.levels, (0, 0));
    let inputs = Inputs { condition: cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
    conn(&inputs, &config(run)).unwrap()
}

/// Every window node of focal cell (i, j) with whether the search reached it.
fn probe(cond: &Array2<f32>, i: usize, j: usize, r: &Run) -> (String, f32) {
    let mask = cond.mapv(f32::is_nan);
    let rows = cond.nrows() as f64;
    let transforms = level_transforms((RES, 0.0, 0.0, 0.0, -RES, rows * RES), r.levels, (0, 0));
    let inputs = Inputs { condition: cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
    let cfg = config(r);
    let ctx = prepare(&inputs, &cfg).unwrap();
    let mut s = Scratch::new(&ctx);
    let value = s.eval(&ctx, i, j).unwrap();
    let n = s.nodes.len();
    let mut adj = vec![f32::NAN; n];
    let mut geo = vec![f32::NAN; n];
    let mut reached = vec![false; n];
    for k in 0..s.router.paths.len() {
        let v = s.router.paths.node[k] as usize;
        reached[v] = true;
        adj[v] = s.router.paths.adj[k];
        geo[v] = s.router.paths.geo[k];
    }
    if s.source != NONE {
        reached[s.source as usize] = true;
    }
    let mut text = String::from("level,row0,col0,size,frac,cond,count,reached,adj,geo\n");
    for u in 0..n {
        let level = ctx.levels[s.nodes.level[u] as usize];
        let (li, lj) = s.nodes.cell[u];
        writeln!(
            text,
            "{},{},{},{},{},{},{},{},{},{}",
            level,
            li * level,
            lj * level,
            level,
            s.nodes.frac[u],
            s.nodes.cond[u],
            s.nodes.count[u],
            reached[u] as u8,
            adj[u],
            geo[u]
        )
        .unwrap();
    }
    (text, value)
}

fn save(name: &str, a: &Array2<f32>) {
    let mut text = String::new();
    for row in a.rows() {
        let line: Vec<String> = row.iter().map(|v| format!("{v}")).collect();
        text.push_str(&line.join(","));
        text.push('\n');
    }
    std::fs::write(out_dir().join(name), text).unwrap();
}

fn fill<T: Copy>(a: &mut Array2<T>, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>, v: T) {
    for i in rows {
        for j in cols.clone() {
            a[[i, j]] = v;
        }
    }
}

/// Mainland in the north, the focal island A, and a stepping-stone island B to the north-east.
/// Water (NaN) gaps: A-B 6 cells, B-mainland 8 cells, A-mainland 50 cells.
/// `shift` moves everything by (rows, cols) to change the alignment with the overview grid.
fn scene(shift: (usize, usize)) -> (Array2<f32>, Array2<bool>) {
    let (h, w) = (352, 352);
    let mut cond = Array2::from_elem((h, w), f32::NAN);
    let mut island = Array2::from_elem((h, w), false);
    let (dr, dc) = shift;
    fill(&mut cond, 0..100 + dr, 0..w, 0.5); // mainland
    fill(&mut cond, 150 + dr..160 + dr, 140 + dc..178 + dc, 0.9); // island A, 10 x 38
    fill(&mut island, 150 + dr..160 + dr, 140 + dc..178 + dc, true);
    // Island B: a diagonal strip 6 cells wide from the north-east of A towards the mainland
    for k in 0..36 {
        let (r, c) = (143 + dr - k, 180 + dc + k);
        for t in 0..6 {
            cond[[r, c + t]] = 0.9;
        }
    }
    (cond, island)
}

fn alone(cond: &Array2<f32>, island: &Array2<bool>) -> Array2<f32> {
    Array2::from_shape_fn(cond.dim(), |ij| if island[ij] { cond[ij] } else { f32::NAN })
}

#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn island_scene() {
    let (cond, island) = scene((0, 0));
    let land = cond.mapv(|v| v.is_finite());
    save("cond.csv", &cond);
    // Multi-resolution on every land cell, then island A alone, then base resolution references
    save("multi_full.csv", &run(&cond, &land, &DEFAULT));
    save("multi_alone.csv", &run(&alone(&cond, &island), &island, &DEFAULT));
    save("base_full.csv", &run(&cond, &island, &BASE));
    save("base_alone.csv", &run(&alone(&cond, &island), &island, &BASE));
    for (name, i, j) in [("west", 155, 146), ("centre", 155, 160), ("east", 155, 172)] {
        let (text, value) = probe(&cond, i, j, &DEFAULT);
        std::fs::write(out_dir().join(format!("nodes_{name}.csv")), text).unwrap();
        let (_, lone) = probe(&alone(&cond, &island), i, j, &DEFAULT);
        println!("{name} ({i},{j}): full {value:.4}, island alone {lone:.4}");
    }
}

/// Island values against the alignment of the scene with the overview grid.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn island_shift() {
    println!("shift_r,shift_c,mean_full,mean_alone,max_abs_diff,cells_changed,cells");
    for dr in [0usize, 4, 8, 12] {
        for dc in [0usize, 3, 6, 9, 12, 15] {
            let (cond, island) = scene((dr, dc));
            let full = run(&cond, &island, &DEFAULT);
            let lone = run(&alone(&cond, &island), &island, &DEFAULT);
            let (mut sf, mut sl, mut worst, mut changed, mut n) = (0.0f64, 0.0f64, 0.0f32, 0, 0);
            for (ij, &is) in island.indexed_iter() {
                if is {
                    sf += full[ij] as f64;
                    sl += lone[ij] as f64;
                    let d = (full[ij] - lone[ij]).abs();
                    worst = worst.max(d);
                    changed += (d > 1.0e-4) as usize;
                    n += 1;
                }
            }
            println!("{dr},{dc},{:.4},{:.4},{worst:.4},{changed},{n}", sf / n as f64, sl / n as f64);
        }
    }
}

/// Two bodies only: a mainland and a 10 x 38 island separated by `gap` rows of water.
/// Reports how many island cells change when the mainland is present.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn island_gap() {
    println!("gap,levels,cells_changed,cells,max_abs_diff,mean_full,mean_alone");
    let all: [&[usize]; 3] = [&[1, 2, 4, 8, 16], &[1, 2, 4, 8, 16, 32], &[1]];
    for levels in all {
        for gap in [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96] {
            let (h, w) = (400, 352);
            let mut cond = Array2::from_elem((h, w), f32::NAN);
            let mut island = Array2::from_elem((h, w), false);
            fill(&mut cond, 0..100, 0..w, 0.5);
            fill(&mut cond, 100 + gap..110 + gap, 150..188, 0.9);
            fill(&mut island, 100 + gap..110 + gap, 150..188, true);
            let r = if levels.len() == 1 {
                Run { levels, window: 5, outer: 145 }
            } else {
                Run { levels, window: 5, outer: 9 }
            };
            let full = run(&cond, &island, &r);
            let lone = run(&alone(&cond, &island), &island, &r);
            let (mut sf, mut sl, mut worst, mut changed, mut n) = (0.0f64, 0.0f64, 0.0f32, 0, 0);
            for (ij, &is) in island.indexed_iter() {
                if is {
                    sf += full[ij] as f64;
                    sl += lone[ij] as f64;
                    let d = (full[ij] - lone[ij]).abs();
                    worst = worst.max(d);
                    changed += (d > 1.0e-4) as usize;
                    n += 1;
                }
            }
            println!(
                "{gap},{:?},{changed},{n},{worst:.4},{:.4},{:.4}",
                levels.last().unwrap(),
                sf / n as f64,
                sl / n as f64
            );
        }
    }
}

fn compare(name: &str, cond: &Array2<f32>, focal: &Array2<bool>) {
    let multi = run(cond, focal, &DEFAULT);
    let base = run(cond, focal, &BASE);
    let (mut worst, mut sum, mut n) = (0.0f32, 0.0f64, 0);
    for (ij, &is) in focal.indexed_iter() {
        if is {
            let d = (multi[ij] - base[ij]).abs();
            worst = worst.max(d);
            sum += d as f64;
            n += 1;
        }
    }
    println!("{name}: cells {n}, mean |multi - base| {:.4}, max {:.4}", sum / n as f64, worst);
}

/// Genuine connection: the island is joined to the mainland by a causeway one cell wide.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn island_causeway() {
    for (name, diagonal) in [("causeway straight", false), ("causeway diagonal", true)] {
        let (h, w) = (352, 352);
        let mut cond = Array2::from_elem((h, w), f32::NAN);
        let mut island = Array2::from_elem((h, w), false);
        fill(&mut cond, 0..100, 0..w, 0.5);
        fill(&mut cond, 130..140, 150..188, 0.9);
        fill(&mut island, 130..140, 150..188, true);
        for k in 0..30 {
            // Diagonal steps touch only at corners
            let c = if diagonal { 139 + k } else { 169 };
            cond[[100 + k, c]] = 0.9;
        }
        compare(name, &cond, &island);
        let lone = run(&alone(&cond, &island), &island, &DEFAULT);
        let multi = run(&cond, &island, &DEFAULT);
        let changed = island.indexed_iter().filter(|(ij, &is)| is && (multi[*ij] - lone[*ij]).abs() > 1.0e-4).count();
        println!("  island cells that see the mainland through the causeway: {changed} of 380");
    }
}

/// One landmass with an inlet `gap` cells wide: the banks are joined only around its head.
/// Left bank 0.9, right bank 0.5, so a shortcut across the water shows in left-bank scores.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn island_inlet() {
    for gap in [2usize, 8, 16] {
        let (h, w) = (400, 352);
        let mut cond = Array2::from_elem((h, w), f32::NAN);
        fill(&mut cond, 0..h, 0..170, 0.9);
        fill(&mut cond, 0..h, 170 + gap..w, 0.5);
        fill(&mut cond, 0..40, 170..170 + gap, 0.7);
        for row in [60usize, 120, 200, 300] {
            let mut focal = Array2::from_elem((h, w), false);
            fill(&mut focal, row..row + 1, 150..170, true);
            compare(&format!("inlet {gap} wide, {} cells below its head", row - 40), &cond, &focal);
        }
    }
}

/// Data for the two explanatory figures.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn fig_data() {
    // Island scene: multi-resolution and base-resolution results on both islands
    let (cond, _) = scene((0, 0));
    let islands = Array2::from_shape_fn(cond.dim(), |(i, j)| i >= 100 && cond[[i, j]].is_finite());
    save("fig_island_cond.csv", &cond);
    save("fig_island_multi.csv", &run(&cond, &islands, &DEFAULT));
    save("fig_island_base.csv", &run(&cond, &islands, &BASE));

    // Inlet scene: one landmass, an inlet 8 cells wide whose banks join only around its head
    let (h, w, gap) = (480, 352, 8);
    let mut cond = Array2::from_elem((h, w), f32::NAN);
    fill(&mut cond, 0..h, 0..170, 0.9);
    fill(&mut cond, 0..h, 170 + gap..w, 0.5);
    fill(&mut cond, 0..40, 170..170 + gap, 0.7);
    let mut focal = Array2::from_elem((h, w), false);
    fill(&mut focal, 10..400, 138..210, true);
    save("fig_inlet_cond.csv", &cond);
    save("fig_inlet_multi.csv", &run(&cond, &focal, &DEFAULT));
    save("fig_inlet_base.csv", &run(&cond, &focal, &BASE));
    for (name, i, j) in [("near", 100, 165), ("far", 230, 165)] {
        let (text, value) = probe(&cond, i, j, &DEFAULT);
        std::fs::write(out_dir().join(format!("fig_inlet_nodes_{name}.csv")), text).unwrap();
        println!("inlet {name} ({i},{j}): multi {value:.4}");
    }
}

/// Inlet check inside the method: north-bank cells whose window cannot contain the head of the
/// inlet, scored with the far bank present and with it removed.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn inlet_far_bank() {
    let (h, w, gap) = (480, 352, 8);
    let mut both = Array2::from_elem((h, w), f32::NAN);
    fill(&mut both, 0..h, 0..170, 0.9);
    fill(&mut both, 0..h, 170 + gap..w, 0.5);
    fill(&mut both, 0..40, 170..170 + gap, 0.7);
    let mut near_only = both.clone();
    fill(&mut near_only, 0..h, 170 + gap..w, f32::NAN);
    let mut far_only = both.clone();
    fill(&mut far_only, 0..h, 0..170, f32::NAN);
    for (name, col, lone) in [("north bank (0.9)", 165usize, &near_only), ("south bank (0.5)", 182usize, &far_only)] {
        let mut focal = Array2::from_elem((h, w), false);
        fill(&mut focal, 200..330, col..col + 1, true); // 160 to 290 cells from the head
        let (m_both, m_lone) = (run(&both, &focal, &DEFAULT), run(lone, &focal, &DEFAULT));
        let (b_both, b_lone) = (run(&both, &focal, &BASE), run(lone, &focal, &BASE));
        let stat = |a: &Array2<f32>| {
            let v: Vec<f32> = focal.indexed_iter().filter(|(_, &f)| f).map(|(ij, _)| a[ij]).collect();
            (v.iter().cloned().fold(f32::INFINITY, f32::min), v.iter().cloned().fold(0.0, f32::max))
        };
        let maxdiff = |a: &Array2<f32>, b: &Array2<f32>| {
            focal.indexed_iter().filter(|(_, &f)| f).map(|(ij, _)| (a[ij] - b[ij]).abs()).fold(0.0, f32::max)
        };
        println!("{name}: multi both {:?}, multi other bank removed {:?}, max change {:.4}", stat(&m_both), stat(&m_lone), maxdiff(&m_both, &m_lone));
        println!("{name}: level-1 both {:?}, level-1 other bank removed {:?}, max change {:.6}", stat(&b_both), stat(&b_lone), maxdiff(&b_both, &b_lone));
    }
}

/// Figure data inside the method: each island, and each bank of the inlet, scored with the
/// land it cannot reach removed.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn fig_data_alone() {
    let (cond, island_a) = scene((0, 0));
    let island_b = Array2::from_shape_fn(cond.dim(), |(i, j)| i >= 100 && cond[[i, j]].is_finite() && !island_a[[i, j]]);
    let a = run(&alone(&cond, &island_a), &island_a, &DEFAULT);
    let b = run(&alone(&cond, &island_b), &island_b, &DEFAULT);
    save("fig_island_alone.csv", &Array2::from_shape_fn(cond.dim(), |ij| if island_a[ij] { a[ij] } else { b[ij] }));

    let (h, w, gap) = (480, 352, 8);
    let mut both = Array2::from_elem((h, w), f32::NAN);
    fill(&mut both, 0..h, 0..170, 0.9);
    fill(&mut both, 0..h, 170 + gap..w, 0.5);
    fill(&mut both, 0..40, 170..170 + gap, 0.7);
    let mut north_only = both.clone();
    fill(&mut north_only, 0..h, 170 + gap..w, f32::NAN);
    let mut south_only = both.clone();
    fill(&mut south_only, 0..h, 0..170, f32::NAN);
    let mut north = Array2::from_elem((h, w), false);
    fill(&mut north, 10..400, 138..170, true);
    let mut south = Array2::from_elem((h, w), false);
    fill(&mut south, 10..400, 178..210, true);
    let n = run(&north_only, &north, &DEFAULT);
    let s = run(&south_only, &south, &DEFAULT);
    save("fig_inlet_alone.csv", &Array2::from_shape_fn((h, w), |ij| if north[ij] { n[ij] } else { s[ij] }));
}

/// Score of focal cell (i, j) plus the area weight of window cells centred at or below row
/// `south`, split into reached and admitted.
fn probe_south(cond: &Array2<f32>, i: usize, j: usize, south: usize) -> (f32, f32, f32) {
    let mask = cond.mapv(f32::is_nan);
    let rows = cond.nrows() as f64;
    let transforms = level_transforms((RES, 0.0, 0.0, 0.0, -RES, rows * RES), DEFAULT.levels, (0, 0));
    let inputs = Inputs { condition: cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
    let cfg = config(&DEFAULT);
    let ctx = prepare(&inputs, &cfg).unwrap();
    let mut s = Scratch::new(&ctx);
    let value = s.eval(&ctx, i, j).unwrap();
    let mut reached = vec![false; s.nodes.len()];
    for k in 0..s.router.paths.len() {
        reached[s.router.paths.node[k] as usize] = true;
    }
    let (mut hit, mut all) = (0.0f32, 0.0f32);
    for (u, &seen) in reached.iter().enumerate() {
        let level = ctx.levels[s.nodes.level[u] as usize];
        let centre = s.nodes.cell[u].0 * level + level / 2;
        if centre as usize >= south {
            all += s.nodes.count[u];
            if seen {
                hit += s.nodes.count[u];
            }
        }
    }
    (value, hit, all)
}

/// Today's treatment of a straight NoData river between a north bank (0.9) and a south bank (0.5).
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn river_profile() {
    let (h, w) = (480, 352);
    let scene = |width: usize, top: usize| {
        let mut both = Array2::from_elem((h, w), f32::NAN);
        fill(&mut both, 0..top, 0..w, 0.9);
        fill(&mut both, top + width..h, 0..w, 0.5);
        let mut north = both.clone();
        fill(&mut north, top + width..h, 0..w, f32::NAN);
        (both, north)
    };
    for width in [1usize, 3, 6] {
        let (both, north) = scene(width, 240);
        println!("river {width} cells wide: distance, score, score with far bank removed, far-bank area reached");
        for d in [1usize, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50, 60, 80, 100, 130, 150] {
            let (v, hit, all) = probe_south(&both, 240 - d, 176, 240 + width);
            let (lone, _, _) = probe_south(&north, 240 - d, 176, 240 + width);
            println!("  {d:>4}  {v:.3}  {lone:.3}  {:>5.1}% of {all:.0}", 100.0 * hit / all.max(1.0));
        }
    }
    println!("river 3 cells wide, raster shifted by 0 to 15 rows: range of the score at a fixed distance");
    for d in [3usize, 8, 16, 30] {
        let (mut lo, mut hi) = (f32::INFINITY, 0.0f32);
        for shift in 0..16 {
            let (both, _) = scene(3, 240 + shift);
            let (v, _, _) = probe_south(&both, 240 + shift - d, 176, 243 + shift);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        println!("  distance {d:>3}: {lo:.3} to {hi:.3}");
    }
}

// ------------------------------------------------------------------ prototype: gap-width mask
use crate::core::{conn_cross, prepare_cross, Stats};
use crate::{Crossing, CrossingMode};

/// Crossing settings for a width in base cells of the inputs' grid.
fn crossing(inputs: &Inputs, mode: CrossingMode, cells: f64, traversal: f32) -> Crossing {
    let width = (cells * inputs.transforms[&1].x_scale.abs() / 1000.0) as f32;
    Crossing { mode, width, traversal }
}

fn conn_gated(inputs: &Inputs, cfg: &Config, cells: f64, t: f32) -> anyhow::Result<(Array2<f32>, Stats)> {
    conn_cross(inputs, cfg, &crossing(inputs, CrossingMode::Landmass, cells, t), false)
}

fn conn_strict(inputs: &Inputs, cfg: &Config, cells: f64, t: f32) -> anyhow::Result<(Array2<f32>, Stats)> {
    conn_cross(inputs, cfg, &crossing(inputs, CrossingMode::Strict, cells, t), false)
}

fn conn_relaxed(inputs: &Inputs, cfg: &Config, cells: f64, t: f32) -> anyhow::Result<(Array2<f32>, Stats)> {
    conn_cross(inputs, cfg, &crossing(inputs, CrossingMode::Strict, cells, t), true)
}

fn prepare_gated<'a>(inputs: &Inputs<'a>, cfg: &Config<'a>, cells: f64, t: f32) -> anyhow::Result<Context<'a>> {
    prepare_cross(inputs, cfg, &crossing(inputs, CrossingMode::Landmass, cells, t), false)
}

fn prepare_strict<'a>(inputs: &Inputs<'a>, cfg: &Config<'a>, cells: f64, t: f32) -> anyhow::Result<Context<'a>> {
    prepare_cross(inputs, cfg, &crossing(inputs, CrossingMode::Strict, cells, t), false)
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Current,
    Width(f64),  // mode 1: landmass (gap-width traversal mask)
    Strict(f64), // mode 2: strict gap width (pieces linked only where they touch)
    Relaxed(f64), // mode 2 with diagonal contact through a cell sharing the corner
}

impl Mode {
    fn name(&self) -> String {
        match self {
            Mode::Current => "current".into(),
            Mode::Width(w) => format!("landmass W={w}"),
            Mode::Strict(w) => format!("strict W={w}"),
            Mode::Relaxed(w) => format!("relaxed W={w}"),
        }
    }
}

const MODES: [Mode; 7] = [
    Mode::Current,
    Mode::Width(0.0),
    Mode::Width(4.0),
    Mode::Width(8.0),
    Mode::Strict(0.0),
    Mode::Strict(4.0),
    Mode::Strict(8.0),
];
/// Traversal value of crossable NoData: 0 costs as much as fully degraded land.
const CROSS: f32 = 0.0;

fn score(cond: &Array2<f32>, focal: &Array2<bool>, mode: Mode) -> Array2<f32> {
    let mask = Array2::from_shape_fn(cond.dim(), |ij| !focal[ij] || cond[ij].is_nan());
    let rows = cond.nrows() as f64;
    let transforms = level_transforms((RES, 0.0, 0.0, 0.0, -RES, rows * RES), DEFAULT.levels, (0, 0));
    let inputs = Inputs { condition: cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
    let cfg = config(&DEFAULT);
    match mode {
        Mode::Current => conn(&inputs, &cfg).unwrap(),
        Mode::Width(w) => conn_gated(&inputs, &cfg, w, CROSS).unwrap().0,
        Mode::Strict(w) => conn_strict(&inputs, &cfg, w, CROSS).unwrap().0,
        Mode::Relaxed(w) => conn_relaxed(&inputs, &cfg, w, CROSS).unwrap().0,
    }
}

struct Summary {
    changed: usize, // cells whose score differs between the two runs
    cells: usize,
    largest: f32,  // largest change
    mean: f64,     // mean of the first run
    step: f32,     // largest difference between adjacent focal cells in the first run
}

fn summary(with: &Array2<f32>, without: &Array2<f32>, focal: &Array2<bool>) -> Summary {
    let (h, w) = focal.dim();
    let mut s = Summary { changed: 0, cells: 0, largest: 0.0, mean: 0.0, step: 0.0 };
    for ((i, j), &f) in focal.indexed_iter() {
        if !f {
            continue;
        }
        let d = (with[[i, j]] - without[[i, j]]).abs();
        s.changed += (d > 1.0e-4) as usize;
        s.cells += 1;
        s.largest = s.largest.max(d);
        s.mean += with[[i, j]] as f64;
        for (ni, nj) in [(i + 1, j), (i, j + 1)] {
            if ni < h && nj < w && focal[[ni, nj]] {
                s.step = s.step.max((with[[i, j]] - with[[ni, nj]]).abs());
            }
        }
    }
    s.mean /= s.cells.max(1) as f64;
    s
}

fn two_body(gap: usize) -> (Array2<f32>, Array2<bool>) {
    let (h, w) = (400, 352);
    let mut cond = Array2::from_elem((h, w), f32::NAN);
    let mut island = Array2::from_elem((h, w), false);
    fill(&mut cond, 0..100, 0..w, 0.5);
    fill(&mut cond, 100 + gap..110 + gap, 150..188, 0.9);
    fill(&mut island, 100 + gap..110 + gap, 150..188, true);
    (cond, island)
}

/// Island and mainland: is the island affected by the mainland, and is the result free of steps?
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_island_gap() {
    println!("mode, gap, cells changed by the mainland, largest change, island mean, largest step");
    for mode in MODES {
        for gap in [1usize, 2, 3, 4, 5, 6, 8, 9, 12, 16, 24, 32] {
            let (cond, island) = two_body(gap);
            let s = summary(&score(&cond, &island, mode), &score(&alone(&cond, &island), &island, mode), &island);
            println!("{:>13}, {gap:>2}, {:>3}/{}, {:.3}, {:.3}, {:.3}", mode.name(), s.changed, s.cells, s.largest, s.mean, s.step);
        }
    }
}

/// Mainland, lower island A and second island B (gaps of 6 and 8 cells).
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_three_body() {
    let (cond, a) = scene((0, 0));
    let b = Array2::from_shape_fn(cond.dim(), |(i, j)| i >= 100 && cond[[i, j]].is_finite() && !a[[i, j]]);
    println!("mode, island, cells changed by other land, largest change, mean, largest step");
    for mode in [
        Mode::Current,
        Mode::Width(0.0),
        Mode::Width(4.0),
        Mode::Width(8.0),
        Mode::Width(10.0),
        Mode::Width(12.0),
        Mode::Width(16.0),
        Mode::Strict(0.0),
        Mode::Strict(4.0),
        Mode::Strict(8.0),
        Mode::Strict(10.0),
        Mode::Strict(12.0),
        Mode::Strict(16.0),
    ] {
        for (name, isl) in [("A", &a), ("B", &b)] {
            let s = summary(&score(&cond, isl, mode), &score(&alone(&cond, isl), isl, mode), isl);
            println!("{:>13}, {name}, {:>3}/{}, {:.3}, {:.3}, {:.3}", mode.name(), s.changed, s.cells, s.largest, s.mean, s.step);
        }
    }
}

fn river(width: usize, top: usize) -> (Array2<f32>, Array2<f32>) {
    let (h, w) = (480, 352);
    let mut both = Array2::from_elem((h, w), f32::NAN);
    fill(&mut both, 0..top, 0..w, 0.9);
    fill(&mut both, top + width..h, 0..w, 0.5);
    let mut north = both.clone();
    fill(&mut north, top + width..h, 0..w, f32::NAN);
    (both, north)
}

/// Straight river with no land link between the banks: north-bank scores against distance.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_river() {
    let dists = [1usize, 2, 4, 6, 10, 16, 24, 40, 80, 130];
    println!("mode, river width, north-bank score at distances {dists:?}, then the same with the south bank removed");
    for width in [1usize, 3, 6, 10] {
        let (both, north) = river(width, 240);
        let mut focal = Array2::from_elem(both.dim(), false);
        for d in dists {
            focal[[240 - d, 176]] = true;
        }
        for mode in MODES {
            let (with, lone) = (score(&both, &focal, mode), score(&north, &focal, mode));
            let row = |a: &Array2<f32>| dists.iter().map(|&d| format!("{:.3}", a[[240 - d, 176]])).collect::<Vec<_>>().join(" ");
            println!("{:>13}, {width:>2}, {} | {}", mode.name(), row(&with), row(&lone));
        }
    }
}

fn inlet() -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (h, w, gap) = (480, 352, 8);
    let mut both = Array2::from_elem((h, w), f32::NAN);
    fill(&mut both, 0..h, 0..170, 0.9);
    fill(&mut both, 0..h, 170 + gap..w, 0.5);
    fill(&mut both, 0..40, 170..170 + gap, 0.7);
    let mut near = both.clone();
    fill(&mut near, 0..h, 170 + gap..w, f32::NAN);
    let mut far = both.clone();
    fill(&mut far, 0..h, 0..170, f32::NAN);
    (both, near, far)
}

/// Inlet 8 cells wide inside one landmass: cells beyond the window's reach of the head.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_inlet() {
    let (both, near, far) = inlet();
    println!("mode, bank, score with both banks, score with the opposite bank removed, largest change");
    for mode in MODES {
        for (name, col, lone) in [("0.9 bank", 165usize, &near), ("0.5 bank", 182usize, &far)] {
            let mut focal = Array2::from_elem(both.dim(), false);
            fill(&mut focal, 200..330, col..col + 1, true);
            let (with, without) = (score(&both, &focal, mode), score(lone, &focal, mode));
            let s = summary(&with, &without, &focal);
            let mean_without = focal.indexed_iter().filter(|(_, &f)| f).map(|(ij, _)| without[ij] as f64).sum::<f64>() / s.cells as f64;
            println!("{:>13}, {name}, {:.3}, {:.3}, {:.3}", mode.name(), s.mean, mean_without, s.largest);
        }
    }
}

/// A land link one cell wide must keep the island connected in every mode.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_causeway() {
    println!("mode, causeway, island cells that see the mainland, island mean");
    for (name, diagonal) in [("straight", false), ("diagonal", true)] {
        let (h, w) = (352, 352);
        let mut cond = Array2::from_elem((h, w), f32::NAN);
        let mut island = Array2::from_elem((h, w), false);
        fill(&mut cond, 0..100, 0..w, 0.5);
        fill(&mut cond, 130..140, 150..188, 0.9);
        fill(&mut island, 130..140, 150..188, true);
        for k in 0..30 {
            let c = if diagonal { 139 + k } else { 169 };
            cond[[100 + k, c]] = 0.9;
        }
        for mode in MODES {
            let s = summary(&score(&cond, &island, mode), &score(&alone(&cond, &island), &island, mode), &island);
            println!("{:>13}, {name}, {:>3}/{}, {:.3}", mode.name(), s.changed, s.cells, s.mean);
        }
    }
}

/// Sensitivity to how the raster aligns with the overview grid.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_alignment() {
    println!("three bodies, island A, shifted by 0 to 15 columns and rows 0, 5, 11: cells changed (min-max), mean (min-max)");
    for mode in MODES {
        let (mut lo, mut hi, mut mlo, mut mhi) = (usize::MAX, 0usize, f64::INFINITY, 0.0f64);
        for dr in [0usize, 5, 11] {
            for dc in 0..16 {
                let (cond, a) = scene((dr, dc));
                let s = summary(&score(&cond, &a, mode), &score(&alone(&cond, &a), &a, mode), &a);
                (lo, hi) = (lo.min(s.changed), hi.max(s.changed));
                (mlo, mhi) = (mlo.min(s.mean), mhi.max(s.mean));
            }
        }
        println!("{:>13}: {lo}-{hi} of 380, {mlo:.4}-{mhi:.4}", mode.name());
    }
    println!("island and mainland shifted by 0 to 15 rows: cells changed (min-max), mean (min-max)");
    for gap in [3usize, 6, 12] {
        for mode in MODES {
            let (mut lo, mut hi, mut mlo, mut mhi) = (usize::MAX, 0usize, f64::INFINITY, 0.0f64);
            for shift in 0..16usize {
                let (h, w) = (416, 352);
                let mut cond = Array2::from_elem((h, w), f32::NAN);
                let mut island = Array2::from_elem((h, w), false);
                fill(&mut cond, 0..100 + shift, 0..w, 0.5);
                fill(&mut cond, 100 + shift + gap..110 + shift + gap, 150..188, 0.9);
                fill(&mut island, 100 + shift + gap..110 + shift + gap, 150..188, true);
                let s = summary(&score(&cond, &island, mode), &score(&alone(&cond, &island), &island, mode), &island);
                (lo, hi) = (lo.min(s.changed), hi.max(s.changed));
                (mlo, mhi) = (mlo.min(s.mean), mhi.max(s.mean));
            }
            println!("  gap {gap:>2}, {:>13}: {lo}-{hi} of 380, {mlo:.4}-{mhi:.4}", mode.name());
        }
    }
    println!("river 3 cells wide shifted by 0 to 15 rows: score range at 3, 8 and 30 cells from the water");
    for mode in MODES {
        let mut text = String::new();
        for d in [3usize, 8, 30] {
            let (mut lo, mut hi) = (f32::INFINITY, 0.0f32);
            for shift in 0..16 {
                let (both, _) = river(3, 240 + shift);
                let mut focal = Array2::from_elem(both.dim(), false);
                focal[[240 + shift - d, 176]] = true;
                let v = score(&both, &focal, mode)[[240 + shift - d, 176]];
                (lo, hi) = (lo.min(v), hi.max(v));
            }
            write!(text, "  d={d}: {lo:.3}-{hi:.3}").unwrap();
        }
        println!("{:>13}:{text}", mode.name());
    }
}

// ------------------------------------------------------------------ prototype: graph checks
use crate::builder::Context;

/// Build and route the graph of focal cell (i, j), then hand it to `check`.
fn inspect<T>(cond: &Array2<f32>, i: usize, j: usize, mode: Mode, check: impl FnOnce(&Context, &Scratch, f32) -> T) -> T {
    let mask = cond.mapv(f32::is_nan);
    let rows = cond.nrows() as f64;
    let transforms = level_transforms((RES, 0.0, 0.0, 0.0, -RES, rows * RES), DEFAULT.levels, (0, 0));
    let inputs = Inputs { condition: cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
    let cfg = config(&DEFAULT);
    let ctx = match mode {
        Mode::Current => prepare(&inputs, &cfg).unwrap(),
        Mode::Width(w) => prepare_gated(&inputs, &cfg, w, CROSS).unwrap(),
        Mode::Strict(w) => prepare_strict(&inputs, &cfg, w, CROSS).unwrap(),
        Mode::Relaxed(w) => relaxed_ctx(&inputs, &cfg, w),
    };
    let mut s = Scratch::new(&ctx);
    let value = s.eval(&ctx, i, j).unwrap();
    check(&ctx, &s, value)
}

type Rect = (usize, usize, usize, usize);

fn rect(ctx: &Context, s: &Scratch, u: usize, dim: (usize, usize)) -> Rect {
    let level = ctx.levels[s.nodes.level[u] as usize] as usize;
    let (i, j) = (s.nodes.cell[u].0 as usize, s.nodes.cell[u].1 as usize);
    (i * level, ((i + 1) * level).min(dim.0), j * level, ((j + 1) * level).min(dim.1))
}

/// Whether a usable base cell of `a` is the same as or 8-adjacent to a usable base cell of `b`.
fn touches(a: Rect, b: Rect, usable: &impl Fn(usize, usize) -> bool) -> bool {
    for r in a.0..a.1 {
        for c in a.2..a.3 {
            if !usable(r, c) {
                continue;
            }
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    let (nr, nc) = (r as i64 + dr, c as i64 + dc);
                    if nr >= b.0 as i64 && nr < b.1 as i64 && nc >= b.2 as i64 && nc < b.3 as i64 && usable(nr as usize, nc as usize) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Number of 8-connected pieces formed by the usable base cells of `a`.
fn pieces(a: Rect, usable: &impl Fn(usize, usize) -> bool) -> usize {
    let (h, w) = (a.1 - a.0, a.3 - a.2);
    let mut seen = vec![false; h * w];
    let mut count = 0;
    for start in 0..h * w {
        let (r, c) = (a.0 + start / w, a.2 + start % w);
        if seen[start] || !usable(r, c) {
            continue;
        }
        count += 1;
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(k) = stack.pop() {
            let (r, c) = ((k / w) as i64, (k % w) as i64);
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    let (nr, nc) = (r + dr, c + dc);
                    if nr < 0 || nc < 0 || nr >= h as i64 || nc >= w as i64 {
                        continue;
                    }
                    let n = nr as usize * w + nc as usize;
                    if !seen[n] && usable(a.0 + nr as usize, a.2 + nc as usize) {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
        }
    }
    count
}

/// Does the routed graph respect the mask? Counts, for one focal cell, the shortest-path edges
/// whose two cells have no touching usable base cells, and reached cells whose usable base cells
/// form several pieces. Usable means the focal cell's group under a mask, any land otherwise.
fn relaxed_ctx<'a>(inputs: &Inputs<'a>, cfg: &Config<'a>, w: f64) -> Context<'a> {
    let mut ctx = prepare_strict(inputs, cfg, w, CROSS).unwrap();
    ctx.strict.as_mut().unwrap().relax = true;
    ctx
}

fn respect(cond: &Array2<f32>, i: usize, j: usize, mode: Mode) -> String {
    if let Mode::Strict(_) | Mode::Relaxed(_) = mode {
        return respect_strict(cond, i, j, mode);
    }
    inspect(cond, i, j, mode, |ctx, s, value| {
        let dim = cond.dim();
        let group = ctx.gate.as_ref().map(|g| g.group(0, [i, j]));
        let usable = |r: usize, c: usize| match (&ctx.gate, group) {
            (Some(g), Some(id)) => g.group(0, [r, c]) == id,
            _ => cond[[r, c]].is_finite(),
        };
        let mut foreign = 0; // admitted cells with no usable base cell
        for u in 0..s.nodes.len() {
            let a = rect(ctx, s, u, dim);
            if !(a.0..a.1).any(|r| (a.2..a.3).any(|c| usable(r, c))) {
                foreign += 1;
            }
        }
        let tree = &s.router.altered;
        let (mut edges, mut jumps, mut split, mut area) = (0, 0, 0, 0.0f64);
        for &v in tree.order.iter().skip(1) {
            let (v, p) = (v as usize, tree.parent[v as usize] as usize);
            let (a, b) = (rect(ctx, s, p, dim), rect(ctx, s, v, dim));
            edges += 1;
            jumps += !touches(a, b, &usable) as usize;
            split += (pieces(b, &usable) > 1) as usize;
            area += s.nodes.count[v] as f64;
        }
        format!(
            "{:>13}: score {value:.3}, cells in window {}, without usable land {foreign}, reached {}, habitat area reached {area:.0}, \
             path edges {edges}, edges across NoData {jumps}, reached cells in several pieces {split}",
            mode.name(),
            s.nodes.len(),
            tree.order.len().saturating_sub(1)
        )
    })
}

#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_respects_mask() {
    let (three, _) = scene((0, 0));
    let (both, _, _) = inlet();
    let (stream, _) = river(3, 240);
    let wide = [
        Mode::Current,
        Mode::Width(0.0),
        Mode::Width(4.0),
        Mode::Width(8.0),
        Mode::Width(12.0),
        Mode::Strict(0.0),
        Mode::Strict(4.0),
        Mode::Strict(8.0),
        Mode::Strict(12.0),
        Mode::Relaxed(0.0),
    ];
    for (name, cond, i, j, modes) in [
        ("three bodies, western cell of island A", &three, 155usize, 146usize, &wide[..]),
        ("three bodies, eastern cell of island A", &three, 155, 172, &wide[..]),
        ("inlet 8 wide, 0.9 bank, 190 cells from the head", &both, 230, 165, &MODES[..]),
        ("river 3 wide, 3 cells from the water", &stream, 237, 176, &MODES[..]),
    ] {
        println!("{name}");
        for &mode in modes {
            println!("{}", respect(cond, i, j, mode));
        }
    }
}

/// Crossing cost and habitat area of a permitted crossing.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_cost_and_area() {
    for width in [1usize, 3] {
        let (stream, _) = river(width, 240);
        // The same scene with the river as fully degraded land (condition 0)
        let filled = stream.mapv(|v| if v.is_nan() { 0.0 } else { v });
        for ((i, j), mode) in [(240 - 3, 176usize), (240 - 40, 176)]
            .into_iter()
            .flat_map(|ij| [(ij, Mode::Width(4.0)), (ij, Mode::Strict(4.0))])
        {
            let masked = inspect(&stream, i, j, mode, |_, s, v| {
                (s.router.paths.node.clone(), s.router.paths.adj.clone(), s.router.paths.geo.clone(), s.nodes.count.clone(), v)
            });
            let degraded = inspect(&filled, i, j, Mode::Current, |_, s, v| {
                (s.router.paths.node.clone(), s.router.paths.adj.clone(), s.router.paths.geo.clone(), s.nodes.count.clone(), v)
            });
            let nodata = inspect(&stream, i, j, Mode::Current, |_, s, v| (s.nodes.count.iter().map(|&c| c as f64).sum::<f64>(), s.nodes.len(), v));
            // Same destinations in the same order, with identical adjusted and intact path lengths
            assert_eq!(masked.0, degraded.0, "destinations");
            assert!(masked.1.iter().zip(&degraded.1).all(|(a, b)| a.to_bits() == b.to_bits()), "adjusted path lengths");
            assert!(masked.2.iter().zip(&degraded.2).all(|(a, b)| a.to_bits() == b.to_bits()), "intact path lengths");
            // Habitat area: equal to the land-only total of the current engine, below the filled one
            let area = |c: &[f32]| c.iter().map(|&x| x as f64).sum::<f64>();
            let transit = masked.3.iter().filter(|&&c| c == 0.0).count();
            assert!((area(&masked.3) - nodata.0).abs() <= 1.0e-6 * nodata.0, "habitat area");
            println!(
                "{}: river {width} wide, focal {} cells from the water: {} destinations with identical path lengths to the degraded-land scene; \
                 habitat area {:.1} (land only {:.1}, degraded-land scene {:.1}); zero-area cells {transit}; \
                 scores: mask {:.4}, river as degraded land {:.4}, current {:.4}",
                mode.name(),
                240 - i,
                masked.0.len(),
                area(&masked.3),
                nodata.0,
                area(&degraded.3),
                masked.4,
                degraded.4,
                nodata.2
            );
        }
    }
}

/// Scenes without NoData gaps must be unchanged by the mask.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_identity() {
    let mut rng = crate::testkit::Rng::new(11);
    let full = Array2::from_shape_fn((160, 170), |_| (0.05 + 0.95 * rng.unit()) as f32);
    let mut coast = full.clone();
    for ((i, j), v) in coast.indexed_iter_mut() {
        if (i as f64 - 80.0).powi(2) + (j as f64 - 85.0).powi(2) > 60.0f64.powi(2) {
            *v = f32::NAN; // a round island in open water
        }
    }
    for (name, cond) in [("no NoData", &full), ("round island in open water", &coast)] {
        let focal = cond.mapv(|v| v.is_finite());
        let current = score(cond, &focal, Mode::Current);
        for mode in MODES[1..].iter().copied().chain([Mode::Relaxed(0.0)]) {
            let masked = score(cond, &focal, mode);
            let same = current.iter().zip(&masked).all(|(a, b)| a.to_bits() == b.to_bits());
            let worst = current.iter().zip(&masked).filter(|(a, _)| a.is_finite()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            println!("{name}, {}: bit-identical to current {same}, largest difference {worst:.2e}", mode.name());
        }
    }
}

/// One real-raster run. Environment: PROTO_RASTER (raw little-endian f32), PROTO_DIM "rows,cols",
/// PROTO_RES (metres), PROTO_MODE ("current" or a gap width in cells), PROTO_TAG (output name).
#[test]
#[ignore = "real raster; see docs/nodata_crossing_modes.md"]
fn proto_bench() {
    if std::env::var("PROTO_RASTER").is_err() {
        println!("skipped: set PROTO_RASTER, PROTO_DIM and PROTO_RES (see docs/nodata_crossing_modes.md)");
        return;
    }
    use crate::core::conn_stats;
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
    let dim: Vec<usize> = var("PROTO_DIM").split(',').map(|v| v.parse().unwrap()).collect();
    let res: f64 = var("PROTO_RES").parse().unwrap();
    let mode = match var("PROTO_MODE").as_str() {
        "current" => Mode::Current,
        w if w.starts_with('s') => Mode::Strict(w[1..].parse().unwrap()),
        w if w.starts_with('r') => Mode::Relaxed(w[1..].parse().unwrap()),
        w => Mode::Width(w.parse().unwrap()),
    };
    let bytes = std::fs::read(var("PROTO_RASTER")).unwrap();
    let data: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    drop(bytes);
    let cond = Array2::from_shape_vec((dim[0], dim[1]), data).unwrap();
    let mask = cond.mapv(f32::is_nan);
    let levels = [1usize, 2, 4, 8, 16, 32];
    let transforms = level_transforms((res, 0.0, 0.0, 0.0, -res, dim[0] as f64 * res), &levels, (0, 0));
    let inputs = Inputs { condition: &cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
    let cfg = Config {
        levels: &levels,
        lambdas: &LAMBDAS,
        is_geo: false,
        max_cost: 2.0,
        window_size: 5,
        outer_window: 9,
        offsets: (0, 0),
        window_mode: WindowMode::Circular,
        pa_to_pa: false,
        crossing: None,
    };
    let (out, stats) = match mode {
        Mode::Current => conn_stats(&inputs, &cfg),
        Mode::Width(w) => conn_gated(&inputs, &cfg, w, CROSS),
        Mode::Strict(w) => conn_strict(&inputs, &cfg, w, CROSS),
        Mode::Relaxed(w) => conn_relaxed(&inputs, &cfg, w, CROSS),
    }
    .unwrap();
    let raw: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out_dir().join(format!("bench_{}_{}.f32", var("PROTO_TAG"), var("PROTO_MODE"))), raw).unwrap();
    println!(
        "BENCH mode={} setup_s={:.2} process_s={:.2} cache_mb={:.1} scratch_mb={:.1} edges={:?} gate={:?} strict={:?}",
        var("PROTO_MODE"),
        stats.setup_s,
        stats.process_s,
        stats.cache_bytes as f64 / 1.0e6,
        stats.scratch.iter().sum::<usize>() as f64 / 1.0e6,
        stats.edges,
        stats.gate,
        stats.strict
    );
}

/// A river whose banks are one landmass: joined by land at the western edge, more than a
/// window's reach from the focal cells. Wider than the threshold, it should not be crossed.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_river_joined() {
    let dists = [1usize, 4, 10, 24, 40, 80];
    println!("mode, river width, north-bank score at distances {dists:?} | with the south bank removed | path edges across NoData at distance 10");
    for width in [3usize, 6, 10] {
        let (mut both, mut north) = river(width, 240);
        for cond in [&mut both, &mut north] {
            fill(cond, 240..240 + width, 0..4, 0.9); // land joining the banks, 300 cells west
        }
        let mut focal = Array2::from_elem(both.dim(), false);
        for d in dists {
            focal[[240 - d, 310]] = true;
        }
        for mode in MODES {
            let (with, lone) = (score(&both, &focal, mode), score(&north, &focal, mode));
            let row = |a: &Array2<f32>| dists.iter().map(|&d| format!("{:.3}", a[[240 - d, 310]])).collect::<Vec<_>>().join(" ");
            let text = respect(&both, 230, 310, mode);
            let jumps = text.split("edges across NoData ").nth(1).unwrap().split(',').next().unwrap().to_string();
            println!("{:>13}, {width:>2}, {} | {} | {jumps}", mode.name(), row(&with), row(&lone));
        }
    }
}

// ------------------------------------------------------------------ prototype: strict mode checks

/// Base cells of the `idx`-th 8-connected piece of `usable` cells inside `a`, pieces numbered in
/// row-major order of their first cells (independent of `strict::label_level`).
fn piece_cells(a: Rect, usable: &impl Fn(usize, usize) -> bool, idx: usize) -> Vec<(usize, usize)> {
    let (h, w) = (a.1 - a.0, a.3 - a.2);
    let mut seen = vec![false; h * w];
    let mut count = 0;
    for start in 0..h * w {
        let (r, c) = (a.0 + start / w, a.2 + start % w);
        if seen[start] || !usable(r, c) {
            continue;
        }
        let mut cells = Vec::new();
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(k) = stack.pop() {
            cells.push((a.0 + k / w, a.2 + k % w));
            let (r, c) = ((k / w) as i64, (k % w) as i64);
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    let (nr, nc) = (r + dr, c + dc);
                    if nr < 0 || nc < 0 || nr >= h as i64 || nc >= w as i64 {
                        continue;
                    }
                    let n = nr as usize * w + nc as usize;
                    if !seen[n] && usable(a.0 + nr as usize, a.2 + nc as usize) {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
        }
        if count == idx {
            return cells;
        }
        count += 1;
    }
    Vec::new()
}

fn sets_touch(a: &[(usize, usize)], b: &[(usize, usize)]) -> bool {
    let set: std::collections::HashSet<(usize, usize)> = b.iter().copied().collect();
    a.iter().any(|&(r, c)| {
        (-1i64..=1).any(|dr| {
            (-1i64..=1).any(|dc| {
                let (nr, nc) = (r as i64 + dr, c as i64 + dc);
                nr >= 0 && nc >= 0 && set.contains(&(nr as usize, nc as usize))
            })
        })
    })
}

/// Traversable base cells under a gap width: land plus NoData in gaps up to `width` cells.
fn domain(cond: &Array2<f32>, width: f64) -> Array2<bool> {
    let land = cond.mapv(|v| v.is_finite());
    let cross = crate::crossing::crossable(&land, width, 1.0, &vec![1.0; cond.nrows()]).unwrap();
    Array2::from_shape_fn(cond.dim(), |ij| land[ij] || cross[ij])
}

/// Base cells each node stands for: its piece (strict), its group's cells (landmass) or its land.
fn node_cells(ctx: &Context, s: &Scratch, u: usize, cond: &Array2<f32>, mode: Mode, dom: &Array2<bool>, focal: (usize, usize)) -> Vec<(usize, usize)> {
    let a = rect(ctx, s, u, cond.dim());
    match mode {
        Mode::Strict(_) | Mode::Relaxed(_) => piece_cells(a, &|r, c| dom[[r, c]], s.piece[u] as usize),
        Mode::Width(_) => {
            let g = ctx.gate.as_ref().unwrap();
            let id = g.group(0, [focal.0, focal.1]);
            (a.0..a.1).flat_map(|r| (a.2..a.3).map(move |c| (r, c))).filter(|&(r, c)| g.group(0, [r, c]) == id).collect()
        }
        Mode::Current => (a.0..a.1).flat_map(|r| (a.2..a.3).map(move |c| (r, c))).filter(|&(r, c)| cond[[r, c]].is_finite()).collect(),
    }
}

/// `respect` in strict mode, checked piece by piece rather than cell by cell.
fn respect_strict(cond: &Array2<f32>, i: usize, j: usize, mode: Mode) -> String {
    let (Mode::Strict(w) | Mode::Relaxed(w)) = mode else { unreachable!() };
    let dom = domain(cond, w);
    inspect(cond, i, j, mode, |ctx, s, value| {
        let cells = |u: usize| node_cells(ctx, s, u, cond, mode, &dom, (i, j));
        let foreign = (0..s.nodes.len()).filter(|&u| cells(u).is_empty()).count();
        let tree = &s.router.altered;
        let (mut edges, mut jumps, mut area) = (0, 0, 0.0f64);
        for &v in tree.order.iter().skip(1) {
            let (v, p) = (v as usize, tree.parent[v as usize] as usize);
            edges += 1;
            jumps += !sets_touch(&cells(p), &cells(v)) as usize;
            area += s.nodes.count[v] as f64;
        }
        // Every edge of the whole graph, not only the shortest-path tree
        let mut all_jumps = 0;
        for u in 0..s.nodes.len() {
            let cu = cells(u);
            for e in &s.csr.edges[s.csr.offsets[u] as usize..s.csr.offsets[u + 1] as usize] {
                all_jumps += !sets_touch(&cu, &cells(e.to as usize)) as usize;
            }
        }
        format!(
            "{:>13}: score {value:.3}, cells in window {}, without usable land {foreign}, reached {}, habitat area reached {area:.0}, \
             path edges {edges}, edges across NoData {jumps}, graph edges {} of which between pieces that do not touch {all_jumps}",
            mode.name(),
            s.nodes.len(),
            tree.order.len().saturating_sub(1),
            s.csr.edges.len()
        )
    })
}

/// Base-resolution geodesic distance in cells from (i, j) over `dom`, 8-connected.
fn geodesic(dom: &Array2<bool>, i: usize, j: usize) -> Array2<f64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let (h, w) = dom.dim();
    let mut dist = Array2::from_elem((h, w), f64::INFINITY);
    let mut heap = BinaryHeap::new();
    dist[[i, j]] = 0.0;
    // Non-negative f64 values order like their bit patterns
    heap.push((Reverse(0.0f64.to_bits()), i, j));
    while let Some((Reverse(d), r, c)) = heap.pop() {
        let d = f64::from_bits(d);
        if d > dist[[r, c]] {
            continue;
        }
        for dr in -1i64..=1 {
            for dc in -1i64..=1 {
                let (nr, nc) = (r as i64 + dr, c as i64 + dc);
                if (dr, dc) == (0, 0) || nr < 0 || nc < 0 || nr >= h as i64 || nc >= w as i64 {
                    continue;
                }
                let (nr, nc) = (nr as usize, nc as usize);
                if !dom[[nr, nc]] {
                    continue;
                }
                let nd = d + if dr != 0 && dc != 0 { std::f64::consts::SQRT_2 } else { 1.0 };
                if nd < dist[[nr, nc]] {
                    dist[[nr, nc]] = nd;
                    heap.push((Reverse(nd.to_bits()), nr, nc));
                }
            }
        }
    }
    dist
}

/// Shortcut check for one focal cell: compares each reached destination's intact path length in
/// the graph with the base-resolution geodesic to the nearest base cell the node stands for, over
/// the traversable cells the mode permits (land only for the current engine). Destinations with
/// no base-resolution route at all are unintended connections.
fn shortcut(cond: &Array2<f32>, i: usize, j: usize, mode: Mode) -> String {
    let width = match mode {
        Mode::Current => 0.0,
        Mode::Width(w) | Mode::Strict(w) | Mode::Relaxed(w) => w,
    };
    let dom = domain(cond, width);
    let geo = geodesic(&dom, i, j);
    inspect(cond, i, j, mode, |ctx, s, value| {
        let paths = &s.router.paths;
        let (mut unreachable, mut unreachable_area, mut total_area) = (0, 0.0f64, 0.0f64);
        let (mut worst, mut worst_level, mut beyond) = (f64::NEG_INFINITY, 0, 0);
        for k in 0..paths.len() {
            let v = paths.node[k] as usize;
            let area = s.nodes.count[v] as f64;
            total_area += area;
            let g = node_cells(ctx, s, v, cond, mode, &dom, (i, j)).iter().map(|&rc| geo[rc]).fold(f64::INFINITY, f64::min);
            if !g.is_finite() {
                unreachable += 1;
                unreachable_area += area;
                continue;
            }
            let level = ctx.levels[s.nodes.level[v] as usize] as f64;
            let p = paths.geo[k] as f64 / (RES / 1000.0);
            let short = g - p;
            if short > worst {
                (worst, worst_level) = (short, level as usize);
            }
            beyond += (short > 2.0 * level) as usize;
        }
        format!(
            "{:>13}: score {value:.3}, destinations {}, with no base route {unreachable} ({:.1}% of reached habitat), \
             largest shortfall {worst:.1} cells (level {worst_level}), shortfall over 2 x level {beyond}",
            mode.name(),
            paths.len(),
            100.0 * unreachable_area / total_area.max(1.0e-9)
        )
    })
}

/// Graph paths against base-resolution routes, for the three engines.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_shortcut() {
    let (three, _) = scene((0, 0));
    let (inlet8, _, _) = inlet();
    let (stream, _) = river(3, 240);
    let joined = |width: usize| {
        let (mut both, _) = river(width, 240);
        fill(&mut both, 240..240 + width, 0..4, 0.9);
        both
    };
    let cases: Vec<(String, Array2<f32>, usize, usize)> = vec![
        ("three bodies, western cell of island A".into(), three, 155, 146),
        ("inlet 8 wide, 0.9 bank, 190 cells below the head".into(), inlet8.clone(), 230, 165),
        ("inlet 8 wide, 0.9 bank, 60 cells below the head".into(), inlet8.clone(), 100, 165),
        ("inlet 8 wide, 0.9 bank, 20 cells below the head".into(), inlet8, 60, 165),
        ("river 3 wide, banks never joined, 3 cells from the water".into(), stream, 237, 176),
        ("river 3 wide, banks joined 300 cells west, 10 cells from the water".into(), joined(3), 230, 310),
        ("river 6 wide, banks joined 300 cells west, 10 cells from the water".into(), joined(6), 230, 310),
        ("river 10 wide, banks joined 300 cells west, 10 cells from the water".into(), joined(10), 230, 310),
    ];
    for (name, cond, i, j) in &cases {
        println!("{name}");
        for mode in MODES.into_iter().chain([Mode::Relaxed(0.0)]) {
            println!("{}", shortcut(cond, *i, *j, mode));
        }
    }
}

/// Inlet 8 wide: the opposite bank's effect on north-bank cells at several distances below the
/// head. A route around the head inside the window is legitimate; a crossing of the water is not.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_detour() {
    let (both, near, _) = inlet();
    let rows = [42usize, 50, 60, 80, 100, 120, 140, 160, 180, 186, 188, 190, 192, 194, 196, 198, 200, 300];
    println!("north-bank cell 5 cells from the water; rows below the head {:?}", rows.map(|r| r - 40));
    println!("mode, score with both banks | with the opposite bank removed | change");
    for mode in MODES.into_iter().chain([Mode::Relaxed(0.0)]) {
        let mut focal = Array2::from_elem(both.dim(), false);
        for &r in &rows {
            focal[[r, 165]] = true;
        }
        let (with, lone) = (score(&both, &focal, mode), score(&near, &focal, mode));
        let row = |a: &Array2<f32>| rows.iter().map(|&r| format!("{:.3}", a[[r, 165]])).collect::<Vec<_>>().join(" ");
        let change = rows.iter().map(|&r| format!("{:+.3}", with[[r, 165]] - lone[[r, 165]])).collect::<Vec<_>>().join(" ");
        println!("{:>13}, {} | {} | {}", mode.name(), row(&with), row(&lone), change);
    }
}

/// Diagonal one-cell causeway: island cells that do not reach the mainland, and for one of them
/// the causeway cells reached at each level.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_causeway_diag() {
    let (h, w) = (352, 352);
    let mut cond = Array2::from_elem((h, w), f32::NAN);
    let mut island = Array2::from_elem((h, w), false);
    fill(&mut cond, 0..100, 0..w, 0.5);
    fill(&mut cond, 130..140, 150..188, 0.9);
    fill(&mut island, 130..140, 150..188, true);
    for k in 0..30 {
        cond[[100 + k, 139 + k]] = 0.9;
    }
    for mode in [Mode::Current, Mode::Strict(0.0), Mode::Width(0.0)] {
        let lone = score(&alone(&cond, &island), &island, mode);
        let full = score(&cond, &island, mode);
        let cut: Vec<(usize, usize)> = island.indexed_iter().filter(|(ij, &is)| is && (full[*ij] - lone[*ij]).abs() <= 1.0e-4).map(|(ij, _)| ij).collect();
        println!("{}: island cells that do not see the mainland {:?}", mode.name(), cut);
        if let (Mode::Strict(_), Some(&(i, j))) = (mode, cut.first()) {
            let dom = domain(&cond, 0.0);
            inspect(&cond, i, j, mode, |ctx, s, _| {
                let mut reached = vec![false; s.nodes.len()];
                for k in 0..s.router.paths.len() {
                    reached[s.router.paths.node[k] as usize] = true;
                }
                for (u, &seen) in reached.iter().enumerate() {
                    let cells = node_cells(ctx, s, u, &cond, mode, &dom, (i, j));
                    // Causeway cells: rows 100 to 129 on the diagonal
                    if cells.iter().any(|&(r, c)| (100..130).contains(&r) && c == r + 39) {
                        let level = ctx.levels[s.nodes.level[u] as usize];
                        let (a, b) = s.nodes.cell[u];
                        let rows: Vec<usize> = cells.iter().filter(|&&(r, c)| c == r + 39 && (100..130).contains(&r)).map(|&(r, _)| r).collect();
                        println!(
                            "  level {level:>2} cell ({a},{b}) piece {} rim {} frac {:.3} reached {} causeway rows {:?}..{:?}",
                            s.piece[u], s.nodes.rim[u], s.nodes.frac[u], seen, rows.iter().min(), rows.iter().max()
                        );
                    }
                }
            });
        }
    }
}

/// One-cell causeways, straight and diagonal in both orientations, under 64 alignments with the
/// overview grid: island cells (of 380) that do not see the mainland, min-max over alignments.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_causeway_shift() {
    println!("causeway, mode: island cells cut off (min-max over 64 shifts), shifts with any cut");
    for (name, kind) in [("straight", 0), ("diagonal", 1), ("anti-diagonal", 2)] {
        for mode in [Mode::Current, Mode::Width(0.0), Mode::Strict(0.0), Mode::Strict(4.0), Mode::Relaxed(0.0)] {
            let (mut lo, mut hi, mut any) = (usize::MAX, 0, 0);
            for dr in 0..8usize {
                for dc in 0..8usize {
                    let (h, w) = (360, 360);
                    let mut cond = Array2::from_elem((h, w), f32::NAN);
                    let mut island = Array2::from_elem((h, w), false);
                    fill(&mut cond, 0..100 + dr, 0..w, 0.5);
                    fill(&mut cond, 130 + dr..140 + dr, 150 + dc..188 + dc, 0.9);
                    fill(&mut island, 130 + dr..140 + dr, 150 + dc..188 + dc, true);
                    for k in 0..30 {
                        let c = match kind {
                            0 => 169,
                            1 => 139 + k,
                            _ => 198 - k,
                        };
                        cond[[100 + dr + k, c + dc]] = 0.9;
                    }
                    let lone = score(&alone(&cond, &island), &island, mode);
                    let full = score(&cond, &island, mode);
                    let cut = island.indexed_iter().filter(|(ij, &is)| is && (full[*ij] - lone[*ij]).abs() <= 1.0e-4).count();
                    (lo, hi) = (lo.min(cut), hi.max(cut));
                    any += (cut > 0) as usize;
                }
            }
            println!("{name:>13}, {:>13}: {lo}-{hi}, {any} of 64", mode.name());
        }
    }
}

/// Shortcut check on a real raster (PROTO_RASTER, PROTO_DIM, PROTO_RES, PROTO_SAMPLES): focal cells
/// sampled among land cells within 3 cells of NoData. For each mode, every reached destination's
/// intact path length is compared with the base-resolution geodesic, over the cells that mode may
/// cross, inside a square of twice the window's reach around the focal cell.
#[test]
#[ignore = "real raster; see docs/nodata_crossing_modes.md"]
fn proto_real_shortcut() {
    if std::env::var("PROTO_RASTER").is_err() {
        println!("skipped: set PROTO_RASTER, PROTO_DIM and PROTO_RES (see docs/nodata_crossing_modes.md)");
        return;
    }
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
    let dim: Vec<usize> = var("PROTO_DIM").split(',').map(|v| v.parse().unwrap()).collect();
    let res: f64 = var("PROTO_RES").parse().unwrap();
    let samples: usize = var("PROTO_SAMPLES").parse().unwrap();
    let bytes = std::fs::read(var("PROTO_RASTER")).unwrap();
    let data: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let cond = Array2::from_shape_vec((dim[0], dim[1]), data).unwrap();
    let (h, w) = cond.dim();
    let levels = [1usize, 2, 4, 8, 16, 32];
    let reach = 32 * 9;
    let radius = 2 * reach;
    let transforms = level_transforms((res, 0.0, 0.0, 0.0, -res, h as f64 * res), &levels, (0, 0));
    let mask = cond.mapv(f32::is_nan);
    let inputs = Inputs { condition: &cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
    let cfg = Config {
        levels: &levels,
        lambdas: &LAMBDAS,
        is_geo: false,
        max_cost: 2.0,
        window_size: 5,
        outer_window: 9,
        offsets: (0, 0),
        window_mode: WindowMode::Circular,
        pa_to_pa: false,
        crossing: None,
    };
    // Coastal land cells: within 3 cells of NoData, away from the raster edge
    let land = cond.mapv(|v| v.is_finite());
    let near = |i: usize, j: usize| {
        (i.saturating_sub(3)..(i + 4).min(h)).any(|r| (j.saturating_sub(3)..(j + 4).min(w)).any(|c| !land[[r, c]]))
    };
    let mut rng = crate::testkit::Rng::new(2026);
    let mut focal: Vec<(usize, usize)> = match std::env::var("PROTO_FOCAL") {
        Ok(path) => std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| {
                let v: Vec<usize> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
                (v[0], v[1])
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    let listed = focal.len();
    while focal.len() < listed + samples {
        let (i, j) = (rng.below(h), rng.below(w));
        if land[[i, j]] && i > 3 && j > 3 && i + 4 < h && j + 4 < w && near(i, j) {
            focal.push((i, j));
        }
    }
    // Geodesics depend only on the focal cell and the width
    let crop = |i: usize, j: usize| (i.saturating_sub(radius), (i + radius + 1).min(h), j.saturating_sub(radius), (j + radius + 1).min(w));
    let modes = [Mode::Current, Mode::Width(0.0), Mode::Strict(0.0), Mode::Relaxed(0.0)];
    let domains: Vec<(f64, Array2<bool>)> = [0.0, 8.0].iter().map(|&wd| (wd, domain(&cond, wd))).collect();
    println!("{listed} listed focal cells, {samples} sampled within 3 cells of NoData; geodesics inside {}-cell squares", 2 * radius + 1);
    println!("mode: share of reached habitat with no base route nearby (mean over cells), cells with any, cells with a shortfall over 2 x level, largest shortfall (cells, level), mean score");
    for mode in modes {
        let ctx = match mode {
            Mode::Current => prepare(&inputs, &cfg).unwrap(),
            Mode::Width(wd) => prepare_gated(&inputs, &cfg, wd, CROSS).unwrap(),
            Mode::Strict(wd) => prepare_strict(&inputs, &cfg, wd, CROSS).unwrap(),
            Mode::Relaxed(wd) => relaxed_ctx(&inputs, &cfg, wd),
        };
        let width = match mode {
            Mode::Current => 0.0,
            Mode::Width(wd) | Mode::Strict(wd) | Mode::Relaxed(wd) => wd,
        };
        let dom = &domains.iter().find(|(wd, _)| *wd == width).unwrap().1;
        let mut s = Scratch::new(&ctx);
        for (set, cells) in [("listed", &focal[..listed]), ("sampled", &focal[listed..])] {
        if cells.is_empty() {
            continue;
        }
        let (mut share_sum, mut with_any, mut with_short, mut worst, mut worst_level, mut score_sum) = (0.0f64, 0, 0, f64::NEG_INFINITY, 0, 0.0f64);
        for &(i, j) in cells {
            let value = s.eval(&ctx, i, j).unwrap();
            score_sum += value as f64;
            let (r0, r1, c0, c1) = crop(i, j);
            let sub = dom.slice(ndarray::s![r0..r1, c0..c1]).to_owned();
            let geo = geodesic(&sub, i - r0, j - c0);
            let g_at = |r: usize, c: usize| if r >= r0 && r < r1 && c >= c0 && c < c1 { geo[[r - r0, c - c0]] } else { f64::INFINITY };
            let paths = &s.router.paths;
            let (mut lost, mut total, mut short) = (0.0f64, 0.0f64, 0);
            for k in 0..paths.len() {
                let v = paths.node[k] as usize;
                let area = s.nodes.count[v] as f64;
                total += area;
                let g = node_cells(&ctx, &s, v, &cond, mode, dom, (i, j)).iter().map(|&(r, c)| g_at(r, c)).fold(f64::INFINITY, f64::min);
                if !g.is_finite() {
                    lost += area;
                    continue;
                }
                let level = ctx.levels[s.nodes.level[v] as usize] as f64;
                let sf = g - paths.geo[k] as f64 / (res / 1000.0);
                if sf > worst {
                    (worst, worst_level) = (sf, level as usize);
                }
                short += (sf > 2.0 * level) as usize;
            }
            share_sum += lost / total.max(1.0e-9);
            with_any += (lost > 0.0) as usize;
            with_short += (short > 0) as usize;
        }
        let n = cells.len() as f64;
        println!(
            "{:>13}, {set} ({}): {:.4}, {with_any}, {with_short}, {worst:.1} (level {worst_level}), {:.4}",
            mode.name(),
            cells.len(),
            share_sum / n,
            score_sum / n
        );
        }
    }
}

/// Level-1 diagnostic control for `proto_detour`: same reach (outer window 145), no coarse levels.
#[test]
#[ignore = "diagnostic table; run with --release -- --ignored --nocapture"]
fn proto_detour_control() {
    let (both, near, _) = inlet();
    let rows = [42usize, 60, 100, 140, 180, 186, 188, 190, 192, 194, 196, 200];
    let mut focal = Array2::from_elem(both.dim(), false);
    for &r in &rows {
        focal[[r, 165]] = true;
    }
    let (with, lone) = (run(&both, &focal, &BASE), run(&near, &focal, &BASE));
    println!("level-1 control, rows below the head {:?}", rows.map(|r| r - 40));
    println!("change {}", rows.iter().map(|&r| format!("{:+.3}", with[[r, 165]] - lone[[r, 165]])).collect::<Vec<_>>().join(" "));
}

// ------------------------------------------------------------------ regression checks (fast)

/// Levels 1 to 8 (reach 72 cells) keep these scenes small enough for debug builds.
const SMALL: Run = Run { levels: &[1, 2, 4, 8], window: 5, outer: 9 };

fn small(cond: &Array2<f32>, focal: &Array2<bool>, mode: Mode) -> Array2<f32> {
    let mask = Array2::from_shape_fn(cond.dim(), |ij| !focal[ij] || cond[ij].is_nan());
    let rows = cond.nrows() as f64;
    let transforms = level_transforms((RES, 0.0, 0.0, 0.0, -RES, rows * RES), SMALL.levels, (0, 0));
    let inputs = Inputs { condition: cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
    let cfg = config(&SMALL);
    match mode {
        Mode::Current => conn(&inputs, &cfg).unwrap(),
        Mode::Width(w) => conn_gated(&inputs, &cfg, w, CROSS).unwrap().0,
        Mode::Strict(w) => conn_strict(&inputs, &cfg, w, CROSS).unwrap().0,
        Mode::Relaxed(w) => conn_relaxed(&inputs, &cfg, w, CROSS).unwrap().0,
    }
}

/// Focal cells whose scores differ between two runs, out of all focal cells.
fn changed(a: &Array2<f32>, b: &Array2<f32>, focal: &Array2<bool>) -> (usize, usize) {
    let cells: Vec<_> = focal.indexed_iter().filter(|(_, &f)| f).map(|(ij, _)| ij).collect();
    (cells.iter().filter(|&&ij| a[ij].to_bits() != b[ij].to_bits()).count(), cells.len())
}

const CROSSING: [Mode; 7] = [
    Mode::Width(0.0),
    Mode::Width(4.0),
    Mode::Width(8.0),
    Mode::Strict(0.0),
    Mode::Strict(4.0),
    Mode::Strict(8.0),
    Mode::Relaxed(0.0),
];

// Without NoData every mode gives the current engine's output bit for bit
#[test]
fn check_identity() {
    let mut rng = crate::testkit::Rng::new(5);
    let cond = Array2::from_shape_fn((48, 56), |_| (0.05 + 0.95 * rng.unit()) as f32);
    let focal = cond.mapv(|v| v.is_finite());
    let current = small(&cond, &focal, Mode::Current);
    for mode in CROSSING {
        let out = small(&cond, &focal, mode);
        assert!(crate::testkit::same_bits(&out, &current), "{}", mode.name());
    }
}

/// Mainland (rows 0 to 39, condition 0.5) and a 5 x 30 island (0.9) `gap` rows below it.
fn islet(gap: usize, mainland: bool) -> (Array2<f32>, Array2<bool>) {
    let (h, w) = (120, 112);
    let mut cond = Array2::from_elem((h, w), f32::NAN);
    let mut island = Array2::from_elem((h, w), false);
    if mainland {
        fill(&mut cond, 0..40, 0..w, 0.5);
    }
    fill(&mut cond, 40 + gap..45 + gap, 40..70, 0.9);
    fill(&mut island, 40 + gap..45 + gap, 40..70, true);
    (cond, island)
}

// An island is unaffected by land it may not cross to, and wholly affected by land it may
fn check_island(modes: &[Mode]) {
    for &gap in &[2usize, 6, 12] {
        let (with, island) = islet(gap, true);
        let (alone, _) = islet(gap, false);
        for &mode in modes {
            let (Mode::Width(w) | Mode::Strict(w) | Mode::Relaxed(w)) = mode else { unreachable!() };
            let (n, cells) = changed(&small(&with, &island, mode), &small(&alone, &island, mode), &island);
            let expected = if gap as f64 <= w { cells } else { 0 };
            assert_eq!(n, expected, "{}, gap {gap}", mode.name());
        }
    }
    // The current engine bridges the 2-cell gap from every island cell
    let (with, island) = islet(2, true);
    let (alone, _) = islet(2, false);
    let (n, cells) = changed(&small(&with, &island, Mode::Current), &small(&alone, &island, Mode::Current), &island);
    assert_eq!(n, cells);
}

#[test]
fn check_island_m1() {
    check_island(&CROSSING[..3]);
}

#[test]
fn check_island_m2() {
    check_island(&CROSSING[3..]);
}

/// One landmass with an inlet `width` cells wide, joined only around its head (rows 0 to 19).
/// Returns both banks and the west bank alone.
fn inlet_small(width: usize) -> (Array2<f32>, Array2<f32>) {
    let (h, w) = (200, 120);
    let mut both = Array2::from_elem((h, w), f32::NAN);
    fill(&mut both, 0..h, 0..56, 0.9);
    fill(&mut both, 0..h, 56 + width..w, 0.5);
    fill(&mut both, 0..20, 56..56 + width, 0.7);
    let mut west = both.clone();
    fill(&mut west, 0..h, 56 + width..w, f32::NAN);
    (both, west)
}

// West-bank cells beyond the window's reach of the inlet's head: mode 2 never crosses a 6-cell
// inlet unless W permits it; mode 1 (and today's engine) cross it inside the landmass
#[test]
fn check_inlet() {
    let (both, west) = inlet_small(6);
    let mut focal = Array2::from_elem(both.dim(), false);
    fill(&mut focal, 110..190, 50..51, true); // 90 to 170 cells below the head
    let differs = |mode| changed(&small(&both, &focal, mode), &small(&west, &focal, mode), &focal).0;
    for mode in [Mode::Strict(0.0), Mode::Strict(4.0), Mode::Relaxed(0.0)] {
        assert_eq!(differs(mode), 0, "{}", mode.name());
    }
    for mode in [Mode::Current, Mode::Width(0.0), Mode::Strict(8.0)] {
        assert_eq!(differs(mode), 80, "{}", mode.name());
    }
}

// A one-cell straight causeway keeps the island linked in every mode; a diagonal one in every
// mode except plain strict contact, which can cut it (docs/nodata_crossing_modes.md)
#[test]
fn check_causeway() {
    for diagonal in [false, true] {
        let (h, w) = (120, 112);
        let mut cond = Array2::from_elem((h, w), f32::NAN);
        let mut island = Array2::from_elem((h, w), false);
        fill(&mut cond, 0..40, 0..w, 0.5);
        fill(&mut cond, 60..65, 40..70, 0.9);
        fill(&mut island, 60..65, 40..70, true);
        for k in 0..20 {
            cond[[40 + k, if diagonal { 36 + k } else { 55 }]] = 0.9;
        }
        let lone = Array2::from_shape_fn(cond.dim(), |ij| if island[ij] { cond[ij] } else { f32::NAN });
        for mode in [Mode::Current].into_iter().chain(CROSSING) {
            if diagonal && mode == Mode::Strict(0.0) {
                continue;
            }
            let (n, cells) = changed(&small(&cond, &island, mode), &small(&lone, &island, mode), &island);
            assert_eq!(n, cells, "{}, diagonal {diagonal}", mode.name());
        }
    }
}

// ------------------------------------------------------- crossing modes: tiles, grids, PARC, BERI

const ALL: [(CrossingMode, bool); 3] =
    [(CrossingMode::Landmass, false), (CrossingMode::Strict, false), (CrossingMode::Strict, true)];

fn label(mode: CrossingMode, relax: bool) -> &'static str {
    match (mode, relax) {
        (CrossingMode::Landmass, _) => "landmass",
        (CrossingMode::Strict, false) => "strict",
        (CrossingMode::Strict, true) => "relaxed",
    }
}

/// Output of `f` under a crossing mode, or without one.
fn under(f: &crate::testkit::Fixture, mode: Option<(CrossingMode, bool)>, width: f32) -> Array2<f32> {
    match mode {
        None => conn(&f.inputs(), &f.config()).unwrap(),
        Some((mode, relax)) => {
            let cross = Crossing { mode, width, traversal: 0.0 };
            conn_cross(&f.inputs(), &f.config(), &cross, relax).unwrap().0
        }
    }
}

// Without NoData every mode gives the current output bit for bit, on random tiles with
// offsets, geographic grids, traversal, PARC gating, BERI and square windows
#[test]
fn check_random() {
    for seed in 0..40 {
        let mut f = crate::testkit::Fixture::random(seed);
        f.cond.mapv_inplace(|v| if v.is_nan() { 0.3 } else { v });
        let current = under(&f, None, 0.0);
        for (mode, relax) in ALL {
            for width in [0.0, 5.0] {
                let out = under(&f, Some((mode, relax)), width);
                assert!(crate::testkit::same_bits(&out, &current), "seed {seed}, {}, {width}", label(mode, relax));
            }
        }
    }
}

/// Mainland cut by a river (banks joined below it) and an inlet, two islands 6 and 14 rows apart
/// joined by a diagonal causeway, random condition; 250 x 250 cells of 250 m, levels 1 to 8,
/// windows 3 and 5 (reach 40 cells).
fn tile_scene() -> crate::testkit::Fixture {
    let mut rng = crate::testkit::Rng::new(11);
    let (h, w) = (250, 250);
    let values = Array2::from_shape_fn((h, w), |_| (0.1 + 0.9 * rng.unit()) as f32);
    let mut land = Array2::from_elem((h, w), false);
    fill(&mut land, 0..110, 0..w, true);
    fill(&mut land, 0..95, 140..143, false); // river from the tile's top edge
    fill(&mut land, 40..95, 160..167, false); // inlet 7 wide, closed at both ends
    fill(&mut land, 116..124, 100..150, true); // island A, 6 rows below the mainland
    fill(&mut land, 138..146, 110..170, true); // island B, 14 rows below A
    for k in 0..14 {
        land[[124 + k, 150 + k]] = true; // diagonal causeway from A's corner to B
    }
    let cond = Array2::from_shape_fn((h, w), |ij| if land[ij] { values[ij] } else { f32::NAN });
    let levels = vec![1, 2, 4, 8];
    let transforms = level_transforms((RES, 0.0, 0.0, 0.0, -RES, h as f64 * RES), &levels, (0, 0));
    crate::testkit::Fixture {
        mask: cond.mapv(f32::is_nan),
        trav: None,
        trans: None,
        transforms,
        levels,
        lambdas: LAMBDAS.to_vec(),
        is_geo: false,
        max_cost: 2.0,
        window_size: 3,
        outer_window: 5,
        offsets: (0, 0),
        mode: WindowMode::Circular,
        pa_to_pa: false,
        crossing: None,
        cond,
    }
}

/// Transgrids with `bands` bands per scenario, NaN where condition is NaN.
fn climate(cond: &Array2<f32>, scenarios: usize, bands: usize) -> Vec<ndarray::Array3<f32>> {
    (0..scenarios)
        .map(|s| {
            ndarray::Array3::from_shape_fn((cond.nrows(), cond.ncols(), bands), |(i, j, b)| {
                if cond[[i, j]].is_finite() {
                    ((i * 3 + j * 7 + s * 5 + b * 11) % 17) as f32 / 17.0
                } else {
                    f32::NAN
                }
            })
        })
        .collect()
}

/// `f` cut to rows and columns starting at `at` with shape `dim`.
fn tile_of(f: &crate::testkit::Fixture, at: (usize, usize), dim: (usize, usize)) -> crate::testkit::Fixture {
    use ndarray::s;
    let rows = at.0..at.0 + dim.0;
    let cols = at.1..at.1 + dim.1;
    let mut t = f.clone();
    t.cond = f.cond.slice(s![rows.clone(), cols.clone()]).to_owned();
    t.mask = f.mask.slice(s![rows.clone(), cols.clone()]).to_owned();
    t.trav = f.trav.as_ref().map(|a| a.slice(s![rows.clone(), cols.clone()]).to_owned());
    t.trans = f.trans.as_ref().map(|v| v.iter().map(|a| a.slice(s![rows.clone(), cols.clone(), ..]).to_owned()).collect());
    t.offsets = at;
    let base = f.transforms[&1];
    t.transforms = level_transforms(base, &f.levels, at);
    t
}

// A tile with offsets gives the whole raster's output at focal cells whose window, plus the gap
// width, lies inside it: tiles cut through land, a river and coarse cells
#[test]
fn check_tiles() {
    let whole = tile_scene();
    let (at, dim) = ((37, 45), (200, 190));
    // Reach 40 cells, coarse cells 8, gap width up to 8 and contact 1: 60 cells inside the tile
    let focal = Array2::from_shape_fn(whole.cond.dim(), |(i, j)| {
        (97..177).contains(&i) && (105..175).contains(&j) && (i + 2 * j) % 7 == 0 && whole.cond[[i, j]].is_finite()
    });
    let cells: Vec<_> = focal.indexed_iter().filter(|(_, &f)| f).map(|(ij, _)| ij).collect();
    assert!(cells.len() > 200, "{}", cells.len());
    for (pa, beri) in [(false, false), (true, false), (false, true)] {
        let mut f = whole.clone();
        f.pa_to_pa = pa;
        if beri {
            f.trans = Some(climate(&f.cond, 2, 3));
        }
        // Protected (and so evaluated too): the focal cells and a sparse lattice of others
        f.mask = Array2::from_shape_fn(f.cond.dim(), |(i, j)| !focal[[i, j]] && (!pa || i % 9 != 4 || j % 9 != 4));
        let tile = tile_of(&f, at, dim);
        let current = (!pa && !beri).then_some((None, 0.0));
        for (mode, width) in current.into_iter().chain([(Some(ALL[0]), 1.6), (Some(ALL[1]), 1.6), (Some(ALL[2]), 2.0)]) {
            let a = under(&f, mode, width);
            let b = under(&tile, mode, width);
            let name = mode.map_or("current", |(m, r)| label(m, r));
            for &(i, j) in &cells {
                let (x, y) = (a[[i, j]], b[[i - at.0, j - at.1]]);
                assert_eq!(x.to_bits(), y.to_bits(), "{name} W={width} pa={pa} beri={beri} at ({i}, {j}): {x} vs {y}");
            }
        }
    }
}

/// Mainland (rows 0 to 39) and a 5 x 30 island `gap` rows below it, on a projected 250 m grid or
/// a geographic grid of 0.0025 degrees near 42 S (cells 0.278 km high, 0.207 km wide).
fn islet_fixture(gap: usize, mainland: bool, geo: bool, pa: bool, beri: bool) -> crate::testkit::Fixture {
    let (cond, island) = islet(gap, mainland);
    let cond = Array2::from_shape_fn(cond.dim(), |(i, j)| cond[[i, j]] - ((i * 7 + j * 13) % 10) as f32 * 0.03);
    let levels = SMALL.levels.to_vec();
    let base = if geo {
        (0.0025, 0.0, 147.0, 0.0, -0.0025, -41.85)
    } else {
        (RES, 0.0, 0.0, 0.0, -RES, cond.nrows() as f64 * RES)
    };
    crate::testkit::Fixture {
        // Under PARC gating the island is protected and the mainland is not
        mask: Array2::from_shape_fn(cond.dim(), |ij| !island[ij] || cond[ij].is_nan()),
        trav: None,
        trans: beri.then(|| climate(&cond, 2, 2)),
        transforms: level_transforms(base, &levels, (0, 0)),
        levels,
        lambdas: LAMBDAS.to_vec(),
        is_geo: geo,
        max_cost: 2.0,
        window_size: SMALL.window,
        outer_window: SMALL.outer,
        offsets: (0, 0),
        mode: WindowMode::Circular,
        pa_to_pa: pa,
        crossing: None,
        cond,
    }
}

// An island is unaffected, bit for bit, by land beyond the gap width in km and wholly affected
// by land within it, on geographic grids and with PARC gating or BERI
#[test]
fn check_variants() {
    // Gaps of 2 and 6 rows: 0.56 and 1.67 km
    for (geo, pa, beri) in [(true, false, false), (true, true, false), (true, false, true)] {
        for gap in [2usize, 6] {
            let with = islet_fixture(gap, true, geo, pa, beri);
            let alone = islet_fixture(gap, false, geo, pa, beri);
            let island = with.mask.mapv(|m| !m);
            for (mode, relax) in ALL {
                let (n, cells) = changed(&under(&with, Some((mode, relax)), 1.0), &under(&alone, Some((mode, relax)), 1.0), &island);
                let expected = if gap == 2 { cells } else { 0 };
                assert_eq!(n, expected, "{} geo={geo} pa={pa} beri={beri} gap {gap}", label(mode, relax));
            }
        }
    }
}

// Every node standing for crossable NoData alone carries no habitat area, no protected
// fraction and no climate similarity, in both modes
#[test]
fn check_crossable() {
    let (cond, west) = inlet_small(4);
    let _ = west;
    let mut f = islet_fixture(2, true, false, true, true);
    f.cond = cond;
    f.mask = Array2::from_shape_fn(f.cond.dim(), |(_, j)| j % 3 != 0);
    f.trans = Some(climate(&f.cond, 2, 2));
    f.transforms = level_transforms((RES, 0.0, 0.0, 0.0, -RES, f.cond.nrows() as f64 * RES), &f.levels, (0, 0));
    for (mode, relax) in ALL {
        let cross = Crossing { mode, width: 1.0, traversal: 0.25 };
        let ctx = prepare_cross(&f.inputs(), &f.config(), &cross, relax).unwrap();
        let mut s = Scratch::new(&ctx);
        s.eval(&ctx, 100, 50).unwrap();
        let n = &s.nodes;
        let water: Vec<usize> = (0..n.len()).filter(|&u| n.count[u] == 0.0).collect();
        assert!(!water.is_empty(), "{}", label(mode, relax));
        for &u in &water {
            assert_eq!((n.cond[u], n.pa[u]), (0.0, 0.0), "{}", label(mode, relax));
            if n.level[u] == 0 {
                assert_eq!(n.trav[u], 0.25, "{}", label(mode, relax));
            }
        }
    }
}

/// Mainland (rows 0 to `m - 1`) and a 5 x 30 island 2 rows below it, so that for some `m` coarse
/// cells hold land of both; climate is finite everywhere, also on NoData, and the same with and
/// without the mainland.
fn shared_fixture(m: usize, mainland: bool, pa: bool, beri: bool) -> crate::testkit::Fixture {
    let (h, w) = (120, 112);
    let mut cond = Array2::from_elem((h, w), f32::NAN);
    if mainland {
        fill(&mut cond, 0..m, 0..w, 0.5);
    }
    fill(&mut cond, m + 2..m + 7, 40..70, 0.9);
    let cond = Array2::from_shape_fn((h, w), |(i, j)| cond[[i, j]] - ((i * 7 + j * 13) % 10) as f32 * 0.03);
    let island = |i: usize, j: usize| (m + 2..m + 7).contains(&i) && (40..70).contains(&j);
    let levels = SMALL.levels.to_vec();
    crate::testkit::Fixture {
        mask: Array2::from_shape_fn((h, w), |(i, j)| !island(i, j)),
        trav: None,
        trans: beri.then(|| climate(&Array2::zeros((h, w)), 2, 2)),
        transforms: level_transforms((RES, 0.0, 0.0, 0.0, -RES, h as f64 * RES), &levels, (0, 0)),
        levels,
        lambdas: LAMBDAS.to_vec(),
        is_geo: false,
        max_cost: 2.0,
        window_size: SMALL.window,
        outer_window: SMALL.outer,
        offsets: (0, 0),
        mode: WindowMode::Circular,
        pa_to_pa: pa,
        crossing: None,
        cond,
    }
}

// An island is unaffected, bit for bit, by land it may not cross to when coarse cells hold land
// of both (several parts per cell), plain, PARC and BERI with climate on NoData
#[test]
fn check_shared() {
    for m in [33, 34, 36, 37] {
        for (pa, beri) in [(false, false), (true, false), (false, true)] {
            let with = shared_fixture(m, true, pa, beri);
            let alone = shared_fixture(m, false, pa, beri);
            let island = with.mask.mapv(|masked| !masked);
            for (mode, relax) in ALL {
                let cross = Crossing { mode, width: 0.0, traversal: 0.0 };
                let (a, stats) = conn_cross(&with.inputs(), &with.config(), &cross, relax).unwrap();
                let parts = match mode {
                    CrossingMode::Landmass => stats.gate.unwrap().mixed,
                    CrossingMode::Strict => stats.strict.unwrap().multi.iter().sum(),
                };
                assert!(parts > 0, "m {m}: no cell holds several parts");
                let (n, _) = changed(&a, &under(&alone, Some((mode, relax)), 0.0), &island);
                assert_eq!(n, 0, "{} m {m} pa={pa} beri={beri}", label(mode, relax));
            }
        }
    }
}

// A 3 km channel on 1 km cells is crossed at W = 3 km and not at 2.5 km, whatever the grid's
// rotation: the far bank changes every focal cell, or none
#[test]
fn check_rotated() {
    let (h, w) = (60, 60);
    let levels = SMALL.levels.to_vec();
    let bank = |far: bool| {
        Array2::from_shape_fn((h, w), |(i, j)| match i {
            0..=27 => 0.8 - ((i * 7 + j * 13) % 10) as f32 * 0.03,
            28..=30 => f32::NAN,
            _ if far => 0.3,
            _ => f32::NAN,
        })
    };
    let focal = Array2::from_shape_fn((h, w), |(i, j)| (20..28).contains(&i) && (20..40).contains(&j));
    for deg in [0.0f64, 30.0, 45.0, 90.0] {
        let (s, c) = deg.to_radians().sin_cos();
        let base = (1000.0 * c, 1000.0 * s, 0.0, 1000.0 * s, -1000.0 * c, 0.0);
        let fixture = |cond: Array2<f32>| crate::testkit::Fixture {
            mask: focal.mapv(|f| !f),
            trav: None,
            trans: None,
            transforms: level_transforms(base, &levels, (0, 0)),
            levels: levels.clone(),
            lambdas: LAMBDAS.to_vec(),
            is_geo: false,
            max_cost: 2.0,
            window_size: SMALL.window,
            outer_window: SMALL.outer,
            offsets: (0, 0),
            mode: WindowMode::Circular,
            pa_to_pa: false,
            crossing: None,
            cond,
        };
        let (with, alone) = (fixture(bank(true)), fixture(bank(false)));
        for (mode, relax) in ALL {
            for (width, crossed) in [(2.5, false), (3.0, true)] {
                let (n, cells) = changed(&under(&with, Some((mode, relax)), width), &under(&alone, Some((mode, relax)), width), &focal);
                assert_eq!(n, if crossed { cells } else { 0 }, "{} {deg} degrees, W = {width} km", label(mode, relax));
            }
        }
    }
}

// Traversal values on NoData that may not be crossed change nothing: the output equals the run
// with NaN there, whether or not other cells are crossable
#[test]
fn check_stray() {
    let (h, w) = (32, 40);
    // West and east land split by a NoData channel 2 cells (0.5 km) wide; a one-cell lake in the
    // west is crossable from W = 0.25 km
    let cond = Array2::from_shape_fn((h, w), |(i, j)| match j {
        _ if (i, j) == (6, 5) => f32::NAN,
        18 | 19 => f32::NAN,
        _ if j < 18 => 0.8 - ((i * 7 + j * 13) % 10) as f32 * 0.03,
        _ => 0.3 + ((i * 5 + j * 3) % 7) as f32 * 0.04,
    });
    let levels = SMALL.levels.to_vec();
    let fixture = |stray: f32| crate::testkit::Fixture {
        mask: Array2::from_shape_fn((h, w), |(i, j)| cond[[i, j]].is_nan() || (i + j) % 3 != 0),
        trav: Some(Array2::from_shape_fn((h, w), |(i, j)| match j {
            18 | 19 => stray,
            _ => cond[[i, j]].mul_add(0.5, 0.2),
        })),
        trans: None,
        transforms: level_transforms((RES, 0.0, 0.0, 0.0, -RES, h as f64 * RES), &levels, (0, 0)),
        levels: levels.clone(),
        lambdas: LAMBDAS.to_vec(),
        is_geo: false,
        max_cost: 2.0,
        window_size: SMALL.window,
        outer_window: SMALL.outer,
        offsets: (0, 0),
        mode: WindowMode::Circular,
        pa_to_pa: false,
        crossing: None,
        cond: cond.clone(),
    };
    let (clean, stray) = (fixture(f32::NAN), fixture(0.5));
    let focal = clean.mask.mapv(|masked| !masked);
    for (mode, relax) in ALL {
        for width in [0.0, 0.3] {
            let (n, cells) = changed(&under(&stray, Some((mode, relax)), width), &under(&clean, Some((mode, relax)), width), &focal);
            assert_eq!(n, 0, "{} W={width}: {n} of {cells} cells", label(mode, relax));
        }
    }
}



/// The bundled EPSG:4326 rasters dumped by `docs/island_bridging_figs/repro/bundled_npy.py` into
/// `CROSS_DATA`: condition, protected fraction, base transform and transgrids.
fn bundled() -> (Array2<f32>, Array2<f32>, crate::affine::Affine, Vec<ndarray::Array3<f32>>) {
    use crate::testkit::npy;
    use ndarray::{Ix2, Ix3};
    let dir = PathBuf::from(std::env::var("CROSS_DATA").expect("set CROSS_DATA"));
    let cond = npy::f32s(&dir.join("condition.npy")).into_dimensionality::<Ix2>().unwrap();
    let pa = npy::f32s(&dir.join("pa.npy")).into_dimensionality::<Ix2>().unwrap();
    let t: Vec<f64> = std::fs::read_to_string(dir.join("transform.txt")).unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
    let trans = (0..)
        .map(|k| dir.join(format!("trans_{k}.npy")))
        .take_while(|p| p.exists())
        .map(|p| npy::f32s(&p).into_dimensionality::<Ix3>().unwrap())
        .collect();
    (cond, pa, crate::affine::Affine::from((t[0], t[1], t[2], t[3], t[4], t[5])), trans)
}

fn geo_fixture(cond: Array2<f32>, mask: Array2<bool>, base: crate::affine::Affine, levels: &[usize], pa: bool) -> crate::testkit::Fixture {
    crate::testkit::Fixture {
        mask,
        trav: None,
        trans: None,
        transforms: level_transforms(base, levels, (0, 0)),
        levels: levels.to_vec(),
        lambdas: LAMBDAS.to_vec(),
        is_geo: true,
        max_cost: 2.0,
        window_size: 5,
        outer_window: 9,
        offsets: (0, 0),
        mode: WindowMode::Circular,
        pa_to_pa: pa,
        crossing: None,
        cond,
    }
}

// Bundled EPSG:4326 data: every land piece other than the largest scores the same, bit for bit,
// alone and with all other land when no crossing reaches it (W = 0, and W = 2 km for pieces whose
// gap-width group holds no other piece), with and without PARC gating; a tile with offsets
// matches the whole raster; and the memory held for BERI parts is printed.
#[test]
#[ignore = "bundled data dumped to CROSS_DATA; run in release"]
fn bundled_geo() {
    let (cond0, pa0, base, trans) = bundled();
    let levels = [1, 2, 4, 8, 16];
    let land = cond0.mapv(f32::is_finite);
    let (pieces, n) = crate::gate::label(&land);
    let mut sizes = vec![0usize; n + 1];
    pieces.iter().for_each(|&p| sizes[p as usize] += 1);
    let largest = (1..=n).max_by_key(|&p| sizes[p]).unwrap() as u32;
    println!("{n} pieces, largest {} cells", sizes[largest as usize]);
    let (dy, dx) = crate::crossing::cell_km(&base, cond0.nrows(), true);
    let width = 2.0;
    let cross = crate::crossing::crossable(&land, width as f64, dy, &dx).unwrap();
    let (groups, _) = crate::gate::label(&Array2::from_shape_fn(land.dim(), |ij| land[ij] || cross[ij]));
    let mut members: std::collections::HashMap<u32, std::collections::HashSet<u32>> = Default::default();
    for (ij, &p) in pieces.indexed_iter() {
        if p != 0 {
            members.entry(groups[ij]).or_default().insert(p);
        }
    }
    let lone = |p: u32| {
        let at = pieces.indexed_iter().find(|(_, &q)| q == p).unwrap().0;
        members[&groups[at]].len() == 1
    };
    for pa in [false, true] {
        // PARC preparation as in Python: condition = max(condition, pa); protected where pa > 0
        let cond = if pa { Array2::from_shape_fn(cond0.dim(), |ij| cond0[ij].max(pa0[ij]) + 0.0 * pa0[ij]) } else { cond0.clone() };
        let focal = |p: u32| Array2::from_shape_fn(cond.dim(), |ij| {
            let mine = if p == 0 { pieces[ij] != largest && pieces[ij] != 0 } else { pieces[ij] == p };
            !(mine && cond[ij].is_finite() && (!pa || pa0[ij] > 0.0))
        });
        let all = geo_fixture(cond.clone(), focal(0), base, &levels, pa);
        for (mode, relax) in ALL {
            for w in [0.0f32, width] {
                let together = under(&all, Some((mode, relax)), w);
                let (mut same, mut cells, mut tested) = (0, 0, 0);
                for p in 1..=n as u32 {
                    if p == largest || (w > 0.0 && !lone(p)) {
                        continue;
                    }
                    let mask = focal(p);
                    if mask.iter().all(|&m| m) {
                        continue;
                    }
                    tested += 1;
                    let alone_cond = Array2::from_shape_fn(cond.dim(), |ij| if pieces[ij] == p { cond[ij] } else { f32::NAN });
                    let alone = under(&geo_fixture(alone_cond, mask.clone(), base, &levels, pa), Some((mode, relax)), w);
                    for (ij, &m) in mask.indexed_iter() {
                        if !m {
                            cells += 1;
                            same += (alone[ij].to_bits() == together[ij].to_bits()) as usize;
                        }
                    }
                }
                println!("{} W={w} pa={pa}: {tested} pieces, {same} of {cells} cells identical", label(mode, relax));
                assert_eq!(same, cells, "{} W={w} pa={pa}", label(mode, relax));
            }
        }
    }

    // Tile with offsets against the whole raster (levels 1 to 8, reach 72 cells)
    let small = [1, 2, 4, 8];
    let (at, dim) = ((101, 77), (300, 350));
    let focal = Array2::from_shape_fn(cond0.dim(), |(i, j)| {
        (at.0 + 84..at.0 + dim.0 - 84).contains(&i) && (at.1 + 84..at.1 + dim.1 - 84).contains(&j) && (i + j) % 3 == 0
    });
    let whole = geo_fixture(cond0.clone(), focal.mapv(|f| !f), base, &small, false);
    let tile = tile_of(&whole, at, dim);
    for (mode, relax) in ALL {
        let (a, b) = (under(&whole, Some((mode, relax)), width), under(&tile, Some((mode, relax)), width));
        let cells: Vec<_> = focal.indexed_iter().filter(|(ij, &f)| f && cond0[*ij].is_finite()).map(|(ij, _)| ij).collect();
        let same = cells.iter().filter(|&&(i, j)| a[[i, j]].to_bits() == b[[i - at.0, j - at.1]].to_bits()).count();
        println!("tile {}: {same} of {} cells identical", label(mode, relax), cells.len());
        assert_eq!(same, cells.len());
    }

    // Memory held by each mode, with and without the bundled transgrids
    let mut f = geo_fixture(cond0.clone(), cond0.mapv(|_| true), base, &levels, false);
    for beri in [false, true] {
        f.trans = beri.then(|| trans.clone());
        for (mode, relax) in ALL.into_iter().take(2) {
            for w in [0.0f32, width] {
                let c = Crossing { mode, width: w, traversal: 0.0 };
                let (_, stats) = conn_cross(&f.inputs(), &f.config(), &c, relax).unwrap();
                let bytes = stats.gate.map(|g| g.bytes).or(stats.strict.map(|s| s.bytes)).unwrap();
                println!("{} W={w} beri={beri} ({} scenarios): held {:.2} MB", label(mode, relax), trans.len(), bytes as f64 / 1e6);
            }
        }
    }
}

/// `a` moved `shift` rows down and columns right inside a larger NaN or `false` frame.
fn shifted<T: Copy>(a: &Array2<T>, shift: (usize, usize), empty: T) -> Array2<T> {
    let (h, w) = a.dim();
    let mut out = Array2::from_elem((h + 8, w + 8), empty);
    out.slice_mut(ndarray::s![shift.0..shift.0 + h, shift.1..shift.1 + w]).assign(a);
    out
}

// Raster alignment against the overview grid never moves island cells between linked and
// sealed: under shifts the island is wholly linked across a 3-row gap and never across a 6-row
// gap at W = 4 cells, in both modes
#[test]
fn check_alignment() {
    for shift in [(1, 0), (3, 5), (6, 2), (7, 7)] {
        for (gap, linked) in [(3usize, true), (6, false)] {
            let (with, island) = islet(gap, true);
            let (alone, _) = islet(gap, false);
            let (with, alone, island) = (shifted(&with, shift, f32::NAN), shifted(&alone, shift, f32::NAN), shifted(&island, shift, false));
            for mode in [Mode::Width(4.0), Mode::Strict(4.0), Mode::Relaxed(4.0)] {
                let (n, cells) = changed(&small(&with, &island, mode), &small(&alone, &island, mode), &island);
                assert_eq!(n, if linked { cells } else { 0 }, "{} gap {gap} shift {shift:?}", mode.name());
            }
        }
    }
}

// A one-cell diagonal causeway keeps every island cell linked under relaxed contact (the mode 2
// rule) at W = 0, in every alignment tried
#[test]
fn check_causeway_shifts() {
    let (h, w) = (120, 112);
    let mut cond = Array2::from_elem((h, w), f32::NAN);
    let mut island = Array2::from_elem((h, w), false);
    fill(&mut cond, 0..40, 0..w, 0.5);
    fill(&mut cond, 60..65, 40..70, 0.9);
    fill(&mut island, 60..65, 40..70, true);
    for k in 0..20 {
        cond[[40 + k, 36 + k]] = 0.9;
    }
    let lone = Array2::from_shape_fn(cond.dim(), |ij| if island[ij] { cond[ij] } else { f32::NAN });
    for shift in [(0, 0), (1, 3), (2, 6), (3, 1), (5, 4), (6, 7), (7, 2)] {
        let (c, l, f) = (shifted(&cond, shift, f32::NAN), shifted(&lone, shift, f32::NAN), shifted(&island, shift, false));
        let (n, cells) = changed(&small(&c, &f, Mode::Relaxed(0.0)), &small(&l, &f, Mode::Relaxed(0.0)), &f);
        assert_eq!(n, cells, "shift {shift:?}");
    }
}
