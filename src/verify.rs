//! Verification runs on captured workloads (ignored by default; not timed).
//!
//! `CONN_INPUTS=<dir> CONN_OUT=<dir> cargo test --release verify -- --ignored --nocapture`
//! reads every `<dir>/<workload>/rust.txt` (or the names in `CONN_WORKLOADS`) and, per workload:
//! compares Optimized with the Canonical reference bit for bit, runs the structural comparison
//! on every unmasked cell, and writes the Optimized, Canonical and Legacy outputs as raw f32.
use crate::builder::tests::{compare_tally, context, table_inject, Tally};
use crate::builder::Scratch;
use crate::core;
use crate::tables::STATUS_NAMES;
use crate::testkit::{load_workload, npy, same_bits, Fixture};
use rayon::prelude::*;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

fn workloads() -> (PathBuf, PathBuf, Vec<String>) {
    let inputs = PathBuf::from(std::env::var("CONN_INPUTS").expect("CONN_INPUTS"));
    let out = PathBuf::from(std::env::var("CONN_OUT").expect("CONN_OUT"));
    std::fs::create_dir_all(&out).unwrap();
    let names = match std::env::var("CONN_WORKLOADS") {
        Ok(list) => list.split(',').map(str::to_string).collect(),
        Err(_) => {
            let mut names: Vec<String> = std::fs::read_dir(&inputs)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().join("rust.txt").exists())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();
            names.sort();
            names
        }
    };
    (inputs, out, names)
}

fn optimized(f: &Fixture) -> (ndarray::Array2<f32>, core::Stats) {
    core::conn_stats(
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
    .unwrap()
}

/// Structural comparison of every unmasked cell with per-edge distance tallies.
fn structure(f: &Fixture) -> (usize, Tally) {
    let ctx = context(f);
    let prepared = f.prepared();
    let cells: Vec<(usize, usize)> = f.mask.indexed_iter().filter(|(_, &m)| !m).map(|(c, _)| c).collect();
    let tally = cells
        .par_chunks(64)
        .map_init(
            || Scratch::new(&ctx),
            |scratch, chunk| {
                let mut tally = Tally::default();
                for &(i, j) in chunk {
                    compare_tally(&ctx, &prepared, scratch, i, j, &mut tally);
                }
                tally
            },
        )
        .reduce(Tally::default, |mut a, b| {
            a.add(&b);
            a
        });
    (cells.len(), tally)
}

fn named(counts: &[u64]) -> String {
    let total: u64 = counts.iter().sum();
    STATUS_NAMES
        .iter()
        .zip(counts)
        .filter(|(_, &n)| n > 0)
        .map(|(name, &n)| format!("{name} {n} ({:.4}%)", 100.0 * n as f64 / total.max(1) as f64))
        .collect::<Vec<_>>()
        .join(", ")
}

#[test]
#[ignore]
fn verify_workloads() {
    let (inputs, out, names) = workloads();
    for name in names {
        let f = load_workload(&inputs.join(&name));
        let mut report = String::new();
        let t = Instant::now();
        let (opt, stats) = optimized(&f);
        writeln!(report, "{name}: optimized {:.2}s {stats:?}", t.elapsed().as_secs_f64()).unwrap();

        let t = Instant::now();
        let can = f.canonical(None);
        let equal = same_bits(&opt, &can);
        writeln!(report, "  canonical {:.2}s, optimized == canonical: {equal}", t.elapsed().as_secs_f64()).unwrap();

        let t = Instant::now();
        let leg = f.legacy();
        writeln!(report, "  legacy {:.2}s", t.elapsed().as_secs_f64()).unwrap();

        writeln!(report, "  optimized edges by table status: {}", named(&stats.edges)).unwrap();

        let t = Instant::now();
        let ctx = context(&f);
        let inject = table_inject(&ctx);
        let injected = f.canonical(Some(&inject));
        let inj_equal = same_bits(&opt, &injected);
        writeln!(report, "  canonical+tables {:.2}s, optimized == injected: {inj_equal}", t.elapsed().as_secs_f64())
            .unwrap();

        let t = Instant::now();
        let (cells, tally) = structure(&f);
        writeln!(
            report,
            "  structure: {cells} cells, {} edges equal ({:.2}s); tables: {}; one-ulp {}, further {}, cost mismatches {}",
            tally.edges,
            t.elapsed().as_secs_f64(),
            named(&tally.status),
            tally.ulp1,
            tally.ulp_more,
            tally.cost_mismatch
        )
        .unwrap();

        npy::write_f32(&out.join(format!("{name}.optimized.f32")), &opt);
        npy::write_f32(&out.join(format!("{name}.canonical.f32")), &can);
        npy::write_f32(&out.join(format!("{name}.legacy.f32")), &leg);
        npy::write_f32(&out.join(format!("{name}.injected.f32")), &injected);
        std::fs::write(out.join(format!("{name}.verify.txt")), &report).unwrap();
        print!("{report}");
        assert!(equal, "{name}: optimized differs from canonical");
        assert!(inj_equal, "{name}: optimized differs from canonical with table distances");
        assert_eq!((tally.ulp1, tally.ulp_more, tally.cost_mismatch), (0, 0, 0), "{name}");
    }
}

// Per-phase timing of one workload on one thread (development aid)
#[test]
#[ignore]
fn profile_phases() {
    let (inputs, _, names) = workloads();
    for name in names {
        let f = load_workload(&inputs.join(&name));
        let ctx = context(&f);
        let mut s = Scratch::new(&ctx);
        let cells: Vec<(usize, usize)> =
            f.mask.indexed_iter().filter(|(_, &m)| !m).map(|(c, _)| c).step_by(7).collect();
        let (mut build, mut route, mut score, mut edges, mut nodes) = (0.0, 0.0, 0.0, 0usize, 0usize);
        for &(i, j) in &cells {
            let t = Instant::now();
            s.build(&ctx, i, j);
            build += t.elapsed().as_secs_f64();
            edges += s.csr.edges.len();
            nodes += s.nodes.len();
            let t = Instant::now();
            if s.source != crate::graph::NONE {
                s.router.route(&s.csr, s.source);
            }
            route += t.elapsed().as_secs_f64();
            let t = Instant::now();
            std::hint::black_box(s.eval(&ctx, i, j));
            score += t.elapsed().as_secs_f64();
        }
        // Distance cost alone: the legacy calls for every edge of the sampled cells
        let t = Instant::now();
        let mut acc = 0.0f32;
        for &(i, j) in &cells {
            let (x1, y1) = ctx.transforms[0].unwrap().xy(i as i32, j as i32);
            for k in 0..(edges / cells.len()) {
                let (x2, y2) = ctx.transforms[0].unwrap().xy(i as i32 + (k % 3) as i32 - 1, j as i32 + 1);
                acc += crate::distances::distance_km(x1, y1, x2, y2, ctx.is_geo);
            }
        }
        let dist = t.elapsed().as_secs_f64();
        std::hint::black_box(acc);
        let n = cells.len() as f64;
        println!(
            "{name}: {} cells, {:.0} nodes, {:.0} edges per cell | build {:.1} us, route {:.1} us, full eval {:.1} us, distances {:.1} us per cell",
            cells.len(), nodes as f64 / n, edges as f64 / n,
            build / n * 1e6, route / n * 1e6, score / n * 1e6, dist / n * 1e6
        );
    }
}

// Scratch, cache and allocation figures of each workload in an 8-thread pool
#[test]
#[ignore]
fn resource_report() {
    use crate::testkit::alloc;
    let (inputs, out, names) = workloads();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let mut report = String::new();
    for name in names {
        let f = load_workload(&inputs.join(&name));
        let (_, stats) = pool.install(|| optimized(&f));
        let total: usize = stats.scratch.iter().sum();
        let max = stats.scratch.iter().max().copied().unwrap_or(0);

        // Steady state on one thread: a second pass over the same cells must not allocate
        let ctx = context(&f);
        let mut scratch = Scratch::new(&ctx);
        let cells: Vec<(usize, usize)> =
            f.mask.indexed_iter().filter(|(_, &m)| !m).map(|(c, _)| c).step_by(13).collect();
        let before = alloc::count();
        cells.iter().for_each(|&(i, j)| _ = std::hint::black_box(scratch.eval(&ctx, i, j)));
        let growth = alloc::count() - before;
        let before = alloc::count();
        cells.iter().for_each(|&(i, j)| _ = std::hint::black_box(scratch.eval(&ctx, i, j)));
        let steady = alloc::count() - before;
        writeln!(
            report,
            "{name}: {} scratch instances, max {:.0} KiB each, {:.2} MiB in all; cache {:.2} MiB \
             (tables {:.2} MiB, template phases per level {:?}); setup {:.4}s, process {:.3}s; \
             {} cells: {growth} first-pass growth allocations, {steady} steady-state",
            stats.scratch.len(),
            max as f64 / 1024.0,
            total as f64 / 1048576.0,
            stats.cache_bytes as f64 / 1048576.0,
            stats.table_bytes as f64 / 1048576.0,
            stats.cached,
            stats.setup_s,
            stats.process_s,
            cells.len()
        )
        .unwrap();
        assert_eq!(steady, 0, "{name}");
    }
    std::fs::write(out.join("resources.txt"), &report).unwrap();
    print!("{report}");
}
