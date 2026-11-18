//! Verification runs on captured workloads (ignored by default; not timed).
//!
//! `CONN_INPUTS=<dir> CONN_OUT=<dir> cargo test --release -p connectivity-core verify -- --ignored --nocapture`
//! reads every `<dir>/<workload>/rust.txt` (or the names in `CONN_WORKLOADS`) and, per workload:
//! compares Optimized with the Canonical reference bit for bit, runs the structural comparison
//! on every unmasked cell, and writes the Optimized, Canonical and Legacy outputs as raw f32.
//! The references have no NoData crossing modes: a crossing workload is checked for identical
//! output on 1, 2 and 12 threads and its output written, for comparison with a recorded baseline
//! (`tools/core_split/perf.py` compares whole-call outputs of two builds).
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
    core::conn_stats(&f.inputs(), &f.config()).unwrap()
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
        npy::write_f32(&out.join(format!("{name}.optimized.f32")), &opt);
        if f.crossing.is_some() {
            for threads in [1, 2, 12] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                let again = pool.install(|| optimized(&f).0);
                assert!(same_bits(&opt, &again), "{name}: {threads} threads differ");
            }
            writeln!(report, "  crossing {:?}: no reference; 1, 2 and 12 threads identical", f.crossing).unwrap();
            std::fs::write(out.join(format!("{name}.verify.txt")), &report).unwrap();
            print!("{report}");
            continue;
        }

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

// Per-phase timing on one thread (attribution aid, not a benchmark): preparation, then for every
// seventh unmasked cell the steps of `Scratch::eval` in order (admission, BERI similarities, edges,
// both searches, scoring), each timed around the same function `eval` calls. Whole-call timings
// come from the public-API harness (`tools/core_split/perf.py`); this test binary also counts
// allocations. `CONN_STEP` changes the sampling step.
#[test]
#[ignore]
fn profile_phases() {
    let (inputs, out, names) = workloads();
    let step: usize = std::env::var("CONN_STEP").map_or(7, |s| s.parse().unwrap());
    let mut report = String::new();
    for name in names {
        let f = load_workload(&inputs.join(&name));
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            let t = Instant::now();
            let ctx = context(&f);
            let prepare = t.elapsed().as_secs_f64();
            let mut s = Scratch::new(&ctx);
            let cells: Vec<(usize, usize)> =
                f.mask.indexed_iter().filter(|(_, &m)| !m).map(|(c, _)| c).step_by(step).collect();
            let mut time = [0f64; 5];
            let (mut nodes, mut edges, mut reached, mut sum) = (0usize, 0usize, 0usize, 0f64);
            for &(i, j) in &cells {
                let t0 = Instant::now();
                s.admit_window(&ctx, i, j);
                let t1 = Instant::now();
                s.climate(&ctx, i, j);
                let t2 = Instant::now();
                s.connect(&ctx, i, j);
                let t3 = Instant::now();
                s.route(&ctx).unwrap();
                let t4 = Instant::now();
                sum += s.score(&ctx) as f64;
                let t5 = Instant::now();
                for (k, (a, b)) in [(t0, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5)].into_iter().enumerate() {
                    time[k] += (b - a).as_secs_f64();
                }
                nodes += s.nodes.len();
                edges += s.csr.edges.len();
                reached += s.router.paths.len();
            }
            let n = cells.len() as f64;
            let total: f64 = time.iter().sum();
            let pct = |k: usize| 100.0 * time[k] / total;
            writeln!(
                report,
                "{name}: prepare {prepare:.4}s; {} cells, {:.2} us/cell: admit {:.1}%, sims {:.1}%, link {:.1}%, \
                 route {:.1}%, score {:.1}%; {:.0} nodes, {:.0} edges, {:.0} destinations per cell; sum {sum:.6}",
                cells.len(),
                total / n * 1e6,
                pct(0),
                pct(1),
                pct(2),
                pct(3),
                pct(4),
                nodes as f64 / n,
                edges as f64 / n,
                reached as f64 / n,
            )
            .unwrap();
        });
    }
    std::fs::write(out.join("phases.txt"), &report).unwrap();
    print!("{report}");
}

// Scratch, cache and allocation figures of each workload in an 8-thread pool. The cache is also
// rebuilt on one thread so the allocator can count its retained and peak heap.
#[test]
#[ignore]
fn resource_report() {
    use crate::builder::tests::measured;
    use crate::template::CACHE_BUDGET;
    use crate::testkit::alloc;
    let (inputs, out, names) = workloads();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let mut report = String::new();
    for name in names {
        let f = load_workload(&inputs.join(&name));
        let (_, stats) = pool.install(|| optimized(&f));
        let total: usize = stats.scratch.iter().sum();
        let max = stats.scratch.iter().max().copied().unwrap_or(0);

        let ctx = context(&f);
        let (tables, templates, retained, peak) = measured(&ctx, &f, CACHE_BUDGET);
        assert_eq!(retained, tables.bytes + templates.bytes, "{name}");
        assert_eq!(retained, stats.cache_bytes, "{name}");
        assert!(peak <= CACHE_BUDGET, "{name}");

        // Steady state on one thread: a second pass over the same cells must not allocate
        let mut scratch = Scratch::new(&ctx);
        let cells: Vec<(usize, usize)> =
            f.mask.indexed_iter().filter(|(_, &m)| !m).map(|(c, _)| c).step_by(13).collect();
        let before = alloc::count();
        cells.iter().for_each(|&(i, j)| _ = std::hint::black_box(scratch.eval(&ctx, i, j).unwrap()));
        let growth = alloc::count() - before;
        let before = alloc::count();
        cells.iter().for_each(|&(i, j)| _ = std::hint::black_box(scratch.eval(&ctx, i, j).unwrap()));
        let steady = alloc::count() - before;
        writeln!(
            report,
            "{name}: {} scratch instances, max {:.0} KiB each, {:.2} MiB in all; cache retained \
             {:.3} MiB measured (= reported), peak {:.3} MiB measured ({:.3} reported bound), \
             fixed {} B, tables {:.3} MiB, phases {:?}, links {:?}; climate {:.3} MiB; setup {:.4}s, process {:.3}s; \
             {} cells: {growth} first-pass growth allocations, {steady} steady-state",
            stats.scratch.len(),
            max as f64 / 1024.0,
            total as f64 / 1048576.0,
            retained as f64 / 1048576.0,
            peak as f64 / 1048576.0,
            stats.cache_peak as f64 / 1048576.0,
            stats.cache_fixed,
            stats.table_bytes as f64 / 1048576.0,
            stats.cached,
            stats.linked,
            stats.climate_bytes as f64 / 1048576.0,
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
