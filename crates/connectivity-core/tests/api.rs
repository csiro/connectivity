//! The public API on the multi-level fixture of the unit tests, through crate paths only.
use connectivity_core::ndarray::{Array2, Array3};
use connectivity_core::{
    connectivity, level_transforms, pixel_coverage, Affine, Config, Inputs, WindowMode,
};
use std::collections::HashMap;

struct Tile {
    condition: Array2<f32>,
    mask: Array2<bool>,
    traversal: Array2<f32>,
    transgrids: Vec<Array3<f32>>,
    transforms: HashMap<i32, Affine>,
}

const LEVELS: [usize; 3] = [1, 2, 4];
const OFFSETS: (usize, usize) = (3, 5);

/// `testkit::Fixture::multilevel`, built with the public helpers.
fn tile() -> Tile {
    let condition = Array2::from_shape_fn((13, 14), |(i, j)| {
        if (i * 7 + j * 11) % 29 == 0 {
            f32::NAN
        } else {
            ((i * 7 + j * 13) % 10 + 1) as f32 / 10.0
        }
    });
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
    let transgrids = (0..2)
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
    let base = (1000.0, 0.0, 500_000.0, 0.0, -1000.0, 6_000_000.0);
    let transforms = level_transforms(base, &LEVELS, OFFSETS);
    Tile { condition, mask, traversal, transgrids, transforms }
}

fn inputs<'a>(t: &'a Tile, beri: bool) -> Inputs<'a> {
    Inputs {
        condition: &t.condition,
        mask: &t.mask,
        traversal: Some(&t.traversal),
        transgrids: beri.then_some(t.transgrids.as_slice()),
        transforms: &t.transforms,
    }
}

fn config(window_mode: WindowMode, pa_to_pa: bool) -> Config<'static> {
    Config {
        levels: &LEVELS,
        lambdas: &[2.0, 20.0],
        is_geo: false,
        max_cost: 20.0,
        window_size: 3,
        outer_window: 5,
        offsets: OFFSETS,
        window_mode,
        pa_to_pa,
    }
}

fn same_bits(a: &Array2<f32>, b: &Array2<f32>) -> bool {
    a.dim() == b.dim() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

// The frozen sample values of the unit tests' `multilevel_values`
#[test]
fn multilevel_samples() {
    let t = tile();
    let cells = [(0, 1), (3, 5), (6, 8), (12, 13)];
    for (mode, beri, pa_to_pa, expected) in [
        (WindowMode::Circular, false, false, [0.035036724, 0.060928639, 0.057472181, 0.023413222]),
        (WindowMode::Circular, false, true, [0.031013189, 0.051940750, 0.050136760, 0.017623197]),
        (WindowMode::Circular, true, false, [0.287769228, 0.330879152, 0.336351663, 0.270322204]),
        (WindowMode::Square, false, false, [0.036365397, 0.062699854, 0.057887629, 0.023465034]),
        (WindowMode::Square, false, true, [0.032288730, 0.053552385, 0.050358951, 0.017685590]),
        (WindowMode::Square, true, false, [0.288273066, 0.334522188, 0.339534730, 0.269480884]),
    ] {
        let out = connectivity(&inputs(&t, beri), &config(mode, pa_to_pa)).unwrap();
        for (cell, expected) in cells.iter().zip(expected) {
            let actual = out[*cell];
            assert!((actual - expected).abs() <= 1.0e-6, "{mode:?} {beri} {pa_to_pa} {cell:?}: {actual}");
        }
        for (cell, &masked) in t.mask.indexed_iter() {
            assert_eq!(out[cell].is_nan(), masked);
        }
    }
}

// Any thread count of the caller's pool gives the same bits
#[test]
fn thread_pools() {
    let t = tile();
    for beri in [false, true] {
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| connectivity(&inputs(&t, beri), &config(WindowMode::Circular, true)).unwrap())
        };
        let one = run(1);
        assert!(same_bits(&one, &run(3)) && same_bits(&one, &run(8)));
    }
}

#[test]
fn optional_inputs() {
    let t = tile();
    let mut plain = inputs(&t, false);
    plain.traversal = None;
    let expected = connectivity(&plain, &config(WindowMode::Circular, false)).unwrap();
    plain.transgrids = Some(&[]);
    assert!(same_bits(&connectivity(&plain, &config(WindowMode::Circular, false)).unwrap(), &expected));
    let empty = Config { lambdas: &[], ..config(WindowMode::Circular, false) };
    let zeros = connectivity(&plain, &empty).unwrap();
    assert!(t.mask.iter().zip(&zeros).all(|(&m, &v)| if m { v.is_nan() } else { v == 0.0 }));
}

#[test]
fn default_config() {
    let c = Config::default();
    assert_eq!(c.levels, &[1, 2, 4, 8, 16, 32]);
    assert_eq!(c.lambdas, &[2.0, 20.0, 200.0]);
    assert_eq!((c.max_cost, c.window_size, c.outer_window), (2.0, 3, 9));
    assert_eq!((c.is_geo, c.offsets, c.window_mode, c.pa_to_pa), (false, (0, 0), WindowMode::Circular, false));
}

#[test]
fn window_modes() {
    assert_eq!(WindowMode::parse(" SqUaRe "), Ok(WindowMode::Square));
    assert_eq!(WindowMode::parse("circular"), Ok(WindowMode::Circular));
    assert_eq!(
        WindowMode::parse("hexagon"),
        Err("Invalid window_mode 'hexagon'. Expected 'square' or 'circular'.".to_string())
    );
}

// Level transforms match rasterio's `base * scale(L) * translation(col0 // L, row0 // L)`
#[test]
fn tile_transforms() {
    let base = Affine::from((250.0, 0.0, 500_000.0, 0.0, -250.0, 6_000_000.0));
    let transforms = level_transforms(base, &[1, 2, 4], (5, 11));
    for (level, (x0, y0)) in [(1, (502_750.0, 5_998_750.0)), (2, (502_500.0, 5_999_000.0)), (4, (502_000.0, 5_999_000.0))] {
        let t = transforms[&level];
        let s = 250.0 * level as f64;
        assert_eq!((t.x_scale, t.x_skew, t.x_origin, t.y_skew, t.y_scale, t.y_origin), (s, 0.0, x0, 0.0, -s, y0));
        assert_eq!(t.xy(0, 0), (x0 + s / 2.0, y0 - s / 2.0));
    }
    assert_eq!(base.col_row(500_625.0, 5_999_000.0), (2.5, 4.0));
}

#[test]
fn traversal_shape() {
    let t = tile();
    let short = t.traversal.slice(connectivity_core::ndarray::s![..12, ..]).to_owned();
    let mut bad = inputs(&t, false);
    bad.traversal = Some(&short);
    let error = connectivity(&bad, &config(WindowMode::Circular, false)).unwrap_err();
    assert!(error.to_string().starts_with("Dimension mismatch for key"), "{error}");
}

#[test]
#[should_panic(expected = "Window size must be odd-numbered")]
fn even_window_panics() {
    let t = tile();
    connectivity(&inputs(&t, false), &Config { window_size: 4, ..config(WindowMode::Circular, false) }).unwrap();
}

#[test]
#[should_panic(expected = "Missing base-level transform (key=1).")]
fn needs_base_transform() {
    let t = tile();
    let transforms: HashMap<i32, Affine> = t.transforms.iter().filter(|(&l, _)| l != 1).map(|(&l, &a)| (l, a)).collect();
    let all_masked = Array2::from_elem((13, 14), true);
    let inputs = Inputs { mask: &all_masked, transforms: &transforms, ..inputs(&t, false) };
    connectivity(&inputs, &config(WindowMode::Circular, false)).unwrap();
}

#[test]
fn coverage_fractions() {
    // 10 m pixels from (1000, 2000); pixel (row 1, col 2) spans x 1020..1030, y 1980..1990
    let transform = Affine::from((10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0));
    let square = |x0: f64, y0: f64, x1: f64, y1: f64| vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    let polygons = vec![
        (square(1020.0, 1990.0, 1030.0, 1980.0), vec![]),
        (square(1040.0, 1990.0, 1045.0, 1980.0), vec![]),
        (square(1000.0, 1960.0, 1040.0, 1940.0), vec![square(1010.0, 1955.0, 1030.0, 1945.0)]),
    ];
    let out = pixel_coverage(&polygons, &transform, 6, 6);
    assert_eq!((out[[1, 2]], out[[1, 4]]), (1.0, 0.5));
    for row in 4..6 {
        assert_eq!(out.row(row).to_vec(), vec![1.0, 0.5, 0.5, 1.0, 0.0, 0.0]);
    }
    assert_eq!(out.sum(), 7.5);
    assert_eq!(pixel_coverage(&[], &transform, 2, 3), Array2::<f32>::zeros((2, 3)));
}
