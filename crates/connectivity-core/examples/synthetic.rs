//! Connectedness, PARC, BERI and pixel coverage on synthetic arrays, with no file I/O.
//!
//! `cargo run --release -p connectivity-core --example synthetic`
use connectivity_core::ndarray::{Array2, Array3, Zip};
use connectivity_core::{connectivity, level_transforms, pixel_coverage, Config, Inputs};
use std::error::Error;

fn summary(name: &str, values: &Array2<f32>) {
    let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let mean = finite.iter().map(|&v| v as f64).sum::<f64>() / finite.len().max(1) as f64;
    let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!("{name:24} {} cells evaluated, mean {mean:.4}, max {max:.4}", finite.len());
}

fn main() -> Result<(), Box<dyn Error>> {
    // A 96 x 128 tile whose top-left pixel is row 40, column 24 of a projected raster with
    // 100 m pixels; overview cells and transforms are anchored to the full raster's grid
    let (rows, cols, offsets) = (96, 128, (40, 24));
    let base = (100.0, 0.0, 400_000.0, 0.0, -100.0, 6_100_000.0);
    let levels = [1, 2, 4, 8];
    let transforms = level_transforms(base, &levels, offsets);

    // Habitat patches in [0, 1] around a lake of missing data; masked cells are skipped
    let condition = Array2::from_shape_fn((rows, cols), |(i, j)| {
        let (y, x) = (i as f32 - 48.0, j as f32 - 64.0);
        if x * x + y * y < 120.0 {
            f32::NAN
        } else {
            (0.5 + 0.5 * (i as f32 / 9.0).sin() * (j as f32 / 13.0).cos()).clamp(0.0, 1.0)
        }
    });
    let mask = condition.mapv(f32::is_nan);
    let inputs = Inputs { condition: &condition, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
    let config = Config { levels: &levels, lambdas: &[2.0, 20.0], outer_window: 5, offsets, ..Config::default() };

    let connectedness = connectivity(&inputs, &config)?;
    summary("connectedness", &connectedness);
    // Connected Habitat is caller post-processing; Python's default is option 3
    let mut habitat = connectedness.clone();
    Zip::from(&mut habitat).and(&condition).for_each(|h, &c| *h = (c * *h).sqrt());
    summary("connected habitat", &habitat);

    // A road two rows wide that is hard to cross (traversal = 1 - resistance) on otherwise
    // condition-based traversal
    let traversal = Array2::from_shape_fn((rows, cols), |(i, j)| if i == 70 || i == 71 { 0.05 } else { condition[[i, j]] });
    let with_road = connectivity(&Inputs { traversal: Some(&traversal), ..inputs }, &config)?;
    summary("connectedness, road", &with_road);

    // PARC: protected-area proportions from a reserve polygon with a hole, in world coordinates
    let tile = transforms[&1];
    let (x0, y0) = (tile.x_origin, tile.y_origin);
    let square = |a: f64, b: f64, c: f64, d: f64| vec![(x0 + a, y0 - b), (x0 + c, y0 - b), (x0 + c, y0 - d), (x0 + a, y0 - d)];
    let reserve = vec![(square(500.0, 300.0, 4_050.0, 3_520.0), vec![square(1_500.0, 1_000.0, 2_500.0, 2_000.0)])];
    let pa = pixel_coverage(&reserve, &tile, rows, cols);
    // As numpy.maximum in Python: missing condition stays missing
    let parc_condition = Zip::from(&condition).and(&pa).map_collect(|&c, &p| if c.is_nan() { c } else { c.max(p) });
    let parc_mask = Zip::from(&parc_condition).and(&pa).map_collect(|&c, &p| !(p > 0.0 && c.is_finite()));
    let parc_inputs = Inputs { condition: &parc_condition, mask: &parc_mask, ..inputs };
    summary("PARC", &connectivity(&parc_inputs, &config)?);
    summary("PARC, pa_to_pa", &connectivity(&parc_inputs, &Config { pa_to_pa: true, ..config })?);

    // BERI: the current climate first, then two warmer scenarios, with three bands each
    let climate = |shift: f32| {
        Array3::from_shape_fn((rows, cols, 3), |(i, j, b)| (i as f32 / 40.0 + j as f32 / 60.0) * (b + 1) as f32 * 0.1 + shift)
    };
    let scenarios = [climate(0.0), climate(0.05), climate(0.15)];
    summary("BERI", &connectivity(&Inputs { transgrids: Some(&scenarios), ..inputs }, &config)?);
    Ok(())
}
