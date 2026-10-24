use std::collections::HashMap;
use std::convert::From;

/// Affine transform from (row, col) pixel indices to world (x, y) coordinates.
///
/// The fields follow the rasterio/`affine` tuple order `(a, b, c, d, e, f)`, i.e.
/// `tuple(transform)[:6]` in Python, not GDAL's `GetGeoTransform` order:
/// `x = a * col + b * row + c` and `y = d * col + e * row + f`.
/// Projected coordinates are in metres; geographic coordinates are longitude (x) and
/// latitude (y) in degrees.
#[derive(Debug, Clone, Copy)]
pub struct Affine {
    pub x_scale: f64,   // pixel width
    pub x_skew: f64,    // rotation/skew term (usually 0)
    pub x_origin: f64,  // top-left corner X
    pub y_skew: f64,    // rotation/skew term (usually 0)
    pub y_scale: f64,   // pixel height (negative if north-up)
    pub y_origin: f64,  // top-left corner Y
}

impl Affine {
    /// Convert (row, col) to (x, y) at pixel centre by default.
    pub fn xy(&self, row: i32, col: i32) -> (f64, f64) {
        let col_f = (col as f64) + 0.5;
        let row_f = (row as f64) + 0.5;

        let x = self.x_scale * col_f + self.x_skew * row_f + self.x_origin;
        let y = self.y_skew * col_f + self.y_scale * row_f + self.y_origin;
        (x, y)
    }

    /// Convert world (x, y) to continuous (col, row) pixel coordinates.
    pub fn col_row(&self, x: f64, y: f64) -> (f64, f64) {
        let det = self.x_scale * self.y_scale - self.x_skew * self.y_skew;
        let dx = x - self.x_origin;
        let dy = y - self.y_origin;
        let col = (self.y_scale * dx - self.x_skew * dy) / det;
        let row = (-self.y_skew * dx + self.x_scale * dy) / det;
        (col, row)
    }
}

impl From<(f64, f64, f64, f64, f64, f64)> for Affine {
    /// From an `(a, b, c, d, e, f)` tuple in rasterio/`affine` order.
    fn from(t: (f64, f64, f64, f64, f64, f64)) -> Self {
        Self {
            x_scale: t.0,
            x_skew: t.1,
            x_origin: t.2,
            y_skew: t.3,
            y_scale: t.4,
            y_origin: t.5,
        }
    }
}

/// Level transforms of a tile, keyed by level, anchored like `rastio.read_raster`:
/// `base * Affine.scale(level) * Affine.translation(col0 // level, row0 // level)`,
/// with the `affine` package's operation order.
///
/// `base` is the full raster's base-level transform and `offsets` the tile's `(row0, col0)`
/// in base pixels, so level `L` maps the tile's overview grid, whose cell (0, 0) is global
/// overview cell `(row0 / L, col0 / L)`, and level 1 maps the tile itself.
pub fn level_transforms(
    base: impl Into<Affine>,
    levels: &[usize],
    offsets: (usize, usize),
) -> HashMap<i32, Affine> {
    let Affine { x_scale: a, x_skew: b, x_origin: c, y_skew: d, y_scale: e, y_origin: f } = base.into();
    levels
        .iter()
        .map(|&level| {
            let l = level as f64;
            let (sa, sb, sd, se) = (a * l + b * 0.0, a * 0.0 + b * l, d * l + e * 0.0, d * 0.0 + e * l);
            let (tc, tr) = ((offsets.1 / level) as f64, (offsets.0 / level) as f64);
            let t = (sa, sb, sa * tc + sb * tr + c, sd, se, sd * tc + se * tr + f);
            (level as i32, Affine::from(t))
        })
        .collect()
}
