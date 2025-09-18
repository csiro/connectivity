// Define the Affine struct for transforming ij to xy
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
}

impl From<(f64, f64, f64, f64, f64, f64)> for Affine {
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

