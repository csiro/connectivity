// Define the Affine struct for transforming ij to xy
#[derive(Debug, Clone, Copy)]
struct Affine {
    x_scale: f64,   // pixel width
    x_skew: f64,    // rotation/skew term (usually 0)
    x_origin: f64,  // top-left corner X
    y_skew: f64,    // rotation/skew term (usually 0)
    y_scale: f64,   // pixel height (negative if north-up)
    y_origin: f64,  // top-left corner Y
}

impl Affine {
    /// Convert (row, col) to (x, y) at pixel centre by default.
    fn xy(&self, row: f64, col: f64) -> (f64, f64) {
        let col_f = col + 0.5; // Offset of 0.5 for pixel centre
        let row_f = row + 0.5;

        let x = self.x_scale * col_f + self.x_skew * row_f + self.x_origin;
        let y = self.y_skew * col_f + self.y_scale * row_f + self.y_origin;
        (x, y)
    }

    /// Upper-left corner coordinates (no offset).
    fn xy_corner(&self, row: f64, col: f64) -> (f64, f64) {
        let x = self.x_scale * col + self.x_skew * row + self.x_origin;
        let y = self.y_skew * col + self.y_scale * row + self.y_origin;
        (x, y)
    }
}

// // Example: north-up raster with 30 m pixels
// let transform = Affine {
//     x_scale: 30.0,  // pixel width
//     b: 0.0,   // no rotation
//     c: 255000.0, // x origin
//     d: 0.0,   // no rotation
//     e: -30.0, // pixel height (negative because y decreases downward)
//     f: 4100000.0, // y origin
// };

// let row = 0.0;
// let col = 0.0;

// // Get pixel centre
// let (x, y) = transform.xy(row, col));


/// Euclidean distance between two 2D points (f64 precision).
pub fn euclidean(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    // Use hypot to avoid intermediate overflow/underflow and get accurate result
    // Benchmark (dx*dx + dy*dy).sqrt() for perfromance; prefer hypot for stability.
    dx.hypot(dy)
}


// Calculate Haversine distance for points in geographic coordinates system
pub fn haversine(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let r = 6_371_000.0; // mean Earth radius in meters
    let (lat1, lon1) = (lat1.to_radians(), lon1.to_radians());
    let (lat2, lon2) = (lat2.to_radians(), lon2.to_radians());

    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2)
        + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    r * c
}

