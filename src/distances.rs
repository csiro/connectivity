

// Define the Affine struct for transforming ij to xy
#[derive(Debug, Clone, Copy)]
struct Affine {
    a: f64, // pixel width
    b: f64, // row rotation (usually 0)
    c: f64, // x origin (upper-left corner)
    d: f64, // col rotation (usually 0)
    e: f64, // pixel height (negative if north-up)
    f: f64, // y origin (upper-left corner)
}

impl Affine {
    /// Convert (row, col) to (x, y) in map coordinates.
    /// For pixel centres, use offset = (0.5, 0.5).
    fn xy(&self, row: f64, col: f64, offset: (f64, f64)) -> (f64, f64) {
        let (off_row, off_col) = offset;
        let col_f = col + off_col;
        let row_f = row + off_row;

        let x = self.a * col_f + self.b * row_f + self.c;
        let y = self.d * col_f + self.e * row_f + self.f;
        (x, y)
    }
}

// // Example: north-up raster with 30 m pixels
// let transform = Affine {
//     a: 30.0,  // pixel width
//     b: 0.0,   // no rotation
//     c: 255000.0, // x origin
//     d: 0.0,   // no rotation
//     e: -30.0, // pixel height (negative because y decreases downward)
//     f: 4100000.0, // y origin
// };

// let row = 0.0;
// let col = 0.0;

// // Get pixel centre
// let (x, y) = transform.xy(row, col, (0.5, 0.5));


// Calculate Haversine distance for points in geographic coordinates system
pub fn distance_haversine(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
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

