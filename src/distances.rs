/// Euclidean distance for projected coordinates systems
fn euclidean(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    // Use hypot to avoid intermediate overflow/underflow and get accurate result
    // Benchmark (dx*dx + dy*dy).sqrt() for perfromance; prefer hypot for stability.
    dx.hypot(dy)
}


// Calculate Haversine distance for points in geographic coordinates system
fn haversine(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
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


/// Calculate distance of cells in kilometer
pub fn distance_km(x1: f64, y1: f64, x2: f64, y2: f64, geo: bool) -> f32 {
    let dist: f64 = if geo {
        haversine(x1, y1, x2, y2)
    } else {
        euclidean(x1, y1, x2, y2)
    };

    (dist / 1000.0)  as f32
}

