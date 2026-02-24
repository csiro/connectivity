use crate::graph::EdgeData;

// Compute the connectedness from a segment and a lambda
pub fn connectedness(segment: &[EdgeData], lambda: f32) -> f32 {
    let (sum_numerator, sum_denominator): (f32, f32) = segment
        .iter()
        .map(|edge| {
            let numerator = (-(edge.adj_dist / lambda)).exp() * edge.condition;
            let denominator = (-(edge.geo_dist / lambda)).exp();
            (numerator, denominator)
        })
        .fold((0.0, 0.0), |(acc_num, acc_den), (num, den)| {
            (acc_num + num, acc_den + den)
        });

    if sum_denominator > 0.0 {
        sum_numerator / sum_denominator
    } else {
        0.0
    }
}


/// Aggregating senarios
#[inline]
fn minimax(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    
    let mean = x.iter().sum::<f32>() / x.len() as f32;  
    // Find the minimum value
    let min = x.iter()
        .fold(f32::INFINITY, |acc, &val| acc.min(val));
    
    0.5 * (mean + min)
}

/// Compute a BERI score from a segment and a lambda
pub fn beri_score(segment: &[EdgeData], lambda: f32) -> f32 {
    const DENOM_VAL: f32 = 283.465; // 5.785 * (50 - 1)
    
    if segment.is_empty() {
        return 0.0;
    }

    // Get number of scenarios (including current)
    let n_scenario = match segment.first() {
        Some(ed) => ed.similarities.len(),
        None => return 0.0,
    };
    if n_scenario == 0 {
        return 0.0;
    }
    // Initialize numerator vector with capacity
    let mut numerator = vec![0.0f32; n_scenario];
    let mut denominator = 0.0f32;

    let inv_lambda: f32 = 1.0 / lambda;

    // Process each segment
    // for &(edge.adj_dist, dist, cond, ref similarities) in segment {
    for edge in segment {
        // Skip invalid data points
        if edge.similarities.len() < n_scenario {
            continue;
        }
        
        // Calculate numerator weight with edge.adj_dist
        let weight_num: f32 = {
            let t = edge.adj_dist * inv_lambda;
            let exp_term = (t * t) / DENOM_VAL;
            (-exp_term).exp() * edge.condition
        };
        // Calculate denominator weight with dist
        let weight_denom: f32 = {
            let t = edge.geo_dist * inv_lambda;
            let exp_term = (t * t) / DENOM_VAL;
            (-exp_term).exp()
        };

        // Update numerator values with similarity of scenarios
        // In case the similarity was None, return NAN value for BERI
        for (i, sim_opt) in edge.similarities.iter().take(n_scenario).enumerate() {
            let sim = match sim_opt {
                Some(s) => *s,
                None => 0.0,
            };
            numerator[i] += weight_num * sim;
        }

        // Update denominator using the first similarity value (i.e. current climate)
        denominator += weight_denom * edge.similarities[0].unwrap_or(0.0);
    }
    
    if denominator > 0.0 {
        minimax(&numerator) / denominator
    } else {
        0.0
    }
}
