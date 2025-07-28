use std::f32::consts::E;

// Compute the connectedness from a segment and a lambda
pub fn connectedness(segment: &[(f32, f32, f32, Vec<f32>)], lambda: f32) -> f32 {
    let (sum_numerator, sum_denominator): (f32, f32) = segment
        .iter()
        .map(|(dist_adj, dist, condition, _)| {
            let numerator = E.powf(-(dist_adj / lambda)) * condition;
            let denominator = E.powf(-(dist / lambda));
            (numerator, denominator)
        })
        .fold((0.0, 0.0), |(acc_num, acc_den), (num, den)| {
            (acc_num + num, acc_den + den)
        });

    if sum_denominator > 0.0  {
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
pub fn beri_score(segment: &[(f32, f32, f32, Vec<f32>)], lambda: f32) -> f32 {
    const DENOM_VAL: f32 = 283.465; // 5.785 * (50 - 1)
    
    if segment.is_empty() {
        return 0.0;
    }

    // Get number of scenarios (including current)
    let n_scenario = segment[0].3.len();
    if n_scenario == 0 {
        return 0.0;
    }
    // Initialize numerator vector with capacity
    let mut numerator = vec![0.0f32; n_scenario];
    let mut denominator = 0.0f32;

    let inv_lambda = 1.0 / lambda;

    // Process each segment
    for (dist_adj, dist, cond, ref similarities) in segment {
        // Skip invalid data points
        if similarities.len() < n_scenario {
            continue;
        }
        
        // Calculate numerator weight with dist_adj
        let weight_num = {
            let t = dist_adj * inv_lambda;
            let exp_term = (t * t) / DENOM_VAL;
            cond * (-exp_term).exp()
        };
        // Calculate denominator weight with dist
        let weight_denom = {
            let t = dist * inv_lambda;
            let exp_term = (t * t) / DENOM_VAL;
            (-exp_term).exp()
        };
        
        // Update numerator values with similarity of scenarios
        for (i, &sim) in similarities.iter().take(n_scenario).enumerate() {
            numerator[i] += weight_num * sim;
        }

        // Update denominator using the first similarity value (i.e. current climate)
        denominator += weight_denom * similarities[0];
    }
    
    if denominator > 0.0 {
        minimax(&numerator) / denominator
    } else {
        0.0
    }
}

