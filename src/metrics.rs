use std::f32::consts::E;

// Compute the connectedness from a segment and a lambda
pub fn connectedness(segment: &[(f32, f32, f32, Vec<f32>)], lambda: f32) -> f32 {
    let sum_conn: f32 = segment
        .iter()
        .map(|(dist_adj, dist, condition, _)| {
            let numerator = E.powf(- (dist_adj / lambda)) * condition;
            let denominator = E.powf(- (dist / lambda));
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        })
        .sum();

    let len_conn: f32 = segment.len() as f32;

    if len_conn > 0.0 {
        sum_conn / len_conn
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

    // Process each segment
    for (dist_adj, dist, cond, similarities) in segment {
        // Skip invalid data points
        if similarities.len() < n_scenario {
            continue;
        }
        
        // Calculate numerator weight with dist_adj
        let dist_lambda_num = dist_adj / lambda;
        let exp_term_num = dist_lambda_num * dist_lambda_num / DENOM_VAL;
        let weight_num = E.powf(-exp_term_num) * cond;
        
        // Update numerator values with similarity of scenarios
        for (i, &sim) in similarities.iter().take(n_scenario).enumerate() {
            numerator[i] += weight_num * sim;
        }
        
        // Calculate denominator weight with dist
        let dist_lambda_denom = dist / lambda;
        let exp_term_denom = dist_lambda_denom * dist_lambda_denom / DENOM_VAL;
        let weight_denom = E.powf(-exp_term_denom);
        
        // Update denominator using the first similarity value (i.e. current climate)
        denominator += weight_denom * similarities[0];
    }
    
    if denominator > 0.0 {
        minimax(&numerator) / denominator
    } else {
        0.0
    }
}

