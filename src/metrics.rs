use crate::graph::Nodes;
use crate::routing::Paths;

// Compute the connectedness from the destination stream and a lambda.
// Destinations are visited in adjusted settlement order; attributes come from each
// destination node.
pub fn connectedness(paths: &Paths, nodes: &Nodes, lambda: f32) -> f32 {
    let (sum_numerator, sum_denominator): (f32, f32) = (0..paths.len())
        .map(|k| {
            let v = paths.node[k] as usize;
            let area = nodes.count[v];
            // `pa` is the destination's protected fraction (1.0 when pa_to_pa is off), applied to
            // the numerator only so the denominator keeps the full reachable area in both cases.
            let numerator = (-(paths.adj[k] / lambda)).exp() * nodes.cond[v] * nodes.pa[v] * area;
            let denominator = (-(paths.geo[k] / lambda)).exp() * area;
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

/// Compute a BERI score from the destination stream and a lambda. `n_scenario` is the number of
/// scenarios (current first) and `numerator` a reusable per-scenario accumulator.
pub fn beri_score(
    paths: &Paths,
    nodes: &Nodes,
    n_scenario: usize,
    lambda: f32,
    numerator: &mut Vec<f32>,
) -> f32 {
    const DENOM_VAL: f32 = 283.465; // 5.785 * (50 - 1)
    
    if paths.is_empty() {
        return 0.0;
    }
    if n_scenario == 0 {
        return 0.0;
    }
    // Reset the numerator vector
    numerator.clear();
    numerator.resize(n_scenario, 0.0);
    let mut denominator = 0.0f32;

    let inv_lambda: f32 = 1.0 / lambda;

    // Process each destination
    for k in 0..paths.len() {
        let v = paths.node[k] as usize;
        // Calculate numerator weight with the adjusted distance
        let area = nodes.count[v];
        let weight_num: f32 = {
            let t = paths.adj[k] * inv_lambda;
            let exp_term = (t * t) / DENOM_VAL;
            (-exp_term).exp() * nodes.cond[v] * area
        };
        // Calculate denominator weight with the intact distance
        let weight_denom: f32 = {
            let t = paths.geo[k] * inv_lambda;
            let exp_term = (t * t) / DENOM_VAL;
            (-exp_term).exp() * area
        };

        // Update numerator values with similarity of scenarios
        // A missing similarity (NO_SIM) counts as 0.0, as `None` did
        let sims = &nodes.sims[v * n_scenario..(v + 1) * n_scenario];
        for (i, &sim) in sims.iter().enumerate() {
            let sim = if sim.is_nan() { 0.0 } else { sim };
            numerator[i] += weight_num * sim;
        }

        // Update denominator using the first similarity value (i.e. current climate)
        let current = if sims[0].is_nan() { 0.0 } else { sims[0] };
        denominator += weight_denom * current;
    }

    if denominator > 0.0 {
        minimax(numerator) / denominator
    } else {
        0.0
    }
}
