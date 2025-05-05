use std::collections::{HashMap, HashSet};
use pathfinding::prelude::dijkstra_all;
use ordered_float::NotNan; // Add this to your Cargo.toml: ordered_float = "3.7.0"

fn main() {
    // Your predefined graph data
    let nodes = vec![1, 12, 14];
    let neighbors = vec![12, 14, 16];
    let costs = vec![1.3, 4.1, 5.2];
    
    // Convert the graph data into an adjacency map
    // HashMap<NodeID, Vec<(NeighborID, Cost)>>
    let mut graph: HashMap<u32, Vec<(u32, NotNan<f32>)>> = HashMap::new();
    
    // Build the graph from the input data
    for i in 0..nodes.len() {
        let node = nodes[i];
        let neighbor = neighbors[i];
        let cost = NotNan::new(costs[i]).unwrap(); // Convert f32 to NotNan<f32> for Ord trait
        
        // Add this edge to the graph
        graph.entry(node)
            .or_insert_with(Vec::new)
            .push((neighbor, cost));
        
        // Uncomment below if the graph is undirected
        // graph.entry(neighbor)
        //     .or_insert_with(Vec::new)
        //     .push((node, cost));
    }
    
    // Create a set of all unique nodes in the graph
    let mut all_nodes: HashSet<u32> = HashSet::new();
    for &node in &nodes {
        all_nodes.insert(node);
    }
    for &neighbor in &neighbors {
        all_nodes.insert(neighbor);
    }
    
    // Create the successors function that uses our adjacency map
    let successors = |node: &u32| -> Vec<(u32, NotNan<f32>)> {
        // Return the neighbors of this node from our graph
        // If the node isn't in our graph, return an empty vector
        match graph.get(node) {
            Some(neighbors) => neighbors.clone(),
            None => Vec::new(),
        }
    };
    
    // Starting node for Dijkstra's algorithm
    let start_node = 1;
    
    // Run Dijkstra's algorithm
    let reachables: HashMap<u32, (u32, NotNan<f32>)> = dijkstra_all(&start_node, successors);
    
    // Print the results
    println!("Distances from node {}:", start_node);
    for node in all_nodes {
        match reachables.get(&node) {
            Some(&(predecessor, cost)) => {
                println!("Node {}: predecessor = {}, cost = {:.2}", 
                         node, predecessor, cost.into_inner());
            },
            None if node == start_node => {
                println!("Node {}: This is the start node", node);
            },
            None => {
                println!("Node {}: Not reachable from start node", node);
            }
        }
    }
}
