/// Same-level queen-neighbour offsets, visited as (di, dj) = (COLS[k], ROWS[k]) as in the
/// legacy builder; this fixes each node's same-level edge order.
pub const COLS: [i32; 8] = [0, 1, 0, -1, 1, 1, -1, -1];
pub const ROWS: [i32; 8] = [1, 0, -1, 0, 1, -1, 1, -1];

/// Missing node or edge index
pub const NONE: u32 = u32::MAX;

/// Similarity of a cell whose transgrid values are not all finite (`None` in the legacy
/// path). BERI treats it as 0.0 in both the numerator and the current-climate denominator.
pub const NO_SIM: f32 = f32::NAN;

/// Window cells of one focal cell in admission order: ascending level, then row-major.
/// Node attributes are flat arrays indexed by node; similarities are [node × scenario].
#[derive(Default)]
pub struct Nodes {
    pub level: Vec<u8>,        // level index
    pub cell: Vec<(i32, i32)>, // tile-local row and column at that level
    pub frac: Vec<f32>,        // area fraction of the cell inside the window
    pub cond: Vec<f32>,        // condition
    pub trav: Vec<f32>,        // traversal value (condition when no traversal raster)
    pub weight: Vec<f32>,      // path weight w = (1 - max_cost) * trav + max_cost
    pub count: Vec<f32>,       // cell count or area weight times the fraction
    pub pa: Vec<f32>,          // protected fraction (1.0 when pa_to_pa is off)
    pub rim: Vec<bool>,        // geometric outer-edge flag (circular windows)
    pub sims: Vec<f32>,        // similarity per scenario; NO_SIM for missing
}

impl Nodes {
    pub fn len(&self) -> usize {
        self.cond.len()
    }

    pub fn clear(&mut self) {
        self.level.clear();
        self.cell.clear();
        self.frac.clear();
        self.cond.clear();
        self.trav.clear();
        self.weight.clear();
        self.count.clear();
        self.pa.clear();
        self.rim.clear();
        self.sims.clear();
    }

    /// Append a copy of node `u` (the synthetic edge's fake target) at `cell`.
    pub fn push_copy(&mut self, u: usize, cell: (i32, i32), n_scen: usize) {
        self.level.push(self.level[u]);
        self.cell.push(cell);
        self.frac.push(self.frac[u]);
        self.cond.push(self.cond[u]);
        self.trav.push(self.trav[u]);
        self.weight.push(self.weight[u]);
        self.count.push(self.count[u]);
        self.pa.push(self.pa[u]);
        self.rim.push(self.rim[u]);
        self.sims.extend_from_within(u * n_scen..(u + 1) * n_scen);
    }

    /// Retained capacity in bytes
    pub fn bytes(&self) -> usize {
        self.level.capacity()
            + self.cell.capacity() * 8
            + self.rim.capacity()
            + 4 * (self.frac.capacity()
                + self.cond.capacity()
                + self.trav.capacity()
                + self.weight.capacity()
                + self.count.capacity()
                + self.pa.capacity()
                + self.sims.capacity())
    }
}

/// A directed edge with both integer costs, the adjusted distance summed along paths and the
/// distance it came from.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Edge {
    pub to: u32,
    pub adj_cost: u32, // (adj * 100).round()
    pub geo_cost: u32, // (dist * 100).round()
    pub adj: f32,      // w * dist, with w of the target node
    pub dist: f32,     // distance in kilometres
}

impl Edge {
    #[inline]
    pub fn new(to: u32, w: f32, dist: f32) -> Self {
        let adj = w * dist;
        Self {
            to,
            // Integer costs exactly as the legacy adjacency: 2-digit precision, saturating casts
            adj_cost: (adj * 100.0).round() as u32,
            geo_cost: (dist * 100.0).round() as u32,
            adj,
            dist,
        }
    }
}

/// Outgoing edges in compressed sparse row form: node u owns edges[offsets[u]..offsets[u + 1]].
#[derive(Default)]
pub struct Csr {
    pub offsets: Vec<u32>,
    pub edges: Vec<Edge>,
}

impl Csr {
    pub fn clear(&mut self) {
        self.offsets.clear();
        self.offsets.push(0);
        self.edges.clear();
    }

    /// Close the current node's edge list.
    #[inline]
    pub fn close(&mut self) {
        self.offsets.push(self.edges.len() as u32);
    }

    pub fn bytes(&self) -> usize {
        self.offsets.capacity() * 4 + self.edges.capacity() * std::mem::size_of::<Edge>()
    }
}

/// Get possible higher-level cells for a current-level edge cell.
///
/// Fractional annulus windows can overlap the higher-level cell containing the
/// current edge cell at the rim, so include it alongside adjacent cells.
pub fn get_edge_neighbours(
    i: i32,
    j: i32,
    level: i32,
    offsets: (usize, usize),
) -> ([(i32, i32); 4], usize) {
    let (tile_row0_u, tile_col0_u) = offsets;
    let tile_row0 = tile_row0_u as i32;
    let tile_col0 = tile_col0_u as i32;

    #[inline]
    fn to_higher_local(idx: i32, level: i32, tile0: i32) -> i32 {
        let curr_origin = tile0.div_euclid(level);
        let higher_origin = tile0.div_euclid(level * 2);
        (curr_origin + idx).div_euclid(2) - higher_origin
    }

    // Higher level cell containing the target cell
    let target_higher = (
        to_higher_local(i, level, tile_row0),
        to_higher_local(j, level, tile_col0),
    );
    // 8 neighbor offsets: N, S, W, E, NW, NE, SW, SE
    const OFFSETS: [(i32, i32); 8] = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ];

    let mut higher_cells = [(0, 0); 4];
    let mut count = 1;
    higher_cells[0] = target_higher;

    for (di, dj) in OFFSETS {
        let ni = i + di;
        let nj = j + dj;
        let higher = (
            to_higher_local(ni, level, tile_row0),
            to_higher_local(nj, level, tile_col0),
        );

        // Skip if it's the same as target's higher cell
        if higher == target_higher {
            continue;
        }

        // Check if we already have this higher cell
        let mut found = false;
        for k in 0..count {
            if higher_cells[k] == higher {
                found = true;
                break;
            }
        }

        // Add if not found and we have space
        if !found && count < higher_cells.len() {
            higher_cells[count] = higher;
            count += 1;
        }
    }

    (higher_cells, count)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn includes_parent_cell() {
        let (cells, count) = get_edge_neighbours(1, 1, 1, (0, 0));
        let cells = &cells[..count];

        assert_eq!(count, 4);
        assert!(cells.contains(&(0, 0)));
        assert!(cells.contains(&(1, 0)));
        assert!(cells.contains(&(0, 1)));
        assert!(cells.contains(&(1, 1)));
    }
}
