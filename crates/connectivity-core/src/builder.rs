use crate::affine::Affine;
use crate::distances;
use crate::graph::{Csr, Edge, Nodes, COLS, NONE, NO_SIM, ROWS};
use crate::metrics;
use crate::routing::Router;
use crate::tables::{Tables, BOUNDED, STATUSES, SYNTHETIC};
use crate::template::Templates;
use crate::utils;
use crate::window::{circle_frac, similarities, similarity, square_frac, WindowMode};
use ndarray::{s, Array2, Array3, ArrayView1};

/// Level-indexed overviews and settings shared by every focal cell of one call.
pub struct Context<'a> {
    pub levels: Vec<i32>, // ascending
    pub cond: Vec<Array2<f32>>,
    pub weight: Vec<Array2<f32>>, // cell count or area weight
    pub pa: Option<Vec<Array2<f32>>>,
    pub trav: Option<Vec<Array2<f32>>>,
    pub trans: Vec<Vec<Array3<f32>>>, // [scenario][level]
    pub contiguous: bool,             // every transgrid overview has standard layout
    pub base: Option<usize>,          // index of level 1
    pub transforms: Vec<Option<Affine>>,
    pub fringe: Vec<bool>, // links to the next level (it is twice this one)
    pub clip: Vec<[bool; 4]>, // first/last row and column cells are cut by the tile edge
    pub templates: Templates,
    pub tables: Tables,
    pub base_shape: (i32, i32),
    pub offsets: (usize, usize),
    pub mode: WindowMode,
    pub is_geo: bool,
    pub factor: f32, // max_cost
    pub pa_to_pa: bool,
    pub lambdas: &'a [f32],
    pub run_beri: bool,
}

impl Context<'_> {
    pub fn n_scen(&self) -> usize {
        self.trans.len()
    }

    /// Current-climate transgrid of the base level (`utils::get_current`'s source)
    fn current(&self) -> Option<&Array3<f32>> {
        self.trans.first().zip(self.base).map(|(scen, l)| &scen[l])
    }

    /// Level-l area fraction and outer-edge flag of local cell (li, lj) whose footprint is cut by
    /// the tile edge, computed as `collect_annulus` does on the clipped footprint.
    fn clipped(&self, l: usize, li: i32, lj: i32, base_i: i32, base_j: i32) -> (f32, bool) {
        let shape = &self.templates.shapes[l];
        let level = shape.level;
        let (tile_row0, tile_col0) = (self.offsets.0 as i32, self.offsets.1 as i32);
        let tile_r0 = tile_row0 as f64;
        let tile_c0 = tile_col0 as f64;
        let tile_r1 = (tile_row0 + self.base_shape.0) as f64;
        let tile_c1 = (tile_col0 + self.base_shape.1) as f64;
        let source_r = tile_row0 as f64 + base_i as f64 + 0.5;
        let source_c = tile_col0 as f64 + base_j as f64 + 0.5;

        let global_i = tile_row0.div_euclid(level) + li;
        let cell_r0 = (global_i * level) as f64;
        let cell_r1 = cell_r0 + level as f64;
        let global_j = tile_col0.div_euclid(level) + lj;
        let cell_c0 = (global_j * level) as f64;
        let cell_c1 = cell_c0 + level as f64;
        let (r0, r1) = (cell_r0.max(tile_r0), cell_r1.min(tile_r1));
        let (c0, c1) = (cell_c0.max(tile_c0), cell_c1.min(tile_c1));
        match self.mode {
            WindowMode::Circular => circle_frac(r0, r1, c0, c1, source_r, source_c, shape.inner, shape.outer),
            WindowMode::Square => (
                square_frac(r0, r1, c0, c1, source_r, source_c, shape.inner, shape.outer),
                false,
            ),
        }
    }
}

/// Dense lookup over the search box's cells inside the tile, in local cells of one level; NONE
/// marks cells that are not nodes.
#[derive(Default)]
struct Grid {
    origin: (i32, i32),
    shape: (i32, i32),
    index: Vec<u32>,
}

impl Grid {
    fn reset(&mut self, origin: (i32, i32), shape: (i32, i32)) {
        self.origin = origin;
        self.shape = shape;
        self.index.clear();
        self.index.resize((shape.0 * shape.1) as usize, NONE);
    }

    #[inline]
    fn get(&self, i: i32, j: i32) -> u32 {
        let (r, c) = (i - self.origin.0, j - self.origin.1);
        if r < 0 || c < 0 || r >= self.shape.0 || c >= self.shape.1 {
            return NONE;
        }
        self.index[(r * self.shape.1 + c) as usize]
    }

    #[inline]
    fn set(&mut self, i: i32, j: i32, node: u32) {
        let (r, c) = (i - self.origin.0, j - self.origin.1);
        self.index[(r * self.shape.1 + c) as usize] = node;
    }
}

/// Reusable per-thread state for one focal cell at a time: nodes, CSR, routing and metrics.
#[derive(Default)]
pub struct Scratch {
    pub nodes: Nodes,
    pub csr: Csr,
    pub router: Router,
    pub ranges: Vec<(u32, u32)>, // node range of each level
    bounds: Vec<(i32, i32, i32, i32)>, // admitted rows and columns of each level
    grids: Vec<Grid>,          // node of each searched cell, per level
    first: Vec<(i32, i32)>,    // local cell of each level's first box candidate
    src: Vec<(i32, i32)>,      // local cell of the source at each level
    parity: Vec<usize>,        // parent-grid parity of the source's cell, per level
    numerator: Vec<f32>,       // BERI scenario accumulator
    pub counts: [u64; STATUSES], // edges by distance-table status
    pub source: u32,           // NONE when the focal cell is not in its own window
    pub isolated: bool,
}

impl Scratch {
    /// Scratch sized for this call's largest windows inside the tile, so cells never allocate.
    pub fn new(ctx: &Context) -> Self {
        let mut scratch = Self::default();
        let n_levels = ctx.levels.len();
        let mut nodes = 0;
        for l in 0..n_levels {
            // A level admits at most the search box's cells inside the tile's level grid
            let (height, width) = ctx.cond[l].dim();
            let cells = ctx.templates.shapes[l].len().min(height * width);
            let mut grid = Grid::default();
            grid.index.reserve(cells);
            scratch.grids.push(grid);
            nodes += cells;
        }
        scratch.first = vec![(0, 0); n_levels];
        scratch.src = vec![(0, 0); n_levels];
        scratch.parity = vec![0; n_levels];
        let n = &mut scratch.nodes;
        n.level.reserve(nodes + 1);
        n.cell.reserve(nodes + 1);
        n.frac.reserve(nodes + 1);
        n.cond.reserve(nodes + 1);
        n.trav.reserve(nodes + 1);
        n.weight.reserve(nodes + 1);
        n.count.reserve(nodes + 1);
        n.pa.reserve(nodes + 1);
        n.rim.reserve(nodes + 1);
        n.sims.reserve((nodes + 1) * ctx.n_scen());
        scratch.csr.offsets.reserve(nodes + 2);
        scratch.csr.edges.reserve(12 * nodes);
        scratch.router.reserve(nodes + 1, 12 * nodes);
        scratch.ranges.reserve(n_levels);
        scratch.bounds.reserve(n_levels);
        scratch.numerator.reserve(ctx.n_scen());
        scratch
    }

    /// Retained capacity in bytes
    pub fn bytes(&self) -> usize {
        self.nodes.bytes()
            + self.csr.bytes()
            + self.router.bytes()
            + self.ranges.capacity() * 8
            + self.bounds.capacity() * 16
            + self.grids.iter().map(|g| g.index.capacity() * 4).sum::<usize>()
            + self.grids.capacity() * std::mem::size_of::<Grid>()
            + (self.first.capacity() + self.src.capacity()) * 8
            + self.parity.capacity() * 8
            + self.numerator.capacity() * 4
            + std::mem::size_of_val(&self.counts)
    }

    /// Indicator value of focal cell ij.
    pub fn eval(&mut self, ctx: &Context, i: usize, j: usize) -> f32 {
        self.build(ctx, i, j);
        if self.source == NONE || (self.isolated && ctx.pa_to_pa) {
            // No destinations: the legacy graph searched from a phantom node, or was emptied
            self.router.clear();
        } else {
            self.router.route(&self.csr, self.source);
        }
        self.score(ctx)
    }

    /// Nodes and edges of focal cell ij.
    pub fn build(&mut self, ctx: &Context, i: usize, j: usize) {
        let (bi, bj) = (i as i32, j as i32);
        self.nodes.clear();
        self.csr.clear();
        self.ranges.clear();
        self.bounds.clear();
        let current = utils::get_current(ctx.current(), i, j);
        for l in 0..ctx.levels.len() {
            self.admit(ctx, l, bi, bj);
        }
        if !ctx.trans.is_empty() {
            self.similarities(ctx, &current);
        }

        // The focal cell must be in its own first-level window to be a source
        self.source = self.grids[0].get(bi, bj);
        self.isolated = self.source != NONE
            && (0..8).all(|k| self.grids[0].get(bi + COLS[k], bj + ROWS[k]) == NONE);
        if self.isolated {
            self.synthetic(ctx, bi, bj);
        } else {
            self.link(ctx);
        }
    }

    /// Admit level `l`'s window cells in template (row-major) order, as `collect_annulus` does.
    fn admit(&mut self, ctx: &Context, l: usize, base_i: i32, base_j: i32) {
        let shape = &ctx.templates.shapes[l];
        let level = shape.level;
        let start = self.nodes.len() as u32;
        let mut bounds = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);

        // Source position: phase inside its level cell, parent-grid parity and local cell
        let global = (ctx.offsets.0 as i32 + base_i, ctx.offsets.1 as i32 + base_j);
        let phase = (global.0.rem_euclid(level), global.1.rem_euclid(level));
        let cell = (global.0.div_euclid(level), global.1.div_euclid(level));
        self.parity[l] = (cell.0.rem_euclid(2) * 2 + cell.1.rem_euclid(2)) as usize;
        let origin = ((ctx.offsets.0 as i32).div_euclid(level), (ctx.offsets.1 as i32).div_euclid(level));
        self.src[l] = (cell.0 - origin.0, cell.1 - origin.1);
        let first = (self.src[l].0 + shape.start, self.src[l].1 + shape.start);
        self.first[l] = first;

        // Box candidates inside the tile's level grid, row-major
        let (height, width) = (ctx.cond[l].nrows() as i32, ctx.cond[l].ncols() as i32);
        let rows = (-first.0).max(0)..shape.side.min(height - first.0);
        let cols = (-first.1).max(0)..shape.side.min(width - first.1);
        let extent = ((rows.end - rows.start).max(0), (cols.end - cols.start).max(0));
        self.grids[l].reset((first.0 + rows.start, first.1 + cols.start), extent);

        // Geometry of this phase: cached, or computed cell by cell with the same function
        let cached = ctx.templates.geom(l, phase);
        let clip = ctx.clip[l];
        for r in rows {
            let li = first.0 + r;
            let row_clipped = (li == 0 && clip[0]) || (li == height - 1 && clip[1]);
            for q in cols.clone() {
                let lj = first.1 + q;
                let clipped = row_clipped || (lj == 0 && clip[2]) || (lj == width - 1 && clip[3]);
                let (fraction, rim) = if clipped {
                    ctx.clipped(l, li, lj, base_i, base_j)
                } else if let Some((frac, rims)) = cached {
                    let c = (r * shape.side + q) as usize;
                    (frac[c], rims[c])
                } else {
                    shape.cell_geom(ctx.mode, phase, r, q)
                };
                if fraction <= 0.0 {
                    continue;
                }
                if self.push(ctx, l, [li as usize, lj as usize], fraction, rim) {
                    self.grids[l].set(li, lj, self.nodes.len() as u32 - 1);
                    bounds = (bounds.0.min(li), bounds.1.max(li), bounds.2.min(lj), bounds.3.max(lj));
                }
            }
        }
        self.ranges.push((start, self.nodes.len() as u32));
        self.bounds.push(bounds);
    }

    /// Admit one cell when its condition and weight are usable; returns whether it was added.
    #[inline]
    fn push(&mut self, ctx: &Context, l: usize, at: [usize; 2], fraction: f32, rim: bool) -> bool {
        let condition = ctx.cond[l][at];
        if condition.is_nan() {
            return false;
        }
        let weight = ctx.weight[l][at];
        if !weight.is_finite() {
            return false;
        }

        // PARC target-gating: protected fraction of this destination cell (1.0 when gating is
        // off), applied to the connectedness numerator only (see metrics.rs).
        let pa_value = match &ctx.pa {
            Some(pa) => {
                let f = pa[l][at];
                if f.is_finite() {
                    f
                } else {
                    0.0
                }
            }
            None => 1.0,
        };
        // Traversal value falls back to the condition value if resistance is missing or
        // non-finite here, so admitted cells always have a finite traversal value.
        let t = match &ctx.trav {
            Some(trav) => Some(trav[l][at]).filter(|v| v.is_finite()).unwrap_or(condition),
            None => condition,
        };

        let n = &mut self.nodes;
        n.level.push(l as u8);
        n.cell.push((at[0] as i32, at[1] as i32));
        n.frac.push(fraction);
        n.cond.push(condition);
        n.trav.push(t);
        n.weight.push((1.0 - ctx.factor) * t + ctx.factor);
        n.count.push(weight * fraction);
        n.pa.push(pa_value);
        n.rim.push(rim);
        true
    }

    /// Similarity of every node with the focal cell's current climate, for every scenario.
    /// Kept out of admission so connectedness runs do not carry it.
    #[inline(never)]
    fn similarities(&mut self, ctx: &Context, current: &ArrayView1<f32>) {
        let n = &mut self.nodes;
        n.sims.clear();
        for u in 0..n.level.len() {
            let l = n.level[u] as usize;
            let at = [n.cell[u].0 as usize, n.cell[u].1 as usize];
            match current.as_slice() {
                // Contiguous bands: all scenarios of the cell together, eight at a time
                Some(current) if ctx.contiguous => {
                    for group in ctx.trans.chunks(8) {
                        let mut cells: [&[f32]; 8] = [&[]; 8];
                        for (cell, scenario) in cells.iter_mut().zip(group) {
                            let (_, width, bands) = scenario[l].dim();
                            let start = (at[0] * width + at[1]) * bands;
                            *cell = &scenario[l].as_slice().unwrap()[start..start + bands];
                        }
                        similarities(current, &cells[..group.len()], NO_SIM, &mut n.sims);
                    }
                }
                _ => {
                    for scenario in &ctx.trans {
                        let seg_val = scenario[l].slice(s![at[0], at[1], ..]);
                        n.sims.push(similarity(current, &seg_val).unwrap_or(NO_SIM));
                    }
                }
            }
        }
    }

    /// Edges of every node in node order: same-level queen neighbours in COLS/ROWS order, then
    /// links to the next level in `get_edge_neighbours` order for fringe-eligible nodes.
    fn link(&mut self, ctx: &Context) {
        let shapes = &ctx.templates.shapes;
        for l in 0..ctx.levels.len() {
            let shape = &shapes[l];
            let transform = ctx.transforms[l].expect("Missing level in Affine set.");
            // Non-continuous levels have no node mapping in the legacy builder
            let (start, end) = self.ranges[l];
            assert!(
                l == 0 || ctx.fringe[l - 1] || start == end,
                "levels must form a continuous power-of-two sequence"
            );
            let fringe = ctx.fringe[l];
            let (i_min, i_max, j_min, j_max) = self.bounds[l];
            let (first, parity) = (self.first[l], self.parity[l]);
            let up = if fringe { self.src[l + 1] } else { (0, 0) };

            for u in start as usize..end as usize {
                let (i, j) = self.nodes.cell[u];
                // Source coordinates only when an edge falls back to the legacy distance
                let mut xy1 = None;
                for k in 0..8 {
                    let v = self.grids[l].get(i + COLS[k], j + ROWS[k]);
                    if v != NONE {
                        let (d, status) = ctx.tables.get(l, i, k);
                        self.counts[status as usize] += 1;
                        let dist = if status <= BOUNDED {
                            d
                        } else {
                            let (x1, y1) = *xy1.get_or_insert_with(|| transform.xy(i, j));
                            let (ni, nj) = self.nodes.cell[v as usize];
                            let (x2, y2) = transform.xy(ni, nj);
                            distances::distance_km(x1, y1, x2, y2, ctx.is_geo)
                        };
                        let w = self.nodes.weight[v as usize];
                        self.csr.edges.push(Edge::new(v, w, dist));
                    }
                }

                let eligible = match ctx.mode {
                    WindowMode::Circular => self.nodes.rim[u],
                    WindowMode::Square => i == i_min || i == i_max || j == j_min || j == j_max,
                };
                if fringe && eligible {
                    let upper = ctx.transforms[l + 1].unwrap();
                    let link = shape.links(parity, i - first.0, j - first.1);
                    for m in 0..link.count as usize {
                        let target = (up.0 + link.cell[m].0, up.1 + link.cell[m].1);
                        let v = self.grids[l + 1].get(target.0, target.1);
                        if v != NONE {
                            let (d, status) = ctx.tables.get(l, i, 8 + link.slot[m] as usize);
                            self.counts[status as usize] += 1;
                            let dist = if status <= BOUNDED {
                                d
                            } else {
                                let (x1, y1) = *xy1.get_or_insert_with(|| transform.xy(i, j));
                                let (hi, hj) = self.nodes.cell[v as usize];
                                let (x2, y2) = upper.xy(hi, hj);
                                distances::distance_km(x1, y1, x2, y2, ctx.is_geo)
                            };
                            let w = self.nodes.weight[v as usize];
                            self.csr.edges.push(Edge::new(v, w, dist));
                        }
                    }
                }
                self.csr.close();
            }
        }
    }

    /// An isolated source keeps only a synthetic edge to a fake copy of itself one row down,
    /// or no edge at all under PA target-gating.
    fn synthetic(&mut self, ctx: &Context, bi: i32, bj: i32) {
        let transform = ctx.transforms[0].expect("Missing level in Affine set.");
        if ctx.pa_to_pa {
            return;
        }
        let s = self.source as usize;
        // Distance to the next row's centre, also outside the raster
        let (d, status) = ctx.tables.get(0, bi, SYNTHETIC);
        self.counts[status as usize] += 1;
        let dist = if status <= BOUNDED {
            d
        } else {
            let (x1, y1) = transform.xy(bi, bj);
            let (x2, y2) = transform.xy(bi + 1, bj);
            distances::distance_km(x1, y1, x2, y2, ctx.is_geo)
        };
        // The legacy fake node id is the source's + 1, i.e. the next column
        let fake = self.nodes.len() as u32;
        self.nodes.push_copy(s, (bi, bj + 1), ctx.n_scen());
        for u in 0..self.nodes.len() {
            if u == s {
                self.csr.edges.push(Edge::new(fake, self.nodes.weight[s], dist));
            }
            self.csr.close();
        }
    }

    /// BERI or connectedness of the routed destinations, averaged over lambdas.
    fn score(&mut self, ctx: &Context) -> f32 {
        if ctx.lambdas.is_empty() {
            0.0
        } else {
            let (paths, nodes, numerator) = (&self.router.paths, &self.nodes, &mut self.numerator);
            let sum: f32 = ctx
                .lambdas
                .iter()
                .map(|&lambda| {
                    if ctx.run_beri {
                        metrics::beri_score(paths, nodes, ctx.n_scen(), lambda, numerator)
                    } else {
                        metrics::connectedness(paths, nodes, lambda)
                    }
                })
                .sum();

            sum / ctx.lambdas.len() as f32
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::core::prepare;
    use crate::template::link_slot;
    use rayon::prelude::*;
    use crate::legacy::canonical::{edge_order, encode};
    use crate::legacy::Prepared;
    use crate::testkit::Fixture;

    pub fn context(f: &Fixture) -> Context<'_> {
        prepare(&f.inputs(), &f.config()).unwrap()
    }

    /// Legacy node id of optimized node u.
    pub fn node_id(ctx: &Context, s: &Scratch, u: usize) -> u64 {
        let level = ctx.levels[s.nodes.level[u] as usize];
        let (i, j) = s.nodes.cell[u];
        let origin = |tile0: usize| (tile0 as i32).div_euclid(level);
        encode(level, origin(ctx.offsets.0) + i, origin(ctx.offsets.1) + j).unwrap()
    }

    fn bits(sim: Option<f32>) -> u32 {
        sim.unwrap_or(NO_SIM).to_bits()
    }

    /// Per-edge distance verification counts.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct Tally {
        pub edges: u64,
        pub status: [u64; STATUSES], // table hits, then fallback reasons
        pub ulp1: u64,               // distances one f32 ulp from the legacy value
        pub ulp_more: u64,           // distances further away
        pub cost_mismatch: u64,      // adjusted or intact integer costs that differ
    }

    impl Tally {
        pub fn add(&mut self, other: &Tally) {
            self.edges += other.edges;
            self.status.iter_mut().zip(other.status).for_each(|(a, b)| *a += b);
            self.ulp1 += other.ulp1;
            self.ulp_more += other.ulp_more;
            self.cost_mismatch += other.cost_mismatch;
        }
    }

    /// Distance slot of edge u -> v derived from node cells, independently of templates.
    pub fn edge_slot(ctx: &Context, s: &Scratch, u: usize, v: usize) -> usize {
        if s.isolated {
            return SYNTHETIC;
        }
        let (lu, lv) = (s.nodes.level[u] as usize, s.nodes.level[v] as usize);
        let ((i, j), (ni, nj)) = (s.nodes.cell[u], s.nodes.cell[v]);
        if lu == lv {
            return (0..8).find(|&k| (COLS[k], ROWS[k]) == (ni - i, nj - j)).unwrap();
        }
        let origin = |level: i32| ((ctx.offsets.0 as i32).div_euclid(level), (ctx.offsets.1 as i32).div_euclid(level));
        let (o, ou) = (origin(ctx.levels[lu]), origin(ctx.levels[lv]));
        let (g, h) = ((o.0 + i, o.1 + j), (ou.0 + ni, ou.1 + nj));
        let parity = (g.0.rem_euclid(2), g.1.rem_euclid(2));
        8 + link_slot(parity, (h.0 - g.0.div_euclid(2), h.1 - g.1.div_euclid(2))) as usize
    }

    /// Compare one focal cell against the canonical reference; returns the edge count.
    pub fn compare_cell(ctx: &Context, prepared: &Prepared, s: &mut Scratch, i: usize, j: usize) -> usize {
        let mut tally = Tally::default();
        compare_tally(ctx, prepared, s, i, j, &mut tally)
    }

    /// `compare_cell`, also counting table hits, fallbacks and distance differences.
    pub fn compare_tally(
        ctx: &Context,
        prepared: &Prepared,
        s: &mut Scratch,
        i: usize,
        j: usize,
        tally: &mut Tally,
    ) -> usize {
        let at = format!("cell ({i}, {j})");
        let value = s.eval(ctx, i, j);
        let n_scen = ctx.n_scen();

        // Nodes: legacy window cells in level order, then row-major
        let windows = prepared.windows(i, j);
        let mut u = 0;
        for (l, (level, win)) in windows.iter().enumerate() {
            assert_eq!(*level, ctx.levels[l]);
            for idx in 0..win.i_array.len() {
                assert_eq!(s.nodes.level[u] as usize, l, "{at}");
                assert_eq!(s.nodes.cell[u], (win.i_array[idx], win.j_array[idx]), "{at}");
                let trav = win.trav_values.as_ref().map_or(win.values[idx], |t| t[idx]);
                let expected = [win.fractions[idx], win.values[idx], trav, win.counts[idx], win.pa_values[idx]];
                let actual = [s.nodes.frac[u], s.nodes.cond[u], s.nodes.trav[u], s.nodes.count[u], s.nodes.pa[u]];
                assert_eq!(expected.map(f32::to_bits), actual.map(f32::to_bits), "{at}, node {u}");
                for scen in 0..n_scen {
                    assert_eq!(bits(win.sims[scen][idx]), s.nodes.sims[u * n_scen + scen].to_bits(), "{at}");
                }
                u += 1;
            }
        }
        let real_nodes = u;
        assert!(s.nodes.len() == real_nodes || (s.isolated && s.nodes.len() == real_nodes + 1));

        // Directed edges in canonical order, with distances and both integer costs
        let graph = prepared.graph(i, j, &windows);
        let keys = edge_order(&graph, ctx.offsets);
        let mut ordered = Vec::new();
        if s.source != NONE && !(s.isolated && ctx.pa_to_pa) {
            for u in 0..s.nodes.len() {
                let (start, end) = (s.csr.offsets[u] as usize, s.csr.offsets[u + 1] as usize);
                for edge in &s.csr.edges[start..end] {
                    let key = (node_id(ctx, s, u), node_id(ctx, s, edge.to as usize));
                    let legacy = &graph.data[&key];
                    let level = if s.isolated { 0 } else { s.nodes.level[u] as usize };
                    let slot = edge_slot(ctx, s, u, edge.to as usize);
                    tally.status[ctx.tables.get(level, s.nodes.cell[u].0, slot).1 as usize] += 1;
                    tally.edges += 1;
                    let ulps = (edge.dist.to_bits() as i64 - legacy.geo_dist.to_bits() as i64).abs();
                    tally.ulp1 += (ulps == 1) as u64;
                    tally.ulp_more += (ulps > 1) as u64;
                    let adj_cost = (legacy.adj_dist * 100.0).round() as u32;
                    let geo_cost = (legacy.geo_dist * 100.0).round() as u32;
                    tally.cost_mismatch += (edge.adj_cost != adj_cost || edge.geo_cost != geo_cost) as u64;
                    assert_eq!(
                        [edge.adj, edge.dist].map(f32::to_bits),
                        [legacy.adj_dist, legacy.geo_dist].map(f32::to_bits),
                        "{at}, edge {key:?}"
                    );
                    assert_eq!(edge.adj_cost, (legacy.adj_dist * 100.0).round() as u32);
                    assert_eq!(edge.geo_cost, (legacy.geo_dist * 100.0).round() as u32);
                    ordered.push(key);
                }
            }
            assert_eq!(ordered, keys, "{at}: edge order");
        } else if s.source == NONE {
            // The legacy graph still has its edges but routes from a phantom node
            assert_eq!(graph.source, 0, "{at}");
        } else {
            assert!(graph.data.is_empty(), "{at}");
        }

        // Both trees, path sums and the final value
        let routed = graph.route(graph.to_adjacency_in(&keys));
        let ids = &routed.adjacency.nodes;
        if !ordered.is_empty() {
            for (tree, legacy) in [(&s.router.altered, &routed.altered), (&s.router.intact, &routed.intact)] {
                let order: Vec<u64> = tree.order.iter().map(|&v| node_id(ctx, s, v as usize)).collect();
                let expected: Vec<u64> = legacy.order.iter().map(|&v| ids[v]).collect();
                assert_eq!(order, expected, "{at}: settlement order");
                for (&v, &lv) in tree.order.iter().zip(&legacy.order).skip(1) {
                    let parent = node_id(ctx, s, tree.parent[v as usize] as usize);
                    assert_eq!(parent, ids[legacy.parent[lv]], "{at}: parent");
                    assert_eq!(tree.cost[v as usize], legacy.cost[lv], "{at}: path cost");
                }
            }
        }
        let paths = &s.router.paths;
        assert_eq!(paths.len(), routed.paths.len(), "{at}: destinations");
        for (k, legacy) in routed.paths.iter().enumerate() {
            let v = paths.node[k] as usize;
            assert_eq!(
                [paths.adj[k], paths.geo[k], s.nodes.cond[v], s.nodes.pa[v], s.nodes.count[v]].map(f32::to_bits),
                [legacy.adj_dist, legacy.geo_dist, legacy.condition, legacy.pa, legacy.num_cells].map(f32::to_bits),
                "{at}: destination {k}"
            );
            for scen in 0..n_scen {
                assert_eq!(bits(legacy.similarities[scen]), s.nodes.sims[v * n_scen + scen].to_bits());
            }
        }
        assert_eq!(value.to_bits(), prepared.score(&routed.paths).to_bits(), "{at}: value");
        ordered.len()
    }

    pub fn compare_fixture(fixture: &Fixture) -> usize {
        compare_ctx(fixture, &context(fixture))
    }

    pub fn compare_ctx(fixture: &Fixture, ctx: &Context) -> usize {
        let prepared = fixture.prepared();
        let mut scratch = Scratch::new(ctx);
        let mut edges = 0;
        for ((i, j), &masked) in fixture.mask.indexed_iter() {
            if !masked {
                edges += compare_cell(ctx, &prepared, &mut scratch, i, j);
            }
        }
        edges
    }

    #[test]
    fn matches_canonical() {
        let mut edges = 0;
        for seed in 0..300 {
            edges += compare_fixture(&Fixture::random(seed));
        }
        assert!(edges > 100_000, "{edges}");
    }

    #[test]
    fn multilevel_canonical() {
        for mode in [WindowMode::Circular, WindowMode::Square] {
            for beri in [false, true] {
                for pa_to_pa in [false, true] {
                    compare_fixture(&Fixture::multilevel(mode, beri, pa_to_pa));
                }
            }
        }
    }

    #[test]
    fn focused_cases() {
        for seed in 0..40 {
            for fixture in Fixture::focused(seed) {
                compare_fixture(&fixture);
            }
        }
    }

    // Bit-identical output for 1, 2 and all available threads, equal to the canonical reference
    #[test]
    fn thread_counts() {
        for seed in [3, 8, 21, 42, 77] {
            let f = Fixture::random(seed);
            let canonical = f.canonical(None);
            for threads in [1, 2, num_cpus::get().max(3)] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                let out = pool.install(|| crate::core::conn(&f.inputs(), &f.config()).unwrap());
                assert!(crate::testkit::same_bits(&out, &canonical), "seed {seed}, {threads} threads");
            }
        }
    }

    // After one pass over a fixture, evaluating its cells again allocates nothing
    #[test]
    fn steady_state_allocs() {
        use crate::testkit::alloc;
        for seed in [1, 5, 9, 14, 30] {
            let f = Fixture::random(seed);
            let ctx = context(&f);
            let mut scratch = Scratch::new(&ctx);
            let cells: Vec<(usize, usize)> =
                f.mask.indexed_iter().filter(|(_, &m)| !m).map(|(c, _)| c).collect();
            let first = alloc::count();
            for &(i, j) in &cells {
                scratch.eval(&ctx, i, j);
            }
            let growth = alloc::count() - first;
            let start = alloc::count();
            for &(i, j) in &cells {
                scratch.eval(&ctx, i, j);
            }
            assert_eq!(alloc::count() - start, 0, "seed {seed}: steady-state allocations");
            println!("seed {seed}: {} cells, {growth} growth allocations on the first pass", cells.len());
        }
    }

    /// Rebuild a context's templates with another budget and, optionally, build order.
    pub fn retemplate(ctx: &mut Context, f: &Fixture, budget: usize, order: Option<&[(usize, usize)]>) {
        let n = ctx.levels.len();
        let outer = if f.outer_window < f.window_size { f.window_size } else { f.outer_window };
        let effective: Vec<i32> = (0..n).map(|l| if l + 1 == n { outer } else { f.window_size }).collect();
        ctx.templates = Templates::new(
            &ctx.levels, &effective, f.window_size, f.mode, &f.mask, f.offsets, budget, order,
        );
    }

    fn evaluate(ctx: &Context, f: &Fixture) -> Vec<u32> {
        let mut scratch = Scratch::new(ctx);
        f.mask
            .indexed_iter()
            .filter(|(_, &m)| !m)
            .map(|((i, j), _)| scratch.eval(ctx, i, j).to_bits())
            .collect()
    }

    /// Distance tables and templates for a context built on one thread as `prepare` does, with
    /// the heap they retain and the peak heap while building, both counted by the allocator.
    pub fn measured(ctx: &Context, f: &Fixture, budget: usize) -> (Tables, Templates, usize, usize) {
        use crate::testkit::alloc;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            // Warm rayon's own buffers on this thread first
            (0..4096).into_par_iter().for_each(|k| _ = std::hint::black_box(k));
            let n = ctx.levels.len();
            let outer = if f.outer_window < f.window_size { f.window_size } else { f.outer_window };
            let effective: Vec<i32> = (0..n).map(|l| if l + 1 == n { outer } else { f.window_size }).collect();
            let dims: Vec<(usize, usize)> = ctx.cond.iter().map(|a| a.dim()).collect();
            let before = alloc::live();
            alloc::reset_peak();
            let reserve = budget.saturating_sub(Templates::fixed(n));
            let tables = Tables::new(&ctx.levels, &ctx.transforms, &dims, ctx.offsets, &ctx.fringe, ctx.is_geo, reserve);
            let rest = budget.saturating_sub(tables.bytes);
            let templates = Templates::new(&ctx.levels, &effective, f.window_size, f.mode, &f.mask, ctx.offsets, rest, None);
            let retained = (alloc::live() - before) as usize;
            let peak = (alloc::peak() - before) as usize;
            (tables, templates, retained, peak)
        })
    }

    // Allocator-measured cache memory equals the reported retained bytes, and the peak while
    // building stays within the reported peak and the budget, from no budget to unlimited
    #[test]
    fn cache_memory() {
        for seed in 0..20 {
            let f = Fixture::random(seed);
            let ctx = context(&f);
            let full = ctx.templates.bytes + ctx.tables.bytes;
            for budget in [0, 1 << 10, full / 3, full / 2, full, usize::MAX] {
                let (tables, templates, retained, peak) = measured(&ctx, &f, budget);
                let fixed = tables.fixed + templates.fixed;
                let at = format!("seed {seed}, budget {budget}");
                assert_eq!(retained, tables.bytes + templates.bytes, "{at}");
                assert!(peak <= tables.bytes + templates.peak, "{at}: measured peak {peak}");
                assert!(peak <= budget.max(fixed), "{at}: peak {peak}, fixed {fixed}");
            }
        }
    }

    // No cache, partial caches and shuffled build orders give the same nodes and values
    #[test]
    fn cache_variants() {
        for seed in 0..30 {
            let f = Fixture::random(seed);
            let mut ctx = context(&f);
            let expected = evaluate(&ctx, &f);
            let full = ctx.templates.bytes;
            let keys: Vec<(usize, usize)> = (0..ctx.levels.len())
                .flat_map(|l| (0..(ctx.levels[l] * ctx.levels[l]) as usize).map(move |p| (l, p)))
                .collect();
            let mut rng = crate::testkit::Rng::new(seed);
            let mut shuffled = keys.clone();
            for k in (1..shuffled.len()).rev() {
                shuffled.swap(k, rng.below(k + 1));
            }
            let reversed: Vec<(usize, usize)> = keys.iter().rev().copied().collect();
            for (budget, order) in [
                (0, None),
                (full / 2, None),
                (full, Some(shuffled.as_slice())),
                (full, Some(reversed.as_slice())),
            ] {
                retemplate(&mut ctx, &f, budget, order);
                assert!(ctx.templates.bytes <= budget.max(ctx.templates.fixed), "seed {seed}, budget {budget}");
                assert_eq!(evaluate(&ctx, &f), expected, "seed {seed}, budget {budget}");
            }
            // Nothing cached: geometry and links are computed per cell
            retemplate(&mut ctx, &f, 0, None);
            assert!(ctx.templates.cached.iter().all(|&n| n == 0));
            assert!(ctx.templates.linked.iter().all(|&l| !l));
            compare_ctx(&f, &ctx);
        }
    }

    // Maximum levels 32 to 512 on small, sparse tiles: the measured cache stays within its budget
    // and levels that do not fit are evaluated per cell with identical results
    #[test]
    fn max_levels() {
        for (max_level, budget) in [(32usize, 1usize << 20), (64, 1 << 20), (128, 4 << 20), (512, 16 << 20)] {
            let mut f = Fixture::random(max_level as u64);
            f.levels = (0..=max_level.trailing_zeros()).map(|k| 1usize << k).collect();
            f.window_size = 3;
            f.outer_window = 3;
            f.offsets = (3 * max_level + 5, 7 * max_level - 3);
            f.mask.indexed_iter_mut().for_each(|((i, j), m)| *m |= (i * 7 + j * 3) % 5 != 0);
            let base_t = f.transforms[&1];
            let t = (base_t.x_scale, base_t.x_skew, base_t.x_origin, base_t.y_skew, base_t.y_scale, base_t.y_origin);
            f.transforms = crate::level_transforms(t, &f.levels, f.offsets);
            let mut ctx = context(&f);
            for budget in [budget, 64 << 10, 0] {
                let (tables, templates, retained, peak) = measured(&ctx, &f, budget);
                let fixed = tables.fixed + templates.fixed;
                assert_eq!(retained, tables.bytes + templates.bytes, "{max_level}");
                assert!(peak <= budget.max(fixed), "{max_level}: peak {peak} over {budget}");
                let uncached = templates.cached.iter().filter(|&&n| n == 0).count();
                println!(
                    "max level {max_level}, budget {budget}: retained {retained}, peak {peak}, phases {:?}, links {:?}",
                    templates.cached, templates.linked
                );
                assert!(budget > 64 << 10 || max_level < 128 || uncached > 0);
                (ctx.tables, ctx.templates) = (tables, templates);
                compare_ctx(&f, &ctx);
            }
            let stats = crate::core::conn_stats(&f.inputs(), &f.config()).unwrap().1;
            assert!(stats.cache_peak <= crate::template::CACHE_BUDGET);
            assert!(stats.cache_bytes <= stats.cache_peak && stats.cache_fixed <= stats.cache_bytes);
            assert_eq!(stats.scratch.len(), rayon::current_num_threads());
            assert!(stats.setup_s >= 0.0 && stats.process_s >= 0.0 && stats.cached.len() == f.levels.len());
            assert_eq!(stats.linked.len(), f.levels.len());
        }
    }

    // Dense 256 x 256 tiles at maximum level 256, with a standard and a much larger outer window:
    // the measured cache peak stays within the 128 MiB budget (near the limit for 5/9), saturated
    // levels fall back to per-cell evaluation, and sampled cells agree across full, partial and
    // empty caches and with the canonical reference
    #[test]
    fn dense_saturation() {
        use crate::template::CACHE_BUDGET;
        for (window_size, outer_window) in [(5, 9), (9, 31)] {
            let mut f = Fixture::random(256 + outer_window as u64);
            let mut rng = crate::testkit::Rng::new(outer_window as u64);
            f.cond = Array2::from_shape_fn((256, 256), |_| 0.05 + 0.95 * rng.unit() as f32);
            f.mask = Array2::from_elem((256, 256), false);
            f.trav = None;
            f.trans = None;
            f.levels = (0..=8).map(|k| 1usize << k).collect();
            (f.window_size, f.outer_window) = (window_size, outer_window);
            f.offsets = (5, 3);
            f.transforms = crate::level_transforms(
                (100.0, 0.0, 400_000.0, 0.0, -100.0, 6_000_000.0),
                &f.levels,
                f.offsets,
            );
            let mut ctx = context(&f);
            let cells: Vec<(usize, usize)> = (0..48).map(|k| ((k * 37 + 3) % 256, (k * 91 + 11) % 256)).collect();
            let values = |ctx: &Context| -> Vec<u32> {
                let mut scratch = Scratch::new(ctx);
                cells.iter().map(|&(i, j)| scratch.eval(ctx, i, j).to_bits()).collect()
            };
            let mut expected = None;
            for budget in [CACHE_BUDGET, CACHE_BUDGET / 8, 0] {
                let (tables, templates, retained, peak) = measured(&ctx, &f, budget);
                let fixed = tables.fixed + templates.fixed;
                println!(
                    "window {window_size}/{outer_window}, budget {budget}: retained {retained}, peak {peak}, \
                     phases {:?}, links {:?}",
                    templates.cached, templates.linked
                );
                assert_eq!(retained, tables.bytes + templates.bytes);
                assert!(peak <= budget.max(fixed), "peak {peak} over budget {budget}");
                (ctx.tables, ctx.templates) = (tables, templates);
                let got = values(&ctx);
                assert_eq!(expected.get_or_insert_with(|| got.clone()), &got, "budget {budget}");
            }
            let prepared = f.prepared();
            let mut scratch = Scratch::new(&ctx);
            for &(i, j) in cells.iter().step_by(6) {
                compare_cell(&ctx, &prepared, &mut scratch, i, j);
            }
        }
    }

    // Windows far larger than the tile: scratch grows with the tile rather than the window, and
    // uncached links and geometry match the canonical reference
    #[test]
    fn huge_windows() {
        for seed in 0..6 {
            let mut f = Fixture::random(seed);
            (f.window_size, f.outer_window) = (41, 101);
            let mut ctx = context(&f);
            let scratch = Scratch::new(&ctx);
            let tile: usize = ctx.cond.iter().map(|a| a.len()).sum();
            let window: usize = ctx.templates.shapes.iter().map(|s| s.len()).sum();
            assert!(tile < window / 4, "seed {seed}: tile {tile}, window {window}");
            assert!(scratch.bytes() < (tile + 1) * 2048, "seed {seed}: scratch {}", scratch.bytes());
            for budget in [crate::template::CACHE_BUDGET, 0] {
                retemplate(&mut ctx, &f, budget, None);
                compare_ctx(&f, &ctx);
            }
        }
    }

    /// Table distances keyed by legacy node ids, for injection into the canonical reference.
    pub fn table_inject<'c>(ctx: &'c Context) -> impl Fn(u64, u64, bool) -> Option<f32> + Sync + 'c {
        use crate::legacy::canonical::decode;
        move |u, v, synthetic| {
            let (lu, gr, gc) = decode(u);
            let l = ctx.levels.iter().position(|&x| x == lu).unwrap();
            let row = gr - (ctx.offsets.0 as i32).div_euclid(lu);
            let slot = if synthetic {
                SYNTHETIC
            } else {
                let (lv, hr, hc) = decode(v);
                if lv == lu {
                    (0..8).find(|&k| (COLS[k], ROWS[k]) == (hr - gr, hc - gc)).unwrap()
                } else {
                    let parity = (gr.rem_euclid(2), gc.rem_euclid(2));
                    8 + link_slot(parity, (hr - gr.div_euclid(2), hc - gc.div_euclid(2))) as usize
                }
            };
            let (d, status) = ctx.tables.get(l, row, slot);
            (status <= BOUNDED).then_some(d)
        }
    }

    /// Structural comparison with tallies over a fixture set; returns the summed tally.
    fn tally_fixtures(fixtures: impl Iterator<Item = Fixture>) -> Tally {
        let mut total = Tally::default();
        for f in fixtures {
            let ctx = context(&f);
            let prepared = f.prepared();
            let mut scratch = Scratch::new(&ctx);
            for ((i, j), &masked) in f.mask.indexed_iter() {
                if !masked {
                    compare_tally(&ctx, &prepared, &mut scratch, i, j, &mut total);
                }
            }
        }
        total
    }

    // Table hits, fallbacks, one-ulp distance changes and integer-cost mismatches on fixtures
    #[test]
    fn table_tally() {
        let fixtures = (0..300).map(Fixture::random).chain((0..20).flat_map(Fixture::focused));
        let tally = tally_fixtures(fixtures.chain((0..40).map(Fixture::thresholds)));
        println!("{tally:?}");
        assert_eq!((tally.ulp1, tally.ulp_more, tally.cost_mismatch), (0, 0, 0));
        assert!(tally.status[crate::tables::EXACT as usize] > 0);
        assert!(tally.status[crate::tables::BOUNDED as usize] > 0);
        let skew = tally_fixtures((0..20).map(Fixture::skewed));
        println!("skewed: {skew:?}");
        assert_eq!(skew.status[crate::tables::SKEW as usize], skew.edges);
    }

    // Canonical with table distances injected after its edge set is built equals Optimized, and
    // passes the predeclared distance-only criterion against canonical with legacy distances
    #[test]
    fn injected_canonical() {
        let fixtures = (0..80).map(Fixture::random).chain((0..20).map(Fixture::thresholds));
        for (n, f) in fixtures.chain((0..10).map(Fixture::skewed)).enumerate() {
            let ctx = context(&f);
            let inject = table_inject(&ctx);
            let injected = f.canonical(Some(&inject));
            let optimized = crate::core::conn(&f.inputs(), &f.config()).unwrap();
            assert!(crate::testkit::same_bits(&optimized, &injected), "fixture {n}");
            let reference = f.canonical(None);
            for (&a, &b) in injected.iter().zip(&reference) {
                assert_eq!(a.is_finite(), b.is_finite());
                if b.is_finite() {
                    assert!((a - b).abs() <= 1.0e-7 + 1.0e-6 * b.abs(), "fixture {n}: {a} vs {b}");
                }
            }
        }
    }
}
