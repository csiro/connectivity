//! Window templates for one `connectivity()` call.
//!
//! For an unclipped window at level L, the window cells and their area fractions and outer-edge
//! flags depend only on where the source sits in its level-L cell: the phase
//! `(global row mod L, global column mod L)`. Each level's candidates are the cells of the
//! legacy search box, `[start, start + side)` cells around the source's cell in both directions
//! (the same for every phase), listed row-major as `collect_annulus` visits them. Cells whose
//! unclipped fraction is zero stay listed because tile clipping can change their footprint.
//!
//! Connectivity is stored separately from the per-phase geometry:
//! - same-level queen neighbours in COLS/ROWS order do not depend on the phase;
//! - links to level 2L, in `get_edge_neighbours` order, depend on the candidate and on the
//!   parity of the source's level-L cell in the 2L grid. Phase mod L plus that parity is the
//!   phase mod 2L in both coordinates.
//! Links are listed for every candidate, because clipping, missing data or the square-mode
//! bounding box decide at run time which cells are fringe-eligible.
use crate::graph::{get_edge_neighbours, COLS, ROWS};
use crate::window::{circle_frac, square_frac, WindowMode};
use ndarray::Array2;
use rayon::prelude::*;

/// Missing candidate
pub const NO_CAND: u32 = u32::MAX;

/// Shared cache budget for templates and distance tables
pub const CACHE_BUDGET: usize = 128 << 20;

/// Links of one candidate to the next level: candidate indices of that level in
/// `get_edge_neighbours` order (NO_CAND outside its search box), with the distance slot of each.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Links {
    pub cand: [u32; 4],
    pub slot: [u8; 4],
    pub count: u8,
}

/// Distance slot of a link from a level-L cell with parity (pr, pc) in the 2L grid to the 2L
/// cell displaced by (dr, dc) from its parent: ((pr * 2 + pc) * 3 + dr + 1) * 3 + dc + 1.
pub fn link_slot(parity: (i32, i32), delta: (i32, i32)) -> u8 {
    (((parity.0 * 2 + parity.1) * 3 + delta.0 + 1) * 3 + delta.1 + 1) as u8
}

/// Phase-invariant candidate layout and connectivity of one level.
pub struct Shape {
    pub level: i32,
    pub start: i32, // offset of the first candidate row/column from the source's cell
    pub side: i32,  // candidates per row and column
    pub inner: f64,
    pub outer: f64,
    pub nbr: Vec<[u32; 8]>,     // same-level queen neighbours in COLS/ROWS order
    pub links: [Vec<Links>; 4], // by the source cell's parent-grid parity, row * 2 + col
}

/// Legacy search-box rows (start, end) relative to the source's cell for a source at `phase`.
fn search_box(level: i32, outer: f64, phase: i32) -> (i32, i32) {
    let source = phase as f64 + 0.5;
    let start = ((source - outer).floor() as i32).div_euclid(level);
    let end = ((source + outer).ceil() as i32 - 1).div_euclid(level);
    (start, end)
}

impl Shape {
    /// `next` is the (start, side) of the next level when it is twice this one.
    pub fn new(level: i32, window_size: i32, effective: i32, next: Option<(i32, i32)>) -> Self {
        let outer = level as f64 * effective as f64;
        let inner = if level == 1 {
            0.0
        } else {
            level as f64 * window_size as f64 * 0.5
        };
        let (start, end) = search_box(level, outer, 0);
        for phase in 1..level {
            assert_eq!(search_box(level, outer, phase), (start, end), "Phase-dependent search box.");
        }
        let side = (end - start + 1).max(0);
        let cand = |oi: i32, oj: i32, start: i32, side: i32| -> u32 {
            let (r, c) = (oi - start, oj - start);
            if r < 0 || c < 0 || r >= side || c >= side {
                NO_CAND
            } else {
                (r * side + c) as u32
            }
        };

        let n = (side * side) as usize;
        let mut nbr = Vec::with_capacity(n);
        let mut links: [Vec<Links>; 4] = Default::default();
        for r in 0..side {
            for c in 0..side {
                let (oi, oj) = (start + r, start + c);
                nbr.push(std::array::from_fn(|k| cand(oi + COLS[k], oj + ROWS[k], start, side)));
                let Some((next_start, next_side)) = next else {
                    continue;
                };
                for (b, list) in links.iter_mut().enumerate() {
                    // A tile origin at the source cell's parity reproduces the global parent grid
                    let parity = ((b / 2) as i32, (b % 2) as i32);
                    let offsets = ((parity.0 * level) as usize, (parity.1 * level) as usize);
                    let (higher, count) = get_edge_neighbours(oi, oj, level, offsets);
                    let cell = (parity.0 + oi, parity.1 + oj);
                    let parent = (cell.0.div_euclid(2), cell.1.div_euclid(2));
                    let mut entry = Links {
                        cand: [NO_CAND; 4],
                        slot: [0; 4],
                        count: count as u8,
                    };
                    for (m, &(hi, hj)) in higher[..count].iter().enumerate() {
                        entry.cand[m] = cand(hi, hj, next_start, next_side);
                        let cell_parity = (cell.0.rem_euclid(2), cell.1.rem_euclid(2));
                        entry.slot[m] = link_slot(cell_parity, (hi - parent.0, hj - parent.1));
                    }
                    list.push(entry);
                }
            }
        }
        Self {
            level,
            start,
            side,
            inner,
            outer,
            nbr,
            links,
        }
    }

    pub fn len(&self) -> usize {
        (self.side * self.side) as usize
    }

    /// Candidate index of the source's own cell
    pub fn centre(&self) -> u32 {
        (-self.start * self.side - self.start) as u32
    }

    /// Unclipped area fractions and outer-edge flags of every candidate for a source at `phase`
    /// inside its cell, from the existing f64 geometry helpers. Relative coordinates give the
    /// same half-integer differences as the legacy global coordinates, so values are identical.
    pub fn geom(&self, mode: WindowMode, phase: (i32, i32), out: &mut Geom) {
        out.frac.clear();
        out.rim.clear();
        let (source_r, source_c) = (phase.0 as f64 + 0.5, phase.1 as f64 + 0.5);
        for r in 0..self.side {
            let cell_r0 = ((self.start + r) * self.level) as f64;
            let cell_r1 = cell_r0 + self.level as f64;
            for c in 0..self.side {
                let cell_c0 = ((self.start + c) * self.level) as f64;
                let cell_c1 = cell_c0 + self.level as f64;
                let (fraction, rim) = match mode {
                    WindowMode::Circular => circle_frac(
                        cell_r0, cell_r1, cell_c0, cell_c1, source_r, source_c, self.inner, self.outer,
                    ),
                    WindowMode::Square => (
                        square_frac(
                            cell_r0, cell_r1, cell_c0, cell_c1, source_r, source_c, self.inner, self.outer,
                        ),
                        false,
                    ),
                };
                out.frac.push(fraction);
                out.rim.push(rim);
            }
        }
    }

    fn bytes(&self) -> usize {
        self.nbr.capacity() * 32
            + self.links.iter().map(|l| l.capacity() * std::mem::size_of::<Links>()).sum::<usize>()
    }
}

/// Unclipped geometry of one level's candidates for one phase.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Geom {
    pub frac: Vec<f32>,
    pub rim: Vec<bool>,
}

impl Geom {
    fn bytes(n: usize) -> usize {
        5 * n + 2 * std::mem::size_of::<Vec<f32>>()
    }
}

/// Shapes of every level and the cached geometry of the phases that occur in this call.
pub struct Templates {
    pub shapes: Vec<Shape>,
    geoms: Vec<Vec<Option<Box<Geom>>>>, // [level][row phase * L + column phase]; empty if uncached
    pub cached: Vec<usize>,              // phases cached per level
    pub bytes: usize,
}

impl Templates {
    /// Shapes for every level, then cached geometry for the phases of unmasked cells, level by
    /// level from the smallest, while the estimate fits `budget`. Levels that do not fit are
    /// evaluated per cell into scratch with the same `Shape::geom`, so caching never changes
    /// nodes or their order.
    pub fn new(
        levels: &[i32],
        effective: &[i32],
        window_size: i32,
        mode: WindowMode,
        mask: &Array2<bool>,
        offsets: (usize, usize),
        budget: usize,
        order: Option<&[(usize, usize)]>,
    ) -> Self {
        let n_levels = levels.len();
        let mut shapes: Vec<Shape> = Vec::with_capacity(n_levels);
        for l in (0..n_levels).rev() {
            let next = (l + 1 < n_levels && levels[l + 1] == 2 * levels[l])
                .then(|| shapes.last().map(|s: &Shape| (s.start, s.side)).unwrap());
            shapes.push(Shape::new(levels[l], window_size, effective[l], next));
        }
        shapes.reverse();
        let mut bytes: usize = shapes.iter().map(Shape::bytes).sum();

        // Phases of unmasked cells, per level, while the estimate stays within budget
        let mut geoms: Vec<Vec<Option<Box<Geom>>>> = (0..n_levels).map(|_| Vec::new()).collect();
        let mut cached = vec![0; n_levels];
        let mut todo: Vec<(usize, usize)> = Vec::new();
        for (l, shape) in shapes.iter().enumerate() {
            let level = levels[l] as usize;
            let phases = level * level;
            if phases * std::mem::size_of::<Option<Box<Geom>>>() > budget.saturating_sub(bytes) {
                continue;
            }
            let mut needed = vec![false; phases];
            for ((i, j), &masked) in mask.indexed_iter() {
                if !masked {
                    needed[((offsets.0 + i) % level) * level + (offsets.1 + j) % level] = true;
                }
            }
            let count = needed.iter().filter(|&&n| n).count();
            let level_bytes = phases * std::mem::size_of::<Option<Box<Geom>>>()
                + count * (Geom::bytes(shape.len()) + std::mem::size_of::<Geom>());
            if bytes + level_bytes > budget {
                continue;
            }
            bytes += level_bytes;
            cached[l] = count;
            geoms[l] = (0..phases).map(|_| None).collect();
            todo.extend(needed.iter().enumerate().filter(|(_, &n)| n).map(|(p, _)| (l, p)));
        }

        let build = |&(l, p): &(usize, usize)| {
            let level = levels[l] as usize;
            let mut geom = Geom::default();
            shapes[l].geom(mode, ((p / level) as i32, (p % level) as i32), &mut geom);
            (l, p, Box::new(geom))
        };
        let built: Vec<(usize, usize, Box<Geom>)> = match order {
            // Test hook: build one at a time in a given order
            Some(order) => order.iter().filter(|key| todo.contains(key)).map(build).collect(),
            None => todo.par_iter().map(build).collect(),
        };
        for (l, p, geom) in built {
            geoms[l][p] = Some(geom);
        }

        Self {
            shapes,
            geoms,
            cached,
            bytes,
        }
    }

    /// Bytes of cached geometry (excluding the always-built shapes)
    pub fn cached_bytes(&self) -> usize {
        self.bytes - self.shapes.iter().map(Shape::bytes).sum::<usize>()
    }

    /// Cached geometry of level `l` at `phase`, if any
    #[inline]
    pub fn geom(&self, l: usize, phase: (i32, i32)) -> Option<&Geom> {
        let level = self.shapes[l].level;
        self.geoms[l]
            .get((phase.0 * level + phase.1) as usize)
            .and_then(|g| g.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The search box of collect_annulus is the same for every phase
    #[test]
    fn box_is_phase_free() {
        for level in [1, 2, 4, 8, 32] {
            for eff in [1, 3, 5, 7, 9] {
                let shape = Shape::new(level, 3, eff, None);
                assert_eq!((shape.start, shape.side), (-eff, 2 * eff + 1));
            }
        }
        assert_eq!(Shape::new(4, 3, -1, None).side, 0);
    }

    // Links reproduce get_edge_neighbours on global coordinates for every parity
    #[test]
    fn links_match_parents() {
        let (level, eff, next_eff) = (2, 3, 3);
        let next = Shape::new(4, 3, next_eff, None);
        let shape = Shape::new(level, 3, eff, Some((next.start, next.side)));
        for tile in [(0usize, 0usize), (2, 0), (0, 2), (2, 2), (6, 10)] {
            for grow in 40..44 {
                for gcol in 60..64 {
                    let src = (grow / level, gcol / level);
                    let b = ((src.0 % 2) * 2 + src.1 % 2) as usize;
                    let higher_src = (src.0 / 2, src.1 / 2);
                    for c in 0..shape.len() {
                        let (oi, oj) = (shape.start + c as i32 / shape.side, shape.start + c as i32 % shape.side);
                        // Local coordinates in an arbitrary tile
                        let origin = (tile.0 as i32 / level, tile.1 as i32 / level);
                        let local = (src.0 + oi - origin.0, src.1 + oj - origin.1);
                        let (higher, count) = get_edge_neighbours(local.0, local.1, level, tile);
                        let links = shape.links[b][c];
                        assert_eq!(links.count as usize, count);
                        let up_origin = (tile.0 as i32 / (2 * level), tile.1 as i32 / (2 * level));
                        for m in 0..count {
                            let rel = (
                                higher[m].0 + up_origin.0 - higher_src.0,
                                higher[m].1 + up_origin.1 - higher_src.1,
                            );
                            let expected = if rel.0.abs() <= next_eff && rel.1.abs() <= next_eff {
                                ((rel.0 - next.start) * next.side + rel.1 - next.start) as u32
                            } else {
                                NO_CAND
                            };
                            assert_eq!(links.cand[m], expected);
                        }
                    }
                }
            }
        }
    }
}
