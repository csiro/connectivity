//! Window templates for one `connectivity()` call.
//!
//! For an unclipped window at level L, the window cells and their area fractions and outer-edge
//! flags depend only on where the source sits in its level-L cell: the phase
//! `(global row mod L, global column mod L)`. Each level's candidates are the cells of the
//! legacy search box, `[start, start + side)` cells around the source's cell in both directions
//! (the same for every phase), listed row-major as `collect_annulus` visits them. Cells whose
//! unclipped fraction is zero stay listed because tile clipping can change their footprint.
//!
//! Connectivity needs no per-phase data:
//! - same-level queen neighbours are a candidate's COLS/ROWS offsets;
//! - links to level 2L, in `get_edge_neighbours` order, depend on the candidate and on the
//!   parity of the source's level-L cell in the 2L grid. Phase mod L plus that parity is the
//!   phase mod 2L in both coordinates.
//! Links are available for every candidate, because clipping, missing data or the square-mode
//! bounding box decide at run time which cells are fringe-eligible.
//!
//! Memory: links and per-phase geometry are cached only within the call's budget, which bounds
//! the cache's peak heap while it is built and afterwards. Sizes are computed before anything is
//! allocated, and every cached array is an exact-length boxed slice. Whatever does not fit is
//! computed per cell by the same functions, so caching never changes nodes, edges or order.
use crate::graph::get_edge_neighbours;
use crate::window::{circle_frac, square_frac, WindowMode};
use ndarray::Array2;
use rayon::prelude::*;
use std::mem::size_of;

/// Heap budget of one call's shared cache: distance tables, cached links and geometry, their
/// containers and the temporaries that build them, at any moment while building and afterwards.
/// Only the per-level descriptors (a few hundred bytes) are allocated whatever the budget. Worker
/// scratch, overviews and the output are outside it.
pub const CACHE_BUDGET: usize = 128 << 20;

/// Phase without cached geometry
const NO_SLOT: u32 = u32::MAX;

/// Links of one candidate to the next level in `get_edge_neighbours` order: each target cell
/// relative to the source's next-level cell, with its distance slot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Links {
    pub cell: [(i32, i32); 4],
    pub slot: [u8; 4],
    pub count: u8,
}

/// Distance slot of a link from a level-L cell with parity (pr, pc) in the 2L grid to the 2L
/// cell displaced by (dr, dc) from its parent: ((pr * 2 + pc) * 3 + dr + 1) * 3 + dc + 1.
pub fn link_slot(parity: (i32, i32), delta: (i32, i32)) -> u8 {
    (((parity.0 * 2 + parity.1) * 3 + delta.0 + 1) * 3 + delta.1 + 1) as u8
}

/// Candidate layout of one level, with its links cached when they fit the budget.
pub struct Shape {
    pub level: i32,
    pub start: i32, // offset of the first candidate row/column from the source's cell
    pub side: i32,  // candidates per row and column
    pub inner: f64,
    pub outer: f64,
    pub fringe: bool,            // links to the next level, which is twice this one
    links: Option<Box<[Links]>>, // [parity * len + candidate], row * 2 + col parity
}

/// Legacy search-box rows (start, end) relative to the source's cell for a source at `phase`.
fn search_box(level: i32, outer: f64, phase: i32) -> (i32, i32) {
    let source = phase as f64 + 0.5;
    let start = ((source - outer).floor() as i32).div_euclid(level);
    let end = ((source + outer).ceil() as i32 - 1).div_euclid(level);
    (start, end)
}

impl Shape {
    pub fn new(level: i32, window_size: i32, effective: i32, fringe: bool) -> Self {
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
        Self {
            level,
            start,
            side: (end - start + 1).max(0),
            inner,
            outer,
            fringe,
            links: None,
        }
    }

    pub fn len(&self) -> usize {
        (self.side as usize).pow(2)
    }

    /// Bytes of the cached links
    fn link_bytes(&self) -> usize {
        if self.fringe {
            4 * self.len() * size_of::<Links>()
        } else {
            0
        }
    }

    /// Links of box candidate (r, c) for a source cell with parent-grid `parity`.
    #[inline]
    pub fn links(&self, parity: usize, r: i32, c: i32) -> Links {
        match &self.links {
            Some(links) => links[parity * self.len() + (r * self.side + c) as usize],
            None => self.link(parity, r, c),
        }
    }

    /// `get_edge_neighbours` of a candidate, with a tile origin at the source cell's parity so
    /// local indices reproduce the global parent grid and the source's 2L cell is (0, 0).
    fn link(&self, parity: usize, r: i32, c: i32) -> Links {
        let parity = ((parity / 2) as i32, (parity % 2) as i32);
        let (oi, oj) = (self.start + r, self.start + c);
        let offsets = ((parity.0 * self.level) as usize, (parity.1 * self.level) as usize);
        let (higher, count) = get_edge_neighbours(oi, oj, self.level, offsets);
        let cell = (parity.0 + oi, parity.1 + oj);
        let cell_parity = (cell.0.rem_euclid(2), cell.1.rem_euclid(2));
        let parent = (cell.0.div_euclid(2), cell.1.div_euclid(2));
        let mut links = Links {
            cell: [(0, 0); 4],
            slot: [0; 4],
            count: count as u8,
        };
        for (m, &(hi, hj)) in higher[..count].iter().enumerate() {
            links.cell[m] = (hi, hj);
            links.slot[m] = link_slot(cell_parity, (hi - parent.0, hj - parent.1));
        }
        links
    }

    /// Unclipped area fraction and outer-edge flag of box candidate (r, c) for a source at
    /// `phase` inside its cell, from the existing f64 geometry helpers. Relative coordinates give
    /// the same half-integer differences as the legacy global coordinates, so values are identical.
    #[inline]
    pub fn cell_geom(&self, mode: WindowMode, phase: (i32, i32), r: i32, c: i32) -> (f32, bool) {
        let (source_r, source_c) = (phase.0 as f64 + 0.5, phase.1 as f64 + 0.5);
        let cell_r0 = ((self.start + r) * self.level) as f64;
        let cell_r1 = cell_r0 + self.level as f64;
        let cell_c0 = ((self.start + c) * self.level) as f64;
        let cell_c1 = cell_c0 + self.level as f64;
        match mode {
            WindowMode::Circular => circle_frac(
                cell_r0, cell_r1, cell_c0, cell_c1, source_r, source_c, self.inner, self.outer,
            ),
            WindowMode::Square => (
                square_frac(cell_r0, cell_r1, cell_c0, cell_c1, source_r, source_c, self.inner, self.outer),
                false,
            ),
        }
    }

    /// `cell_geom` of every candidate, row-major.
    fn geom_into(&self, mode: WindowMode, phase: (i32, i32), frac: &mut [f32], rim: &mut [bool]) {
        for r in 0..self.side {
            for c in 0..self.side {
                let k = (r * self.side + c) as usize;
                (frac[k], rim[k]) = self.cell_geom(mode, phase, r, c);
            }
        }
    }
}

/// Cached geometry of one level: a block of `len` fractions and rim flags per cached phase.
struct Geoms {
    slot: Box<[u32]>, // block of each phase (row * L + column), or NO_SLOT
    frac: Box<[f32]>,
    rim: Box<[bool]>,
}

/// Shapes of every level and the links and geometry cached for this call.
pub struct Templates {
    pub shapes: Box<[Shape]>,
    geoms: Box<[Option<Geoms>]>,
    pub cached: Box<[usize]>, // phases with cached geometry, per level
    pub linked: Box<[bool]>,  // levels with cached links
    pub fixed: usize,         // per-level descriptors, allocated whatever the budget
    pub bytes: usize,         // retained heap bytes, from allocation sizes
    pub peak: usize,          // largest heap total while building, temporaries included
}

impl Templates {
    /// Shapes of every level, then links and then geometry for the phases of unmasked cells, level
    /// by level from the smallest, each only if the cache's heap stays within `budget`.
    /// `order` is a test hook that builds geometry one phase at a time in the given order.
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
        let n = levels.len();
        let fixed = Self::fixed(n);
        let mut shapes = Vec::with_capacity(n);
        for l in 0..n {
            let fringe = l + 1 < n && levels[l + 1] == 2 * levels[l];
            shapes.push(Shape::new(levels[l], window_size, effective[l], fringe));
        }
        let mut geoms = Vec::with_capacity(n);
        geoms.resize_with(n, || None);
        let mut cached = vec![0; n];
        let mut linked = vec![false; n];
        let (mut used, mut peak) = (fixed, fixed);

        for (l, shape) in shapes.iter_mut().enumerate() {
            let bytes = shape.link_bytes();
            if bytes == 0 || used + bytes > budget {
                continue;
            }
            let mut links = Vec::with_capacity(4 * shape.len());
            for parity in 0..4 {
                for r in 0..shape.side {
                    for c in 0..shape.side {
                        links.push(shape.link(parity, r, c));
                    }
                }
            }
            shape.links = Some(links.into_boxed_slice());
            used += bytes;
            peak = peak.max(used);
            linked[l] = true;
        }

        for (l, shape) in shapes.iter().enumerate() {
            let (level, len) = (levels[l] as usize, shape.len());
            let phases = level * level;
            // The phase index, plus a bitmap of needed phases while it is built
            let index = phases * size_of::<u32>();
            let bitmap = phases.div_ceil(64) * size_of::<u64>();
            if len == 0 || used + index + bitmap > budget {
                continue;
            }
            peak = peak.max(used + bitmap);
            let mut needed = vec![0u64; phases.div_ceil(64)].into_boxed_slice();
            for ((i, j), &masked) in mask.indexed_iter() {
                if !masked {
                    let p = ((offsets.0 + i) % level) * level + (offsets.1 + j) % level;
                    needed[p / 64] |= 1 << (p % 64);
                }
            }
            let count = needed.iter().map(|w| w.count_ones() as usize).sum::<usize>();
            let retained = index + count * len * (size_of::<f32>() + size_of::<bool>());
            let temporary = bitmap + count * size_of::<u32>();
            if used + retained + temporary > budget {
                continue;
            }
            peak = peak.max(used + retained + temporary);

            let mut slot = vec![NO_SLOT; phases].into_boxed_slice();
            let mut blocks: Vec<u32> = Vec::with_capacity(count);
            for p in 0..phases {
                if needed[p / 64] >> (p % 64) & 1 == 1 {
                    slot[p] = blocks.len() as u32;
                    blocks.push(p as u32);
                }
            }
            drop(needed);
            let mut frac = vec![0.0f32; count * len].into_boxed_slice();
            let mut rim = vec![false; count * len].into_boxed_slice();
            let phase = |p: u32| ((p as usize / level) as i32, (p as usize % level) as i32);
            match order {
                Some(order) => {
                    let mut filled = 0;
                    for &(_, p) in order.iter().filter(|&&(ol, p)| ol == l && slot[p] != NO_SLOT) {
                        let b = slot[p] as usize;
                        let (f, r) = (&mut frac[b * len..(b + 1) * len], &mut rim[b * len..(b + 1) * len]);
                        shape.geom_into(mode, phase(p as u32), f, r);
                        filled += 1;
                    }
                    assert_eq!(filled, count, "Build order must cover every cached phase.");
                }
                None => frac
                    .par_chunks_mut(len)
                    .zip(rim.par_chunks_mut(len))
                    .zip(blocks.par_iter())
                    .for_each(|((f, r), &p)| shape.geom_into(mode, phase(p), f, r)),
            }
            drop(blocks);
            geoms[l] = Some(Geoms { slot, frac, rim });
            used += retained;
            cached[l] = count;
        }

        Self {
            shapes: shapes.into_boxed_slice(),
            geoms: geoms.into_boxed_slice(),
            cached: cached.into_boxed_slice(),
            linked: linked.into_boxed_slice(),
            fixed,
            bytes: used,
            peak,
        }
    }

    /// Bytes of the per-level descriptors, which are allocated whatever the budget
    pub fn fixed(n_levels: usize) -> usize {
        n_levels * (size_of::<Shape>() + size_of::<Option<Geoms>>() + size_of::<usize>() + size_of::<bool>())
    }

    /// Cached fractions and rim flags of every candidate of level `l` at `phase`, if any.
    #[inline]
    pub fn geom(&self, l: usize, phase: (i32, i32)) -> Option<(&[f32], &[bool])> {
        let geoms = self.geoms[l].as_ref()?;
        let shape = &self.shapes[l];
        let block = geoms.slot[(phase.0 * shape.level + phase.1) as usize];
        if block == NO_SLOT {
            return None;
        }
        let range = block as usize * shape.len()..(block as usize + 1) * shape.len();
        Some((&geoms.frac[range.clone()], &geoms.rim[range]))
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
                let shape = Shape::new(level, 3, eff, false);
                assert_eq!((shape.start, shape.side), (-eff, 2 * eff + 1));
            }
        }
        assert_eq!(Shape::new(4, 3, -1, false).side, 0);
    }

    // Links reproduce get_edge_neighbours on global coordinates for every parity, cached or not
    #[test]
    fn links_match_parents() {
        let level = 2;
        let mut cached = Shape::new(level, 3, 3, true);
        let computed = Shape::new(level, 3, 3, true);
        let links: Vec<Links> = (0..4)
            .flat_map(|b| (0..cached.side).flat_map(move |r| (0..cached.side).map(move |c| (b, r, c))))
            .map(|(b, r, c)| cached.link(b, r, c))
            .collect();
        cached.links = Some(links.into_boxed_slice());
        for tile in [(0usize, 0usize), (2, 0), (0, 2), (2, 2), (6, 10)] {
            for grow in 40..44 {
                for gcol in 60..64 {
                    let src = (grow / level, gcol / level);
                    let b = ((src.0 % 2) * 2 + src.1 % 2) as usize;
                    let up = (src.0 / 2, src.1 / 2);
                    for r in 0..cached.side {
                        for c in 0..cached.side {
                            let (oi, oj) = (cached.start + r, cached.start + c);
                            // Local coordinates in an arbitrary tile
                            let origin = (tile.0 as i32 / level, tile.1 as i32 / level);
                            let local = (src.0 + oi - origin.0, src.1 + oj - origin.1);
                            let (higher, count) = get_edge_neighbours(local.0, local.1, level, tile);
                            let up_origin = (tile.0 as i32 / (2 * level), tile.1 as i32 / (2 * level));
                            let links = cached.links(b, r, c);
                            assert_eq!(links, computed.links(b, r, c));
                            assert_eq!(links.count as usize, count);
                            for m in 0..count {
                                let rel = (higher[m].0 + up_origin.0 - up.0, higher[m].1 + up_origin.1 - up.1);
                                assert_eq!(links.cell[m], rel);
                            }
                        }
                    }
                }
            }
        }
    }
}
