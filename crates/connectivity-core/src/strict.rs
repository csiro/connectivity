//! NoData crossing, mode 2 (strict gap width).
//!
//! Traversable base cells are land plus crossable NoData (see `crossing`). Every coarse cell is
//! split into its pieces: the 8-connected components of its traversable base cells inside its
//! footprint. A piece is a node; pieces are linked, at the same level and to the next level,
//! only where their base cells overlap or are 8-adjacent. Every graph path therefore follows a
//! base-resolution path that crosses no NoData gap wider than the permitted width, at every
//! level and inside one landmass as well as between landmasses.
//!
//! Storage: per level above 1, one byte each of piece kind, touch directions and relaxed diagonal
//! contact per frame cell, plus the cells holding several pieces, numbered row-major per level.
//! Level 1 needs nothing: its pieces are the traversable base cells, read from the traversal
//! overview. Level 2 cells (2 x 2) always hold one piece, because all cells of a 2 x 2 block are
//! 8-adjacent.
use crate::crossing::{Acc, Base, Part};
use crate::graph::{COLS, NONE, ROWS};
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Axis};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Multiply-and-rotate hash for cell keys: the tables are small and their keys are not chosen by
/// an adversary, so SipHash's cost buys nothing here.
#[derive(Default)]
pub struct CellHasher(u64);

impl Hasher for CellHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }

    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.write_u64(v as u64);
    }

    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(26) ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub type CellMap<V> = HashMap<(u32, u32), V, BuildHasherDefault<CellHasher>>;

pub const SINGLE: u8 = 1;
pub const MULTI: u8 = 2;
/// Opposite of direction k in COLS/ROWS order
pub const OPP: [usize; 8] = [2, 3, 0, 1, 7, 6, 5, 4];

/// A coarse cell holding several pieces.
#[derive(Debug, Default)]
pub struct Multi {
    pub parts: Vec<Part>,
    /// Per piece: (direction k, piece of the neighbouring cell) pairs whose base cells touch
    pub touch: Vec<Vec<(u8, u16)>>,
    /// Per piece: (row, column) of a next-finer frame cell relative to (2I, 2J), and the piece of
    /// that cell, for every finer piece whose base cells overlap or touch this piece
    pub kids: Vec<Vec<(i8, i8, u16)>>,
}

/// Test diagnostics of one strict-mode build.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
#[allow(dead_code)] // read through Debug in island::proto_bench
pub struct StrictStats {
    pub crossable: usize,
    pub multi: Vec<usize>,  // cells with several pieces, per level
    pub pieces: Vec<usize>, // pieces in those cells, per level
    pub bytes: usize,       // kind and touch arrays plus the multi tables (estimate)
}

pub struct Strict {
    /// Per level: 0 no traversable cell, SINGLE or MULTI, per frame cell. Empty at level 1.
    kind: Vec<Array2<u8>>,
    /// Per level: bit k set when traversable base cells touch across direction k. Empty at level 1.
    touch: Vec<Array2<u8>>,
    /// Per level: for a cell holding one piece, bit k (diagonals, 4..8) set when its neighbour in
    /// direction k holds one piece, they do not touch and `through` links them. It answers
    /// relaxed contact between single pieces, which is most of it, without the corner cells'
    /// lookups; pairs involving a cell of several pieces call `through`. Empty at level 1.
    relaxed: Vec<Array2<u8>>,
    /// Per level: the cells holding several pieces, in row-major frame order; a node keeps its
    /// cell's index (`multi_id`), so building edges needs no lookup by cell
    multis: Vec<Vec<Multi>>,
    ids: Vec<CellMap<u32>>,  // index in `multis` of each such frame cell
    shift: Vec<(i32, i32)>, // frame offset of the tile's cells, per level
    #[cfg(test)]
    pub stats: StrictStats,
    /// Relaxed contact (the rule; plain contact only for tests): a diagonal edge between pieces
    /// that do not touch is kept when both touch one piece of a cell sharing their corner.
    pub relax: bool,
}

type Rect = (usize, usize, usize, usize); // rows r0..r1, columns c0..c1

/// Calls `f(x, y)` for every 8-adjacent pair of base cells with x in `a` and y in its neighbour
/// `b` in direction k, until `f` returns true; returns whether it did.
#[inline]
fn pairs(a: Rect, b: Rect, k: usize, mut f: impl FnMut((usize, usize), (usize, usize)) -> bool) -> bool {
    let (di, dj) = (COLS[k], ROWS[k]);
    let rows = match di {
        1 => a.1 - 1..a.1,
        -1 => a.0..a.0 + 1,
        _ => a.0..a.1,
    };
    let cols = match dj {
        1 => a.3 - 1..a.3,
        -1 => a.2..a.2 + 1,
        _ => a.2..a.3,
    };
    for r in rows {
        for c in cols.clone() {
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    let (yr, yc) = (r as i64 + dr, c as i64 + dc);
                    if yr < b.0 as i64 || yr >= b.1 as i64 || yc < b.2 as i64 || yc >= b.3 as i64 {
                        continue;
                    }
                    if f((r, c), (yr as usize, yc as usize)) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// The two cells sharing the corner between cell (i, j) and its diagonal neighbour in direction
/// k, each with the directions from (i, j) to it and from it to that neighbour.
#[inline]
fn corners(i: i32, j: i32, k: usize) -> [(i32, i32, usize, usize); 2] {
    let (di, dj) = (COLS[k], ROWS[k]);
    let dir = |y: i32, x: i32| (0..8).find(|&m| (COLS[m], ROWS[m]) == (y, x)).unwrap();
    [(i + di, j, dir(di, 0), dir(0, dj)), (i, j + dj, dir(0, dj), dir(di, 0))]
}

/// Pushes `v` unless it equals the last entry: neighbouring base cells mostly repeat it.
#[inline]
fn push_new<T: PartialEq>(list: &mut Vec<T>, v: T) {
    if list.last() != Some(&v) {
        list.push(v);
    }
}

/// Sorts `list`, drops repeated entries and frees its spare capacity.
fn compact<T: Ord>(list: &mut Vec<T>) {
    list.sort_unstable();
    list.dedup();
    list.shrink_to_fit();
}

/// Most pieces one coarse cell may hold: labels are u16, with 0 off the domain. Only cells of 512
/// x 512 base cells or more can exceed it (a cell of side n holds at most (n / 2)^2 pieces).
pub const MAX_PIECES: u16 = u16::MAX - 1;

/// Piece labels of every base cell at one level (0 off the domain, else 1 + the piece's index in
/// its coarse cell, numbered in row-major order of first cells) and each coarse cell's kind, or
/// `None` when a cell holds more than `MAX_PIECES` pieces.
pub fn label_level(domain: &Array2<bool>, size: usize) -> Option<(Array2<u16>, Array2<u8>)> {
    let (h, w) = domain.dim();
    let (ch, cw) = (h.div_ceil(size), w.div_ceil(size));
    let mut labels = Array2::<u16>::zeros((h, w));
    let mut kind = Array2::<u8>::zeros((ch, cw));
    labels
        .axis_chunks_iter_mut(Axis(0), size)
        .into_par_iter()
        .zip(kind.axis_iter_mut(Axis(0)).into_par_iter())
        .enumerate()
        .try_for_each(|(ci, (mut lab, mut krow))| -> Option<()> {
            let r0 = ci * size;
            let rows = lab.nrows();
            let mut stack: Vec<(usize, usize)> = Vec::new();
            for cj in 0..cw {
                let (c0, c1) = (cj * size, ((cj + 1) * size).min(w));
                let mut n = 0u16;
                for r in 0..rows {
                    for c in c0..c1 {
                        if !domain[[r0 + r, c]] || lab[[r, c]] != 0 {
                            continue;
                        }
                        if n == MAX_PIECES {
                            return None;
                        }
                        n += 1;
                        lab[[r, c]] = n;
                        stack.push((r, c));
                        while let Some((pr, pc)) = stack.pop() {
                            for dr in -1i64..=1 {
                                for dc in -1i64..=1 {
                                    let (nr, nc) = (pr as i64 + dr, pc as i64 + dc);
                                    if nr < 0 || nr >= rows as i64 || nc < c0 as i64 || nc >= c1 as i64 {
                                        continue;
                                    }
                                    let (nr, nc) = (nr as usize, nc as usize);
                                    if domain[[r0 + nr, nc]] && lab[[nr, nc]] == 0 {
                                        lab[[nr, nc]] = n;
                                        stack.push((nr, nc));
                                    }
                                }
                            }
                        }
                    }
                }
                krow[cj] = n.min(2) as u8;
            }
            Some(())
        })?;
    Some((labels, kind))
}

impl Strict {
    /// Kinds, touch directions and multi-piece tables of every level above 1; fails when a cell
    /// holds more than `MAX_PIECES` pieces.
    pub fn build(base: &Base) -> anyhow::Result<Self> {
        let domain = &base.domain;
        let dim = domain.dim();
        let levels = &base.levels;
        let mut kinds = vec![Array2::<u8>::zeros((0, 0))];
        let mut touches = vec![Array2::<u8>::zeros((0, 0))];
        let mut multis = vec![Vec::new()];
        let mut ids = vec![CellMap::default()];
        #[cfg(test)]
        let mut stats = StrictStats { crossable: base.crossable, multi: vec![0], pieces: vec![0], ..Default::default() };
        let mut prev: Option<Array2<u16>> = None; // labels one level down; None at level 1
        for l in 1..levels.len() {
            let (size, finer) = (levels[l] as usize, levels[l - 1] as usize);
            let Some((labels, kind)) = label_level(domain, size) else {
                anyhow::bail!(
                    "Strict NoData crossing allows at most {MAX_PIECES} separate pieces of land and crossable NoData in one overview cell, but a level-{size} cell holds more; use levels up to 256"
                );
            };
            let (ch, cw) = kind.dim();
            let neighbour = |ci: usize, cj: usize, k: usize| -> Option<(usize, usize)> {
                let (ni, nj) = (ci as i64 + COLS[k] as i64, cj as i64 + ROWS[k] as i64);
                (ni >= 0 && nj >= 0 && (ni as usize) < ch && (nj as usize) < cw).then_some((ni as usize, nj as usize))
            };

            // Touch directions of every cell
            let mut touch = Array2::<u8>::zeros((ch, cw));
            touch.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(ci, mut row)| {
                for cj in 0..cw {
                    if kind[[ci, cj]] == 0 {
                        continue;
                    }
                    let a = base.footprint(ci, cj, size);
                    let mut bits = 0u8;
                    for k in 0..8 {
                        let Some((ni, nj)) = neighbour(ci, cj, k) else { continue };
                        if kind[[ni, nj]] == 0 {
                            continue;
                        }
                        let b = base.footprint(ni, nj, size);
                        if pairs(a, b, k, |x, y| domain[x] && domain[y]) {
                            bits |= 1 << k;
                        }
                    }
                    row[cj] = bits;
                }
            });

            // Pieces, touches and finer links of cells holding several pieces
            let cells: Vec<(usize, usize)> = kind.indexed_iter().filter(|(_, &k)| k == MULTI).map(|(c, _)| c).collect();
            let multi: Vec<Multi> = cells
                .par_iter()
                .map(|&(ci, cj)| {
                    let a = base.footprint(ci, cj, size);
                    let mut sums: Vec<Acc> = Vec::new();
                    for r in a.0..a.1 {
                        for c in a.2..a.3 {
                            let q = labels[[r, c]] as usize;
                            if q == 0 {
                                continue;
                            }
                            if sums.len() < q {
                                sums.resize(q, Acc::default());
                            }
                            base.add(&mut sums[q - 1], r, c);
                        }
                    }
                    let parts: Vec<Part> = sums.iter().map(|acc| base.finish(acc)).collect();
                    let mut touch = vec![Vec::new(); parts.len()];
                    for k in 0..8 {
                        let Some((ni, nj)) = neighbour(ci, cj, k) else { continue };
                        let b = base.footprint(ni, nj, size);
                        pairs(a, b, k, |x, y| {
                            if domain[x] && domain[y] {
                                push_new(&mut touch[labels[x] as usize - 1], (k as u8, labels[y] - 1));
                            }
                            false
                        });
                    }
                    let mut kids = vec![Vec::new(); parts.len()];
                    for r in a.0..a.1 {
                        for c in a.2..a.3 {
                            let q = labels[[r, c]] as usize;
                            if q == 0 {
                                continue;
                            }
                            for dr in -1i64..=1 {
                                for dc in -1i64..=1 {
                                    let (yr, yc) = (r as i64 + dr, c as i64 + dc);
                                    if yr < 0 || yc < 0 || yr >= dim.0 as i64 || yc >= dim.1 as i64 {
                                        continue;
                                    }
                                    let y = (yr as usize, yc as usize);
                                    if !domain[y] {
                                        continue;
                                    }
                                    let piece = prev.as_ref().map_or(0, |p| p[y] - 1);
                                    let rel = ((y.0 / finer) as i64 - 2 * ci as i64, (y.1 / finer) as i64 - 2 * cj as i64);
                                    push_new(&mut kids[q - 1], (rel.0 as i8, rel.1 as i8, piece));
                                }
                            }
                        }
                    }
                    // Retained for the whole call: keep each distinct entry once, with no spare
                    // capacity (the base-cell scan pushes most entries many times)
                    touch.iter_mut().for_each(compact);
                    kids.iter_mut().for_each(compact);
                    Multi { parts, touch, kids }
                })
                .collect();
            #[cfg(test)]
            {
                stats.multi.push(multi.len());
                stats.pieces.push(multi.iter().map(|m| m.parts.len()).sum());
                // kind, touch and relaxed bytes, the cell index map (an estimate) and the tables
                stats.bytes += 3 * kind.len()
                    + 16 * multi.len()
                    + multi
                        .iter()
                        .map(|m| {
                            64 + m.parts.iter().map(|p| p.bytes() + 48).sum::<usize>()
                                + m.touch.iter().map(|t| t.capacity() * 4).sum::<usize>()
                                + m.kids.iter().map(|t| t.capacity() * 4).sum::<usize>()
                        })
                        .sum::<usize>();
            }
            ids.push(cells.iter().enumerate().map(|(n, &(ci, cj))| ((ci as u32, cj as u32), n as u32)).collect());
            kinds.push(kind);
            touches.push(touch);
            multis.push(multi);
            prev = Some(labels);
        }
        let shift = (0..levels.len()).map(|l| base.shift(l)).map(|(a, b)| (a as i32, b as i32)).collect();
        let mut strict = Self {
            kind: kinds,
            touch: touches,
            relaxed: Vec::new(),
            multis,
            ids,
            shift,
            #[cfg(test)]
            stats,
            relax: false,
        };
        strict.relaxed = (0..levels.len()).map(|l| strict.relaxed_level(l)).collect();
        Ok(strict)
    }

    /// The `relaxed` bits of every frame cell at level index l (empty at level 1).
    fn relaxed_level(&self, l: usize) -> Array2<u8> {
        if l == 0 {
            return Array2::zeros((0, 0));
        }
        let kind = &self.kind[l];
        let (h, w) = kind.dim();
        let s = self.shift[l];
        let mut relaxed = Array2::<u8>::zeros((h, w));
        relaxed.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(r, mut row)| {
            for c in 0..w {
                if kind[[r, c]] != SINGLE {
                    continue;
                }
                let (i, j) = (r as i32 - s.0, c as i32 - s.1);
                let touch = self.touch[l][[r, c]];
                for k in 4..8 {
                    let (nr, nc) = (r as i64 + COLS[k] as i64, c as i64 + ROWS[k] as i64);
                    let inside = nr >= 0 && nc >= 0 && (nr as usize) < h && (nc as usize) < w;
                    if !inside || kind[[nr as usize, nc as usize]] != SINGLE || touch & (1 << k) != 0 {
                        continue;
                    }
                    if self.through(l, i, j, 0, k, 0) {
                        row[c] |= 1 << k;
                    }
                }
            }
        });
        relaxed
    }

    /// Kind of the tile's cell (i, j) at level index l > 0.
    #[inline]
    pub fn kind(&self, l: usize, i: i32, j: i32) -> u8 {
        let s = self.shift[l];
        self.kind[l][[(i + s.0) as usize, (j + s.1) as usize]]
    }

    /// Touch directions of the tile's cell (i, j) at level index l > 0.
    #[inline]
    fn touch(&self, l: usize, i: i32, j: i32) -> u8 {
        let s = self.shift[l];
        self.touch[l][[(i + s.0) as usize, (j + s.1) as usize]]
    }

    /// Pieces of the tile's cell (i, j) at level index l when it holds several.
    #[inline]
    pub fn multi(&self, l: usize, i: i32, j: i32) -> Option<&Multi> {
        self.body(l, self.multi_id(l, i, j))
    }

    /// Index of the tile's cell (i, j) at level index l among the level's cells holding several
    /// pieces, or `NONE`.
    #[inline]
    pub fn multi_id(&self, l: usize, i: i32, j: i32) -> u32 {
        if l == 0 || self.kind(l, i, j) != MULTI {
            return NONE;
        }
        let s = self.shift[l];
        self.ids[l][&((i + s.0) as u32, (j + s.1) as u32)]
    }

    /// Pieces of the cell with index `id` (`multi_id`) at level index l; `None` for `NONE`.
    #[inline]
    pub fn body(&self, l: usize, id: u32) -> Option<&Multi> {
        (id != NONE).then(|| &self.multis[l][id as usize])
    }

    /// `relaxed` bits of the tile's cell (i, j) at level index l > 0 holding one piece.
    #[inline]
    pub fn relaxed(&self, l: usize, i: i32, j: i32) -> u8 {
        let s = self.shift[l];
        self.relaxed[l][[(i + s.0) as usize, (j + s.1) as usize]]
    }

    /// The tile's cell at level index l + 1 that contains its cell (i, j) at level index l.
    #[inline]
    pub fn parent(&self, l: usize, i: i32, j: i32) -> (i32, i32) {
        let (s, t) = (self.shift[l], self.shift[l + 1]);
        ((i + s.0).div_euclid(2) - t.0, (j + s.1).div_euclid(2) - t.1)
    }

    /// Position of the tile's cell (i, j) at level index l relative to twice its level-(l + 1)
    /// cell `up`, as stored in `Multi::kids`.
    #[inline]
    pub fn rel(&self, l: usize, i: i32, j: i32, up: (i32, i32)) -> (i8, i8) {
        let (s, t) = (self.shift[l], self.shift[l + 1]);
        ((i + s.0 - 2 * (up.0 + t.0)) as i8, (j + s.1 - 2 * (up.1 + t.1)) as i8)
    }

    /// Number of pieces of the tile's cell (i, j) at level index l > 0.
    #[inline]
    fn pieces(&self, l: usize, i: i32, j: i32) -> usize {
        let s = self.shift[l];
        let (r, c) = (i + s.0, j + s.1);
        let kind = &self.kind[l];
        if r < 0 || c < 0 || r as usize >= kind.nrows() || c as usize >= kind.ncols() {
            return 0;
        }
        match kind[[r as usize, c as usize]] {
            0 => 0,
            SINGLE => 1,
            _ => self.multis[l][self.ids[l][&(r as u32, c as u32)] as usize].parts.len(),
        }
    }

    /// Whether piece `a` of the tile's cell (i, j) at level index l > 0 touches piece `b` of its
    /// neighbour in direction k.
    #[inline]
    pub fn touches(&self, l: usize, i: i32, j: i32, a: usize, k: usize, b: usize) -> bool {
        match (self.multi(l, i, j), self.multi(l, i + COLS[k], j + ROWS[k])) {
            (Some(m), _) => m.touch[a].contains(&(k as u8, b as u16)),
            (None, Some(m)) => m.touch[b].contains(&(OPP[k] as u8, a as u16)),
            (None, None) => self.touch(l, i, j) & (1 << k) != 0,
        }
    }

    /// Relaxed contact for diagonal direction k at level index l > 0: piece `a` of the tile's
    /// cell (i, j) and piece `b` of its neighbour both touch one piece of a cell sharing their
    /// corner, so a base-resolution route runs through that piece.
    #[inline(never)]
    pub fn through(&self, l: usize, i: i32, j: i32, a: usize, k: usize, b: usize) -> bool {
        corners(i, j, k).into_iter().any(|(ci, cj, first, second)| {
            (0..self.pieces(l, ci, cj)).any(|c| self.touches(l, i, j, a, first, c) && self.touches(l, ci, cj, c, second, b))
        })
    }

    /// Calls `f(b)` for every piece b of the neighbour in direction k that piece `a` of the tile's
    /// cell (i, j) at level index l > 0 touches, in ascending order: `touches` for every b at once.
    /// `mx` and `my` are the pieces (`multi`) of the cell and of the neighbour. Touch lists are
    /// sorted, and contact is symmetric (`touches_symmetric` test), so a cell of several pieces
    /// lists its own contacts and a single piece's are found in its neighbour's lists.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn contacts(&self, l: usize, i: i32, j: i32, a: usize, k: usize, mx: Option<&Multi>, my: Option<&Multi>, mut f: impl FnMut(u16)) {
        match (mx, my) {
            (Some(m), _) => {
                let list = &m.touch[a];
                let from = list.partition_point(|&(d, _)| (d as usize) < k);
                list[from..].iter().take_while(|&&(d, _)| d as usize == k).for_each(|&(_, b)| f(b));
            }
            (None, Some(m)) => {
                for (b, list) in m.touch.iter().enumerate() {
                    if list.binary_search(&(OPP[k] as u8, a as u16)).is_ok() {
                        f(b as u16);
                    }
                }
            }
            (None, None) => {
                if self.touch(l, i, j) & (1 << k) != 0 {
                    f(0);
                }
            }
        }
    }

    /// Pieces b of the diagonal neighbour in direction k (4..8) of the tile's cell (i, j) at level
    /// index l > 0 for which `through(l, i, j, a, k, b)` holds, pushed to `out` (unordered, with
    /// repeats): piece `a`'s contacts with each corner cell's pieces composed with theirs, so the
    /// work grows with the contacts rather than the pairs of pieces. `mx` is the cell's pieces.
    #[allow(clippy::too_many_arguments)]
    pub fn through_all(&self, l: usize, i: i32, j: i32, a: usize, k: usize, mx: Option<&Multi>, out: &mut Vec<u16>) {
        let my = self.multi(l, i + COLS[k], j + ROWS[k]);
        for (ci, cj, first, second) in corners(i, j, k) {
            let mc = self.multi(l, ci, cj);
            self.contacts(l, i, j, a, first, mx, mc, |c| {
                self.contacts(l, ci, cj, c as usize, second, mc, my, |b| out.push(b));
            });
        }
    }

    /// Touch directions of the tile's cell (i, j) at level index l; at level 1, the directions
    /// of its traversable neighbours (`passable`).
    #[inline]
    pub fn bits(&self, l: usize, i: i32, j: i32, passable: impl Fn(i32, i32) -> bool) -> u8 {
        if l > 0 {
            return self.touch(l, i, j);
        }
        let mut bits = 0u8;
        for k in 0..8 {
            if passable(i + COLS[k], j + ROWS[k]) {
                bits |= 1 << k;
            }
        }
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A U shape is one piece; two cells that do not touch inside the footprint are two pieces,
    // numbered in row-major order of their first cells; a diagonal contact joins
    #[test]
    fn splits_cells() {
        let mut u = Array2::from_elem((4, 8), false);
        for (r, c) in [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (3, 7)] {
            u[[r, c]] = true;
        }
        let (labels, kind) = label_level(&u, 4).unwrap();
        assert_eq!((kind[[0, 0]], kind[[0, 1]]), (SINGLE, SINGLE));
        assert_eq!((labels[[0, 0]], labels[[0, 2]], labels[[3, 7]]), (1, 1, 1));
        let mut two = Array2::from_elem((4, 4), false);
        two[[0, 3]] = true;
        two[[2, 0]] = true;
        let (labels, kind) = label_level(&two, 4).unwrap();
        assert_eq!(kind[[0, 0]], MULTI);
        assert_eq!((labels[[0, 3]], labels[[2, 0]]), (1, 2));
        two[[1, 1]] = true;
        two[[1, 2]] = true; // (2, 0) - (1, 1) - (1, 2) - (0, 3), diagonal steps
        assert_eq!(label_level(&two, 4).unwrap().1[[0, 0]], SINGLE);
    }

    // Two land strips separated by one NoData column touch only through the crossable column
    #[test]
    fn touch_needs_contact() {
        let build = |cond: &Array2<f32>| {
            let none = Array2::from_elem(cond.dim(), false);
            Strict::build(&Base::new(&none, cond, cond, &[1, 2, 4], (0, 0), None, None, &[])).unwrap()
        };
        let strips = Array2::from_shape_fn((8, 8), |(_, j)| if j != 4 { 0.5 } else { f32::NAN });
        // Level 4: cells (0, 0) and (0, 1); the gap is column 4, the first column of cell (0, 1)
        assert_eq!(build(&strips).touch(2, 0, 0) & 1, 0, "east neighbour across the gap");
        let open = Array2::from_elem((8, 8), 0.5f32);
        assert_eq!(build(&open).touch(2, 0, 0), 0b0001_0011, "east, south and south-east");
    }

    // Touch and kids lists, kept for the whole call, hold each entry once in order and no spare
    // capacity, on row stripes that give every coarse cell several pieces
    #[test]
    fn lists_compact() {
        fn tally<T: Ord>(lists: &[Vec<T>], used: &mut usize, held: &mut usize) {
            for list in lists {
                assert!(list.windows(2).all(|w| w[0] < w[1]), "unsorted or repeated entries");
                *used += list.len();
                *held += list.capacity();
            }
        }
        let rows = Array2::from_shape_fn((64, 64), |(i, _)| if i % 3 != 2 { 0.5f32 } else { f32::NAN });
        let none = Array2::from_elem(rows.dim(), false);
        let st = Strict::build(&Base::new(&none, &rows, &rows, &[1, 2, 4, 8, 16, 32], (0, 0), None, None, &[])).unwrap();
        let (mut cells, mut used, mut held) = (0, 0, 0);
        for m in st.multis.iter().flatten() {
            cells += 1;
            tally(&m.touch, &mut used, &mut held);
            tally(&m.kids, &mut used, &mut held);
        }
        assert!(cells > 100, "{cells} cells with several pieces");
        assert!(held <= used + used / 8, "{held} entries held for {used} used");
    }

    // The relaxed bits are `through` for every diagonal pair of single-piece cells that do not
    // touch, and 0 elsewhere, on random fragmented scenes with frame offsets
    #[test]
    fn relaxed_bits() {
        let mut rng = crate::testkit::Rng::new(5);
        let (mut set, mut checked) = (0, 0);
        for _ in 0..30 {
            let (h, w) = (24 + rng.below(40), 24 + rng.below(40));
            let land = rng.pick(&[0.25, 0.45, 0.6]);
            let cond = Array2::from_shape_fn((h, w), |_| if rng.chance(land) { 0.5f32 } else { f32::NAN });
            let none = Array2::from_elem(cond.dim(), false);
            let offsets = (rng.below(20), rng.below(20));
            let st = Strict::build(&Base::new(&none, &cond, &cond, &[1, 2, 4, 8, 16], offsets, None, None, &[])).unwrap();
            for l in 1..5 {
                let (fh, fw) = st.kind[l].dim();
                let s = st.shift[l];
                for r in 0..fh as i32 {
                    for c in 0..fw as i32 {
                        let (i, j) = (r - s.0, c - s.1);
                        let bits = st.relaxed(l, i, j);
                        for k in 0..8 {
                            let (nr, nc) = (r + COLS[k], c + ROWS[k]);
                            let inside = nr >= 0 && nc >= 0 && nr < fh as i32 && nc < fw as i32;
                            let single = |r: i32, c: i32| st.kind[l][[r as usize, c as usize]] == SINGLE;
                            let expected = k >= 4
                                && inside
                                && single(r, c)
                                && single(nr, nc)
                                && !st.touches(l, i, j, 0, k, 0)
                                && st.through(l, i, j, 0, k, 0);
                            assert_eq!(bits >> k & 1 == 1, expected, "level {l}, cell ({i}, {j}), direction {k}");
                            checked += 1;
                            set += expected as usize;
                        }
                    }
                }
            }
        }
        assert!(set > 50 && set < checked / 4, "{set} of {checked}");
    }

    /// Random fragmented scenes with frame offsets: their Strict and every level's frame cells.
    fn scenes(seed: u64, n: usize) -> Vec<Strict> {
        let mut rng = crate::testkit::Rng::new(seed);
        (0..n)
            .map(|_| {
                let (h, w) = (24 + rng.below(40), 24 + rng.below(40));
                let (land, dots) = (rng.pick(&[0.25, 0.45, 0.6]), rng.chance(0.3));
                let cond = Array2::from_shape_fn((h, w), |(r, c)| {
                    let on = rng.chance(land) && (!dots || (r % 2 == 0 && c % 2 == 0));
                    if on { 0.5f32 } else { f32::NAN }
                });
                let none = Array2::from_elem(cond.dim(), false);
                let offsets = (rng.below(20), rng.below(20));
                Strict::build(&Base::new(&none, &cond, &cond, &[1, 2, 4, 8, 16], offsets, None, None, &[])).unwrap()
            })
            .collect()
    }

    /// Calls `f(l, i, j)` for every cell with pieces at level index l > 0 whose neighbour in
    /// direction k has pieces, with the neighbour's tile coordinates.
    fn pairs_of(st: &Strict, mut f: impl FnMut(usize, i32, i32, usize, i32, i32)) {
        for l in 1..st.kind.len() {
            let (fh, fw) = st.kind[l].dim();
            let s = st.shift[l];
            for r in 0..fh as i32 {
                for c in 0..fw as i32 {
                    for k in 0..8 {
                        let (nr, nc) = (r + COLS[k], c + ROWS[k]);
                        let inside = nr >= 0 && nc >= 0 && nr < fh as i32 && nc < fw as i32;
                        if inside && st.kind[l][[r as usize, c as usize]] != 0 && st.kind[l][[nr as usize, nc as usize]] != 0 {
                            f(l, r - s.0, c - s.1, k, nr - s.0, nc - s.1);
                        }
                    }
                }
            }
        }
    }

    // Contact is symmetric, whichever cell's list answers it
    #[test]
    fn touches_symmetric() {
        let mut multi = 0;
        for st in scenes(9, 30) {
            pairs_of(&st, |l, i, j, k, ni, nj| {
                for a in 0..st.pieces(l, i, j) {
                    for b in 0..st.pieces(l, ni, nj) {
                        let ab = st.touches(l, i, j, a, k, b);
                        assert_eq!(ab, st.touches(l, ni, nj, b, OPP[k], a), "level {l}, ({i}, {j}) {a}, {k}, {b}");
                    }
                }
                multi += (st.pieces(l, i, j) > 1) as usize;
            });
        }
        assert!(multi > 1000, "{multi}");
    }

    // `contacts` and `through_all` list exactly the pieces `touches` and `through` accept
    #[test]
    fn contact_lists() {
        let (mut direct, mut relaxed) = (0, 0);
        for st in scenes(13, 30) {
            pairs_of(&st, |l, i, j, k, ni, nj| {
                let (mx, my) = (st.multi(l, i, j), st.multi(l, ni, nj));
                for a in 0..st.pieces(l, i, j) {
                    let pieces = 0..st.pieces(l, ni, nj);
                    let expected: Vec<u16> = pieces.clone().filter(|&b| st.touches(l, i, j, a, k, b)).map(|b| b as u16).collect();
                    let mut got = Vec::new();
                    st.contacts(l, i, j, a, k, mx, my, |b| got.push(b));
                    assert_eq!(got, expected, "contacts: level {l}, ({i}, {j}) {a}, {k}");
                    direct += got.len();
                    if k >= 4 {
                        let expected: Vec<u16> = pieces.filter(|&b| st.through(l, i, j, a, k, b)).map(|b| b as u16).collect();
                        let mut got = Vec::new();
                        st.through_all(l, i, j, a, k, mx, &mut got);
                        got.sort_unstable();
                        got.dedup();
                        assert_eq!(got, expected, "through: level {l}, ({i}, {j}) {a}, {k}");
                        relaxed += got.len();
                    }
                }
            });
        }
        assert!(direct > 1000 && relaxed > 1000, "{direct} {relaxed}");
    }

    // 65,536 isolated cells fit in level-256 cells (16,384 each) but not in one level-512 cell,
    // which is an error rather than a panic
    #[test]
    fn piece_limit() {
        let dots = Array2::from_shape_fn((512, 512), |(i, j)| i % 2 == 0 && j % 2 == 0);
        assert!(label_level(&dots, 256).is_some());
        assert!(label_level(&dots, 512).is_none());
        let cond = dots.mapv(|d| if d { 0.5f32 } else { f32::NAN });
        let none = Array2::from_elem(cond.dim(), false);
        let error = Strict::build(&Base::new(&none, &cond, &cond, &[1, 512], (0, 0), None, None, &[])).err().unwrap();
        assert_eq!(
            error.to_string(),
            "Strict NoData crossing allows at most 65534 separate pieces of land and crossable NoData in one overview cell, but a level-512 cell holds more; use levels up to 256"
        );
    }
}
