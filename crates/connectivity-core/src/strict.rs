//! NoData crossing, mode 2 (strict gap width).
//!
//! Traversable base cells are land plus crossable NoData (see `crossing`). Every coarse cell is
//! split into its pieces: the 8-connected components of its traversable base cells inside its
//! footprint. A piece is a node; pieces are linked, at the same level and to the next level,
//! only where their base cells overlap or are 8-adjacent. Every graph path therefore follows a
//! base-resolution path that crosses no NoData gap wider than the permitted width, at every
//! level and inside one landmass as well as between landmasses.
//!
//! Storage: per level above 1, one byte of piece kind and one byte of touch directions per frame
//! cell, plus a table for cells holding several pieces. Level 1 needs nothing: its pieces are the
//! traversable base cells, read from the traversal overview. Level 2 cells (2 x 2) always hold one
//! piece, because all cells of a 2 x 2 block are 8-adjacent.
use crate::crossing::{Acc, Base, Part};
use crate::graph::{COLS, ROWS};
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
    multi: Vec<CellMap<Multi>>, // keyed by frame cell
    shift: Vec<(i32, i32)>,                 // frame offset of the tile's cells, per level
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
        let mut multis = vec![CellMap::default()];
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
            let multi: CellMap<Multi> = cells
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
                    ((ci as u32, cj as u32), Multi { parts, touch, kids })
                })
                .collect();
            #[cfg(test)]
            {
                stats.multi.push(multi.len());
                stats.pieces.push(multi.values().map(|m| m.parts.len()).sum());
                stats.bytes += 2 * kind.len()
                    + multi
                        .values()
                        .map(|m| {
                            64 + m.parts.iter().map(|p| p.bytes() + 48).sum::<usize>()
                                + m.touch.iter().map(|t| t.capacity() * 4).sum::<usize>()
                                + m.kids.iter().map(|t| t.capacity() * 4).sum::<usize>()
                        })
                        .sum::<usize>();
            }
            kinds.push(kind);
            touches.push(touch);
            multis.push(multi);
            prev = Some(labels);
        }
        let shift = (0..levels.len()).map(|l| base.shift(l)).map(|(a, b)| (a as i32, b as i32)).collect();
        Ok(Self {
            kind: kinds,
            touch: touches,
            multi: multis,
            shift,
            #[cfg(test)]
            stats,
            relax: false,
        })
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
        if l == 0 || self.kind(l, i, j) != MULTI {
            return None;
        }
        let s = self.shift[l];
        Some(&self.multi[l][&((i + s.0) as u32, (j + s.1) as u32)])
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
            _ => self.multi[l][&(r as u32, c as u32)].parts.len(),
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
        let (di, dj) = (COLS[k], ROWS[k]);
        let dir = |y: i32, x: i32| (0..8).find(|&m| (COLS[m], ROWS[m]) == (y, x)).unwrap();
        [(i + di, j, dir(di, 0), dir(0, dj)), (i, j + dj, dir(0, dj), dir(di, 0))].into_iter().any(|(ci, cj, first, second)| {
            (0..self.pieces(l, ci, cj)).any(|c| self.touches(l, i, j, a, first, c) && self.touches(l, ci, cj, c, second, b))
        })
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
        for m in st.multi.iter().flat_map(|level| level.values()) {
            cells += 1;
            tally(&m.touch, &mut used, &mut held);
            tally(&m.kids, &mut used, &mut held);
        }
        assert!(cells > 100, "{cells} cells with several pieces");
        assert!(held <= used + used / 8, "{held} entries held for {used} used");
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
