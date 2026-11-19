//! NoData crossing, mode 1 (landmass).
//!
//! Land and crossable NoData (see `crossing`) that touch form groups, and a focal cell's window
//! admits only cells holding its own group, with that group's values in a cell holding several.
//! Groups are labelled over the tile alone: land joined only outside the tile counts as separate.
use crate::crossing::{Acc, Base, Part};
use ndarray::parallel::prelude::*;
use ndarray::Array2;
use std::collections::HashMap;

/// Coarse cell holding traversable cells of more than one group.
pub const MIXED: u32 = u32::MAX;

/// Test diagnostics of one gate.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
#[allow(dead_code)] // read through Debug in island::proto_bench
pub struct GateStats {
    pub crossable: usize, // crossable NoData base cells
    pub groups: usize,
    pub mixed: usize, // mixed coarse cells over all levels
    pub bytes: usize, // retained by the gate: group arrays and the mixed table
}

pub struct Gate {
    /// Group, MIXED or 0 of each frame cell, per level (ascending from 1). Level 1 holds each
    /// base cell's group, 0 off the domain.
    levels: Vec<Array2<u32>>,
    /// Values of each group of a mixed cell, keyed by level index and frame cell
    mixed: HashMap<(u8, u32, u32), Vec<(u32, Part)>>,
    shift: Vec<(usize, usize)>, // frame offset of the tile's cells, per level
    #[cfg(test)]
    pub stats: GateStats,
}

/// 8-connected components of `mask`, numbered from 1; 0 elsewhere.
pub fn label(mask: &Array2<bool>) -> (Array2<u32>, usize) {
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    let (h, w) = mask.dim();
    let mut lab = Array2::<u32>::zeros((h, w));
    let mut parent: Vec<u32> = vec![0];
    for i in 0..h {
        for j in 0..w {
            if !mask[[i, j]] {
                continue;
            }
            let mut best = 0u32;
            let before = [
                (i.wrapping_sub(1), j.wrapping_sub(1)),
                (i.wrapping_sub(1), j),
                (i.wrapping_sub(1), j + 1),
                (i, j.wrapping_sub(1)),
            ];
            for (ni, nj) in before {
                if ni < h && nj < w && lab[[ni, nj]] != 0 {
                    let root = find(&mut parent, lab[[ni, nj]]);
                    if best == 0 {
                        best = root;
                    } else if root != best {
                        let (a, b) = (best.min(root), best.max(root));
                        parent[b as usize] = a;
                        best = a;
                    }
                }
            }
            if best == 0 {
                best = parent.len() as u32;
                parent.push(best);
            }
            lab[[i, j]] = best;
        }
    }
    let mut remap = vec![0u32; parent.len()];
    let mut next = 0u32;
    for x in 1..parent.len() {
        let root = find(&mut parent, x as u32) as usize;
        if remap[root] == 0 {
            next += 1;
            remap[root] = next;
        }
        remap[x] = remap[root];
    }
    lab.mapv_inplace(|l| remap[l as usize]);
    (lab, next as usize)
}

/// Group of each cell one level up: the children's common group, MIXED when they differ.
fn coarsen(prev: &Array2<u32>) -> Array2<u32> {
    let (h, w) = prev.dim();
    Array2::from_shape_fn((h.div_ceil(2), w.div_ceil(2)), |(i, j)| {
        let mut group = 0u32;
        for (r, c) in [(2 * i, 2 * j), (2 * i, 2 * j + 1), (2 * i + 1, 2 * j), (2 * i + 1, 2 * j + 1)] {
            if r >= h || c >= w {
                continue;
            }
            match prev[[r, c]] {
                0 => {}
                MIXED => return MIXED,
                v if group == 0 => group = v,
                v if v != group => return MIXED,
                _ => {}
            }
        }
        group
    })
}

impl Gate {
    /// Group arrays for every level and per-group values of every mixed coarse cell.
    pub fn build(base: &Base) -> Self {
        let (labels, _groups) = label(&base.domain);
        let mut arrays = vec![labels];
        let mut mixed = HashMap::new();
        for l in 1..base.levels.len() {
            let next = coarsen(&arrays[l - 1]);
            let size = base.levels[l] as usize;
            let cells: Vec<(usize, usize)> =
                next.indexed_iter().filter(|(_, &g)| g == MIXED).map(|(ij, _)| ij).collect();
            let found: Vec<_> = cells
                .par_iter()
                .map(|&(i, j)| {
                    let (r0, r1, c0, c1) = base.footprint(i, j, size);
                    let mut sums: Vec<(u32, Acc)> = Vec::new();
                    for r in r0..r1 {
                        for c in c0..c1 {
                            let group = arrays[0][[r, c]];
                            if group == 0 {
                                continue;
                            }
                            let k = match sums.iter().position(|p| p.0 == group) {
                                Some(k) => k,
                                None => {
                                    sums.push((group, Acc::default()));
                                    sums.len() - 1
                                }
                            };
                            base.add(&mut sums[k].1, r, c);
                        }
                    }
                    let parts: Vec<(u32, Part)> = sums.iter().map(|(g, acc)| (*g, base.finish(acc))).collect();
                    ((l as u8, i as u32, j as u32), parts)
                })
                .collect();
            mixed.extend(found);
            arrays.push(next);
        }
        #[cfg(test)]
        let stats = GateStats {
            crossable: base.crossable,
            groups: _groups,
            mixed: mixed.len(),
            bytes: arrays.iter().map(|a| a.len() * 4).sum::<usize>()
                + mixed.values().map(|v| 48 + v.iter().map(|(_, p)| 4 + p.bytes()).sum::<usize>()).sum::<usize>(),
        };
        let shift = (0..base.levels.len()).map(|l| base.shift(l)).collect();
        Self {
            levels: arrays,
            mixed,
            shift,
            #[cfg(test)]
            stats,
        }
    }

    /// Group, MIXED or 0 of the tile's cell `at` at level index `l`.
    #[inline]
    pub fn group(&self, l: usize, at: [usize; 2]) -> u32 {
        let s = self.shift[l];
        self.levels[l][[at[0] + s.0, at[1] + s.1]]
    }

    /// Values of `group` in the tile's mixed cell `at` at level index `l`.
    #[inline]
    pub fn part(&self, l: usize, at: [usize; 2], group: u32) -> Option<&Part> {
        let s = self.shift[l];
        let key = (l as u8, (at[0] + s.0) as u32, (at[1] + s.1) as u32);
        self.mixed[&key].iter().find(|p| p.0 == group).map(|p| &p.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_touching_corners() {
        let mask = Array2::from_shape_fn((4, 4), |(i, j)| (i == j) || (i == 0 && j == 3));
        let (lab, n) = label(&mask);
        assert_eq!(n, 2);
        assert_eq!(lab[[0, 0]], lab[[3, 3]]);
        assert_ne!(lab[[0, 0]], lab[[0, 3]]);
    }
}
