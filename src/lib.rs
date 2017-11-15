//! This crate implements the "eytzinger" (aka BFS) array layout where
//! a binary search tree is stored by layer (instead of as a sorted array).
//! This can have significant performance benefits
//! (see [Khuong, Paul-Virak, and Pat Morin. "Array layouts for comparison-based searching."][1]).
//!
//! # Usage
//!
//! ```
//! let mut data = [0, 1, 2, 3, 4, 5, 6];
//! eytzinger::eytzingerize(&mut data, &mut eytzinger::permutation::InplacePermutator);
//! assert_eq!(data, [3, 1, 5, 0, 2, 4, 6]);
//! ```
//!
//! [1]: https://arxiv.org/pdf/1509.05053.pdf

#![warn(missing_docs, missing_debug_implementations)]

use permutation::*;

/// The basic building blocks this is made of.
pub mod foundation {
    /// Given an array size (`n`), tree layer index (`ipk`) and element index (`li`),
    /// this function computes the index of this value in a sorted array.
    ///
    /// This is basically the core of this crate, everything else is trivial.
    /// Also, you usually want to use `PermutationGenerator` instead for convenience.
    ///
    /// # How it works
    ///
    /// This computes the magic function:
    ///
    /// ```text
    /// f(n, k) = 2^floor(log2(n+1)) * 2k  -  max(0, (1+k) * 2^floor(log2(n+1)) - (n + 1))  -  1
    /// (where n ∈ ℕ, k ∈ [0, 1])
    /// ```
    ///
    /// Because this is integer math: `k = zk * 2^-ipk`
    /// And because we only care about certain values of zk: `zk = li * 2 + 1`
    ///
    /// Even though I discovered this on my own and am quite certain that it's correct,
    /// I only have a very vague feeling about **why** it works. If you want to understand this
    /// (you really don't!), have a look at this sequence:
    ///
    /// ```text
    /// a_n = (2n - 2^floor(log2(2n)) + 1) / 2^floor(log2(2n))
    /// (1/2, 1/4, 3/4, 1/8, 3/8, 5/8, 7/8, 1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16, ...)
    /// ```
    ///
    /// The basic idea is that this sequence basically establishes a mapping between a sorted array
    /// and an eytzinger array: If you look at the plot sideways (literally), you can see the tree
    /// with all its layers. The only secret sauce here is `2^floor(log2(x))` which is elementary to
    /// get exponentially growing windows (to build tree layers).
    pub fn get_permutation_element_by_node(n: usize, ipk: usize, li: usize) -> usize {
        let zk = li * 2 + 1;
        // k = zk * 2^-ipk

        let last_power_of_two = (n + 2).next_power_of_two() / 2;

        let y = (last_power_of_two >> (ipk - 1)) * zk;
        let kp = y >> 1;
        let x = kp + last_power_of_two; // (1+k) * last_power_of_two
        let x = x.saturating_sub(n + 1);

        //println!("n={} x={} y={} z={} kp={} lpot={}", n, x,y,z, kp, last_power_of_two);
        y - x - 1
    }

    /// Converts an index in an eytzinger array to the corresponding tree coordinates `(ipk, li)`.
    pub fn index_to_node(i: usize) -> (usize, usize) {
        let ipk = (i + 2).next_power_of_two().trailing_zeros() as usize;
        let li = i + 1 - (1 << (ipk - 1));
        (ipk, li)
    }

    /// Given an array size (`n`) and an index into the eytzinger array (`ì`),
    /// this function computes the index of this value in a sorted array.
    ///
    /// This is simply `index_to_node` + `get_permutation_element_by_node`.
    pub fn get_permutation_element(n: usize, i: usize) -> usize {
        let (ipk, li) = index_to_node(i);
        get_permutation_element_by_node(n, ipk, li)
    }
}

/// Abstractions around applying generic permutations using generic implementations.
pub mod permutation {
    use std::iter::Cloned;
    use std::slice::Iter;

    /// A generic permutation.
    pub trait Permutation {
        /// An iterator through the permutation.
        /// This may be more efficient than indexing a counter.
        type Iter: Iterator<Item=usize>;

        /// Get an iterator.
        fn iterable(&self) -> Self::Iter;
        /// Index into this permutation.
        fn index(&self, i: usize) -> usize;
    }

    impl<'a> Permutation for &'a [usize] {
        type Iter = Cloned<Iter<'a, usize>>;

        fn iterable(&self) -> Self::Iter {
            self.iter().cloned()
        }

        fn index(&self, i: usize) -> usize {
            self[i]
        }
    }

    /// A generic permutator.
    pub trait Permutator<T, P: ?Sized + Permutation> {
        /// Applies the given permutation to the given array.
        fn permute(&mut self, data: &mut [T], permutation: &P);
    }

    /// Simple permutator that does not allocate.
    ///
    /// Worst-case runtime is in `O(n^2)`, so you should only use this for small permutations.
    #[derive(Clone, Copy, Debug)]
    pub struct InplacePermutator;

    impl<T, P: ?Sized + Permutation> Permutator<T, P> for InplacePermutator {
        fn permute(&mut self, data: &mut [T], permutation: &P) {
            for (i, mut p) in permutation.iterable().enumerate() {
                while p < i {
                    p = permutation.index(p);
                }

                if p > i {
                    data.swap(i, p);
                }
            }
        }
    }
}



/// Generates a permutation that transforms a sorted array into an eytzinger array.
///
/// This is an iterator which yields a permutation (indexes into the sorted array)
/// in the order of an eytzinger array.
#[derive(Clone, Debug)]
pub struct PermutationGenerator {
    size: usize,
    ipk: usize,
    li: usize,
}

impl PermutationGenerator {
    /// Generate a new permutation for a sorted array of a given size.
    fn new(size: usize) -> PermutationGenerator {
        PermutationGenerator {
            size,
            ipk: 1,
            li: 0,
        }
    }
}

impl Iterator for PermutationGenerator {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let k2 = 1 << (self.ipk - 1);

        if k2 + self.li - 1 >= self.size {
            return None;
        }

        if self.li >= k2 {
            self.li = 0;
            self.ipk += 1;
        }

        let li = self.li;
        self.li += 1;
        Some(foundation::get_permutation_element_by_node(self.size, self.ipk, li))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let k2 = 1 << (self.ipk - 1);
        let size = self.size - (k2 + self.li - 1);
        (size, Some(size))
    }
}
impl ExactSizeIterator for PermutationGenerator {}

impl Permutation for PermutationGenerator {
    type Iter = PermutationGenerator;

    fn iterable(&self) -> PermutationGenerator {
        self.clone()
    }
    fn index(&self, i: usize) -> usize {
        foundation::get_permutation_element(self.size, i)
    }
}

/// Converts a sorted array to its eytzinger representation.
pub fn eytzingerize<T, P: Permutator<T, PermutationGenerator>>(data: &mut [T], permutator: &mut P) {
    let len = data.len();
    permutator.permute(data, &PermutationGenerator::new(len))
}


#[cfg(test)]
#[macro_use]
extern crate quickcheck;
#[cfg(test)]
mod tests {
    use super::*;
    use super::foundation::*;

    #[test]
    fn magic() {
        for (i, &v) in [0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 7, 8].iter().enumerate() {
            assert_eq!(get_permutation_element_by_node(i + 1, 1, 0), v);
        }
        for (i, &v) in [0, 0, 1, 1, 1, 1, 2, 3, 3].iter().enumerate() {
            assert_eq!(get_permutation_element_by_node(i + 2, 2, 0), v);
        }
        for (i, &v) in [2, 3, 4, 5, 5, 6, 7, 8].iter().enumerate() {
            assert_eq!(get_permutation_element_by_node(i + 3, 2, 1), v);
        }
        for (i, &v) in [0, 0, 0, 0, 1, 1, 1].iter().enumerate() {
            assert_eq!(get_permutation_element_by_node(i + 4, 3, 0), v);
        }
    }

    const REF_PERMUTATIONS: &[&'static [usize]] = &[
        &[],
        &[0],
        &[1, 0],
        &[1, 0, 2],
        &[2, 1, 3, 0],
        &[3, 1, 4, 0, 2],
        &[3, 1, 5, 0, 2, 4],
        &[3, 1, 5, 0, 2, 4, 6],
        &[4, 2, 6, 1, 3, 5, 7, 0],
        &[5, 3, 7, 1, 4, 6, 8, 0, 2],
        &[6, 3, 8, 1, 5, 7, 9, 0, 2, 4],
        &[7, 3, 0xb, 1, 5, 9, 0xd, 0, 2, 4, 6, 8, 0xa, 0xc, 0xe],
    ];

    #[test]
    fn reference_permutations() {
        for &array in REF_PERMUTATIONS {
            let permut: Vec<_> = PermutationGenerator::new(array.len()).collect();
            assert_eq!(array, permut.as_slice());
        }
    }

    #[test]
    fn eytzingerize_simple() {
        let mut permutator = InplacePermutator;
        for &array in REF_PERMUTATIONS {
            let mut payload: Vec<_> = (0..array.len()).collect();
            eytzingerize(payload.as_mut_slice(), &mut permutator);
            assert_eq!(payload, array);
        }
    }

    const NODE_INDEXES: &[(usize, usize)] = &[
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
        (4, 5),
        (4, 6),
        (4, 7),
    ];

    #[test]
    fn calc_index() {
        for (i, &x) in NODE_INDEXES.iter().enumerate() {
            assert_eq!(x, index_to_node(i));
        }
    }

    #[test]
    fn simple_inplace_permutation() {
        let permutation: &[usize] = &[4, 2, 3, 0, 1];
        let mut data = [1, 2, 3, 4, 5];
        InplacePermutator.permute(&mut data, &permutation);
        assert_eq!(data, [5, 3, 4, 1, 2]);
    }

    quickcheck! {
        fn inplace_permutation(junk: Vec<usize>) -> bool {
            // first create a permutation from the random array
            let mut perm: Vec<_> = (0..junk.len()).collect();
            perm.sort_by_key(|&i| junk[i]);

            // now test
            let mut data: Vec<_> = (0..perm.len()).collect();
            InplacePermutator.permute(data.as_mut_slice(), &perm.as_slice());
            data == perm
        }

        fn eytzinger_tree_invariants(length: usize) -> bool {
            let perm: Vec<_> = PermutationGenerator::new(length).collect();

            let mut todo = Vec::new();
            todo.push((0, 0..length));
            let mut checked = 0;
            while let Some((i, range)) = todo.pop() {
                match perm.get(i) {
                    Some(&v) => {
                        if !(range.start <= v && v < range.end) { return false; }
                        todo.push((2 * (i + 1) - 1, range.start..v));
                        todo.push((2 * (i + 1), v..range.end));
                        checked += 1;
                    }
                    None => continue,
                }
            }

            checked == length
        }
    }
}
