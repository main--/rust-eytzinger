//! This crate implements the "eytzinger" (aka BFS) array layout where
//! a binary search tree is stored by layer (instead of as a sorted array).
//! This can have significant performance benefits
//! (see [Khuong, Paul-Virak, and Pat Morin. "Array layouts for comparison-based searching."][1]).
//!
//! # Usage
//!
//! ```
//! use eytzinger::SliceExt;
//! let mut data = [0, 1, 2, 3, 4, 5, 6];
//! data.eytzingerize(&mut eytzinger::permutation::InplacePermutator);
//! assert_eq!(data, [3, 1, 5, 0, 2, 4, 6]);
//! assert_eq!(data.eytzinger_search(&5), Some(2));
//! assert_eq!(data.eytzinger_search_by(|x| x.cmp(&6)), Some(6));
//! ```
//!
//! [1]: https://arxiv.org/pdf/1509.05053.pdf

#![warn(missing_docs, missing_debug_implementations)]

use std::cmp::{Ord, Ordering};
use std::borrow::Borrow;
use permutation::*;

/// The basic building blocks this crate is made of.
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
    #[inline]
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
    #[inline]
    pub fn index_to_node(i: usize) -> (usize, usize) {
        let ipk = (i + 2).next_power_of_two().trailing_zeros() as usize;
        let li = i + 1 - (1 << (ipk - 1));
        (ipk, li)
    }

    /// Given an array size (`n`) and an index into the eytzinger array (`ì`),
    /// this function computes the index of this value in a sorted array.
    ///
    /// This is simply `index_to_node` + `get_permutation_element_by_node`.
    #[inline]
    pub fn get_permutation_element(n: usize, i: usize) -> usize {
        let (ipk, li) = index_to_node(i);
        get_permutation_element_by_node(n, ipk, li)
    }
}

/// Abstractions around applying generic permutations using generic implementations.
pub mod permutation {
    use std::iter::{Cloned, Enumerate};
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

        #[inline]
        fn iterable(&self) -> Self::Iter {
            self.iter().cloned()
        }

        #[inline]
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
    #[derive(Clone, Copy, Debug, Default)]
    pub struct InplacePermutator;

    impl<T, P: ?Sized + Permutation> Permutator<T, P> for InplacePermutator {
        #[inline]
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


    /// Simple permutator that stack-allocates a copy of the data (using recursion).
    ///
    /// Worst-case runtime is `O(n)`, but this takes `O(n)` stack space so it WILL NOT work for large permutations.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct StackCopyPermutator;

    fn recursive_permute<T: Clone, P: ?Sized + Permutation>(data: &mut [T], permutation: &mut Enumerate<P::Iter>) {
        if let Some((i, p)) = permutation.next() {
            let item = data[p].clone();
            recursive_permute::<T, P>(data, permutation);
            data[i] = item;
        }
    }

    impl<T: Clone, P: ?Sized + Permutation> Permutator<T, P> for StackCopyPermutator {
        #[inline]
        fn permute(&mut self, data: &mut [T], permutation: &P) {
            let mut iter = permutation.iterable().enumerate();
            recursive_permute::<T, P>(data, &mut iter);
        }
    }


    /// Simple permutator that heap-allocates a copy of the data.
    ///
    /// Worst-case runtime is `O(n)`, taking `O(n)` heap space in a reusable buffer.
    /// This is an acceptable permutator for large permutations, provided that the data
    /// is (efficiently) cloneable.
    #[derive(Debug, Default)]
    pub struct HeapCopyPermutator<T> {
        buffer: Vec<T>,
    }

    impl<T: Clone, P: ?Sized + Permutation> Permutator<T, P> for HeapCopyPermutator<T> {
        #[inline]
        fn permute(&mut self, data: &mut [T], permutation: &P) {
            self.buffer.clear();
            self.buffer.extend(permutation.iterable().map(|i| data[i].clone()));
            for (i, t) in self.buffer.drain(..).enumerate() {
                data[i] = t;
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
    #[inline]
    pub fn new(size: usize) -> PermutationGenerator {
        PermutationGenerator {
            size,
            ipk: 1,
            li: 0,
        }
    }
}

impl Iterator for PermutationGenerator {
    type Item = usize;

    #[inline]
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

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let k2 = 1 << (self.ipk - 1);
        let size = self.size - (k2 + self.li - 1);
        (size, Some(size))
    }
}
impl ExactSizeIterator for PermutationGenerator {}

impl Permutation for PermutationGenerator {
    type Iter = PermutationGenerator;

    #[inline]
    fn iterable(&self) -> PermutationGenerator {
        self.clone()
    }
    #[inline]
    fn index(&self, i: usize) -> usize {
        foundation::get_permutation_element(self.size, i)
    }
}

/// Converts a sorted array to its eytzinger representation.
///
/// # Example
///
/// ```
/// let mut data = [0, 1, 2, 3, 4, 5, 6];
/// eytzinger::eytzingerize(&mut data, &mut eytzinger::permutation::InplacePermutator);
/// assert_eq!(data, [3, 1, 5, 0, 2, 4, 6]);
/// ```
#[inline]
pub fn eytzingerize<T, P: Permutator<T, PermutationGenerator>>(data: &mut [T], permutator: &mut P) {
    let len = data.len();
    permutator.permute(data, &PermutationGenerator::new(len))
}

/// Eytzinger extension methods for slices.
pub trait SliceExt<T> {
    /// Converts an already sorted array to its eytzinger representation.
    ///
    /// # Example
    ///
    /// ```
    /// use eytzinger::SliceExt;
    /// let mut data = [0, 1, 2, 3, 4, 5, 6];
    /// data.eytzingerize(&mut eytzinger::permutation::InplacePermutator);
    /// assert_eq!(data, [3, 1, 5, 0, 2, 4, 6]);
    /// ```
    fn eytzingerize<P: Permutator<T, PermutationGenerator>>(&mut self, permutator: &mut P);

    /// Binary searches this eytzinger slice for a given element.
    ///
    /// If the value is found then `Some` is returned, containing the index of the matching element;
    /// if the value is not found then `None` is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use eytzinger::SliceExt;
    /// let s = [3, 1, 5, 0, 2, 4, 6];
    /// assert_eq!(s.eytzinger_search(&5), Some(2));
    /// assert_eq!(s.eytzinger_search(&6), Some(6));
    /// assert_eq!(s.eytzinger_search(&7), None);
    /// ```
    fn eytzinger_search<Q: ?Sized>(&self, x: &Q) -> Option<usize> where Q: Ord, T: Borrow<Q>;

    /// Binary searches this eytzinger slice with a comparator function.
    ///
    /// The comparator function should implement an order consistent with the sort order
    /// of the underlying eytzinger slice, returning an order code that indicates whether
    /// its argument is `Less`, `Equal` or `Greater` than the desired target.
    ///
    /// If a matching value is found then `Some` is returned, containing the index of the
    /// matching element; if no match is found then `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use eytzinger::SliceExt;
    /// let s = [3, 1, 5, 0, 2, 4, 6];
    /// assert_eq!(s.eytzinger_search_by(|x| x.cmp(&5)), Some(2));
    /// assert_eq!(s.eytzinger_search_by(|x| x.cmp(&6)), Some(6));
    /// assert_eq!(s.eytzinger_search_by(|x| x.cmp(&7)), None);
    /// ```
    fn eytzinger_search_by<'a, F>(&'a self, f: F) -> Option<usize> where F: FnMut(&'a T) -> Ordering, T: 'a;

    /// Binary searches this sorted slice with a key extraction function.
    ///
    /// Assumes that the slice is eytzinger-sorted by the key, for instance with
    /// `slice::sort_by_key` combined with `eytzinger::eytzingerize` using the
    /// same key extraction function.
    ///
    /// If a matching value is found then `Some` is returned, containing the index of the
    /// matching element; if no match is found then `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use eytzinger::SliceExt;
    /// let s = [(3, 'd'), (1, 'b'), (5, 'f'), (0, 'a'), (2, 'c'), (4, 'e'), (6, 'g')];
    /// assert_eq!(s.eytzinger_search_by_key(&'f', |&(_, b)| b), Some(2));
    /// assert_eq!(s.eytzinger_search_by_key(&'g', |&(_, b)| b), Some(6));
    /// assert_eq!(s.eytzinger_search_by_key(&'x', |&(_, b)| b), None);
    /// ```
    fn eytzinger_search_by_key<'a, B, F, Q: ?Sized>(&'a self, b: &Q, f: F) -> Option<usize>
        where B: Borrow<Q>,
              F: FnMut(&'a T) -> B,
              Q: Ord,
              T: 'a;
}

/// Binary searches this eytzinger slice with a comparator function.
///
/// The comparator function should implement an order consistent with the sort order
/// of the underlying eytzinger slice, returning an order code that indicates whether
/// its argument is `Less`, `Equal` or `Greater` than the desired target.
///
/// If a matching value is found then `Some` is returned, containing the index of the
/// matching element; if no match is found then `None` is returned.
///
/// # Examples
///
/// ```
/// use eytzinger::eytzinger_search_by;
/// let s = [3, 1, 5, 0, 2, 4, 6];
/// assert_eq!(eytzinger_search_by(&s, |x| x.cmp(&3)), Some(0));
/// assert_eq!(eytzinger_search_by(&s, |x| x.cmp(&5)), Some(2));
/// assert_eq!(eytzinger_search_by(&s, |x| x.cmp(&6)), Some(6));
/// assert_eq!(eytzinger_search_by(&s, |x| x.cmp(&7)), None);
/// ```
#[inline]
pub fn eytzinger_search_by<'a, T: 'a, F>(data: &'a [T], f: F) -> Option<usize>
    where F: FnMut(&'a T) -> Ordering {
    eytzinger_search_by_impl(data, f)
}

#[inline]
#[cfg(not(feature = "branchless"))]
fn eytzinger_search_by_impl<'a, T: 'a, F>(data: &'a [T], mut f: F) -> Option<usize>
    where F: FnMut(&'a T) -> Ordering {
    let mut i = 0;
    loop {
        match data.get(i) {
            Some(ref v) => {
                match f(v) {
                    Ordering::Equal => return Some(i),
                    o => {
                        // I was hoping the optimizer could handle this but it can't
                        // So here goes the evil hack: Ordering is -1/0/1
                        // So we use this dirty trick to map this to +2/X/+1
                        let o = o as usize;
                        let o = (o >> 1) & 1;
                        i = 2 * i + 1 + o;
                    }
                };
            }
            None => return None,
        }
    }
}

#[inline]
#[cfg(feature = "branchless")]
fn eytzinger_search_by_impl<'a, T: 'a, F>(data: &'a [T], mut f: F) -> Option<usize>
    where F: FnMut(&'a T) -> Ordering {
    let mut i = 0;
    while i < data.len() {
        let v = &data[i]; // this range check is optimized out :D
        i = match f(v) {
            Ordering::Greater | Ordering::Equal => 2 * i + 1,
            Ordering::Less => 2 * i + 2,
        };
    }

    // magic from the paper to fix up the (incomplete) final tree layer
    // (only difference is that we recheck f() because this is exact search)
    let p = i + 1;
    let j = p >> (1 + (!p).trailing_zeros());
    if j != 0 && (f(&data[j - 1]) == Ordering::Equal) {
        Some(j - 1)
    } else {
        None
    }
}

impl<T> SliceExt<T> for [T] {
    #[inline]
    fn eytzingerize<P: Permutator<T, PermutationGenerator>>(&mut self, permutator: &mut P) {
        eytzingerize(self, permutator)
    }

    #[inline]
    fn eytzinger_search<Q: ?Sized>(&self, x: &Q) -> Option<usize> where Q: Ord, T: Borrow<Q> {
        self.eytzinger_search_by(|e| e.borrow().cmp(x))
    }

    #[inline]
    fn eytzinger_search_by<'a, F>(&'a self, f: F) -> Option<usize> where F: FnMut(&'a T) -> Ordering, T: 'a {
        eytzinger_search_by(self, f)
    }

    #[inline]
    fn eytzinger_search_by_key<'a, B, F, Q: ?Sized>(&'a self, b: &Q, mut f: F) -> Option<usize>
        where B: Borrow<Q>,
              F: FnMut(&'a T) -> B,
              Q: Ord,
              T: 'a {
        self.eytzinger_search_by(|k| f(k).borrow().cmp(b))
    }
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

    #[test]
    fn simple_heap_copy_permutation() {
        let permutation: &[usize] = &[4, 2, 3, 0, 1];
        let mut data = [1, 2, 3, 4, 5];
        HeapCopyPermutator::default().permute(&mut data, &permutation);
        assert_eq!(data, [5, 3, 4, 1, 2]);
    }

    #[test]
    fn search_negative() {
        let data: &[i32] = &[6, 2, 10, 0, 4, 8, 12];
        for i in -10..20 {
            let expected = data.iter().position(|&x| x == i);
            assert_eq!(expected, data.eytzinger_search(&i));
        }
    }

    fn test_permutation<P: Default>(junk: Vec<usize>) -> bool where for<'a> P: Permutator<usize, &'a [usize]> {
        // first create a permutation from the random array
        let mut perm: Vec<_> = (0..junk.len()).collect();
        perm.sort_by_key(|&i| junk[i]);

        // now test
        let mut data: Vec<_> = (0..perm.len()).collect();
        P::default().permute(data.as_mut_slice(), &perm.as_slice());
        data == perm
    }

    quickcheck! {
        fn inplace_permutation(junk: Vec<usize>) -> bool {
            test_permutation::<InplacePermutator>(junk)
        }

        fn stack_permutation(junk: Vec<usize>) -> bool {
            test_permutation::<StackCopyPermutator>(junk)
        }

        fn heap_copy_permutation(junk: Vec<usize>) -> bool {
            test_permutation::<HeapCopyPermutator<_>>(junk)
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

        fn search_works(data: Vec<usize>) -> bool {
            let mut data = data;
            data.sort();
            data.dedup();
            data.eytzingerize(&mut InplacePermutator);

            data.iter().enumerate().all(|(i, v)| data.eytzinger_search(v) == Some(i))
        }
    }
}
