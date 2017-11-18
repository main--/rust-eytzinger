// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(test)]

extern crate test;
extern crate eytzinger;

use eytzinger::SliceExt;
use test::black_box;
use test::Bencher;

enum Cache {
    L1,
    L2,
    L3,
}

fn eytzinger_search<F>(b: &mut Bencher, cache: Cache, mapper: F)
    where F: Fn(usize) -> usize
{
    let size = match cache {
        Cache::L1 => 1000, // 8kb
        Cache::L2 => 10_000, // 80kb
        Cache::L3 => 1_000_000, // 8Mb
    };
    let mut v = (0..size).map(&mapper).collect::<Vec<_>>();
    v.eytzingerize(&mut eytzinger::permutation::InplacePermutator);
    let mut r = 0usize;
    b.iter(move || {
        // LCG constants from https://en.wikipedia.org/wiki/Numerical_Recipes.
        r = r.wrapping_mul(1664525).wrapping_add(1013904223);
        // Lookup the whole range to get 50% hits and 50% misses.
        let i = mapper(r % size);
        black_box(v.eytzinger_search(&i).is_some());
    })
}

#[bench]
fn eytzinger_search_l1(b: &mut Bencher) {
    eytzinger_search(b, Cache::L1, |i| i * 2);
}

#[bench]
fn eytzinger_search_l2(b: &mut Bencher) {
    eytzinger_search(b, Cache::L2, |i| i * 2);
}

#[bench]
fn eytzinger_search_l3(b: &mut Bencher) {
    eytzinger_search(b, Cache::L3, |i| i * 2);
}

#[bench]
fn eytzinger_search_l1_with_dups(b: &mut Bencher) {
    eytzinger_search(b, Cache::L1, |i| i / 16 * 16);
}

#[bench]
fn eytzinger_search_l2_with_dups(b: &mut Bencher) {
    eytzinger_search(b, Cache::L2, |i| i / 16 * 16);
}

#[bench]
fn eytzinger_search_l3_with_dups(b: &mut Bencher) {
    eytzinger_search(b, Cache::L3, |i| i / 16 * 16);
}
