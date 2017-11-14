#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_magic() {
        for (i, &v) in [0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 7, 8].iter().enumerate() {
            assert_eq!(magic(i + 1, 1, 1), v);
        }
        for (i, &v) in [0, 0, 1, 1, 1, 1, 2, 3, 3].iter().enumerate() {
            assert_eq!(magic(i + 2, 2, 1), v);
        }
        for (i, &v) in [2, 3, 4, 5, 5, 6, 7, 8].iter().enumerate() {
            assert_eq!(magic(i + 3, 2, 3), v);
        }
        for (i, &v) in [0, 0, 0, 0, 1, 1, 1].iter().enumerate() {
            assert_eq!(magic(i + 4, 3, 1), v);
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
    fn test_reference_permutations() {
        for &array in REF_PERMUTATIONS {
            let permut: Vec<_> = PermutationGenerator::new(array.len()).collect();
            assert_eq!(array, permut.as_slice());
        }
    }
}

#[inline(never)]
fn magic(n: usize, ipk: usize, zk: usize) -> usize {
    // k = zk * 2^-ipk
    
    let last_power_of_two = (n + 2).next_power_of_two() / 2;

    let y = (last_power_of_two >> (ipk - 1)) * zk;
    let kp = y >> 1;
    //let kp = (last_power_of_two >> ipk) * zk;
    let x = kp + last_power_of_two; // (1+k) * last_power_of_two
    let x = x.saturating_sub(n + 1);

    //println!("n={} x={} y={} z={} kp={} lpot={}", n, x,y,z, kp, last_power_of_two);
    y - x - 1
}

struct PermutationGenerator {
    size: usize,
    ipk: usize,
    li: usize,
}

impl PermutationGenerator {
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

        let zk = self.li * 2 + 1;

        self.li += 1;

        Some(magic(self.size, self.ipk, zk))
    }
}

/*
fn get_permutation(target: &mut [usize]) {
    /*
    let size = target.len();
    //let mut v = target;
    let mut ipk = 1;
    let mut li = 0;
    for i in 0..size {
        if li >= (1 << (ipk - 1)) {
            li = 0;
            ipk += 1;
        }

        let zk = li * 2 + 1;

        //
        //v.push(magic(size, ipk, zk));
        target[i] = magic(size, ipk, zk);
        li += 1;
    }
    v*/
    //PermutationGenerator::new(
}*/

fn main() {
    /*
    for ipk in 1..3 {
        for n in 1..20 {
            println!("{}", magic(n, ipk, 1));
        }
        println!();
    }

    for n in 1..20 {
        println!("{}", magic(n, 2, 3));
    }
    println!();*/
    for i in 0..20 {
        //let mut permut = Vec::with_capacity
        let permut: Vec<_> = PermutationGenerator::new(i).collect();
        println!("{:?}", permut);
    }
}
