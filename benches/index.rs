#![feature(test)]

use rand::Rng;
use tensor::Tensor4;

extern crate test;
#[bench]
fn bench_index(b: &mut test::Bencher) {
    let n = 9;
    let t = Tensor4::zeros(n, n, n, n);
    let mut rng = rand::thread_rng();
    b.iter(|| {
        let a = n as f64 * rng.gen::<f64>();
        let b = n as f64 * rng.gen::<f64>();
        let c = n as f64 * rng.gen::<f64>();
        let d = n as f64 * rng.gen::<f64>();
        t[(a as usize, b as usize, c as usize, d as usize)]
    });
}
