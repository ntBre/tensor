use std::{
    fmt::{Display, LowerExp},
    ops::{Index, IndexMut, Sub},
};

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

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
}

#[derive(Clone)]
pub struct Tensor4 {
    pub data: Vec<f64>,
    d1: usize,
    d2: usize,
    d3: usize,
    d4: usize,
}

// TODO could probably replace these fields with vectors and fc3 index formula
// if they're always symmetric. Then I don't have to do all the symmetry stuff
// myself, I can just sort the indices when I access them
impl Tensor4 {
    /// return a new i x j x k x l tensor
    pub fn zeros(i: usize, j: usize, k: usize, l: usize) -> Self {
        Self {
            data: vec![0.0; i * j * k * l],
            d1: i,
            d2: j,
            d3: k,
            d4: l,
        }
    }

    pub fn print(&self) {
        println!("{}", self);
    }

    /// panics if any of the latter three dimensions is empty
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.d1, self.d2, self.d3, self.d4)
    }

    pub fn equal(&self, other: &Self, eps: f64) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        for i in 0..self.d1 {
            for j in 0..self.d2 {
                for k in 0..self.d3 {
                    for l in 0..self.d4 {
                        if f64::abs(self[(i, j, k, l)] - other[(i, j, k, l)])
                            > eps
                        {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    pub fn fill4a(&mut self, ny: usize) {
        for q in 0..ny {
            for p in 0..=q {
                for n in 0..=p {
                    for m in 0..=n {
                        self[(n, m, p, q)] = self[(m, n, p, q)];
                        self[(n, p, m, q)] = self[(m, n, p, q)];
                        self[(n, p, q, m)] = self[(m, n, p, q)];
                        self[(m, p, n, q)] = self[(m, n, p, q)];
                        self[(p, m, n, q)] = self[(m, n, p, q)];
                        self[(p, n, m, q)] = self[(m, n, p, q)];
                        self[(p, n, q, m)] = self[(m, n, p, q)];
                        self[(m, p, q, n)] = self[(m, n, p, q)];
                        self[(p, m, q, n)] = self[(m, n, p, q)];
                        self[(p, q, m, n)] = self[(m, n, p, q)];
                        self[(p, q, n, m)] = self[(m, n, p, q)];
                        self[(m, n, q, p)] = self[(m, n, p, q)];
                        self[(n, m, q, p)] = self[(m, n, p, q)];
                        self[(n, q, m, p)] = self[(m, n, p, q)];
                        self[(n, q, p, m)] = self[(m, n, p, q)];
                        self[(m, q, n, p)] = self[(m, n, p, q)];
                        self[(q, m, n, p)] = self[(m, n, p, q)];
                        self[(q, n, m, p)] = self[(m, n, p, q)];
                        self[(q, n, p, m)] = self[(m, n, p, q)];
                        self[(m, q, p, n)] = self[(m, n, p, q)];
                        self[(q, m, p, n)] = self[(m, n, p, q)];
                        self[(q, p, m, n)] = self[(m, n, p, q)];
                        self[(q, p, n, m)] = self[(m, n, p, q)];
                    }
                }
            }
        }
    }

    /// panics if self is empty
    pub fn max(&self) -> f64 {
        let mut max = self[(0, 0, 0, 0)];
        for i in 0..self.d1 {
            for j in 0..self.d2 {
                for k in 0..self.d3 {
                    for l in 0..self.d4 {
                        let col = self[(i, j, k, l)];
                        if col > max {
                            max = col;
                        }
                    }
                }
            }
        }
        max
    }
}

impl Index<(usize, usize, usize, usize)> for Tensor4 {
    type Output = f64;

    #[inline]
    fn index(&self, index: (usize, usize, usize, usize)) -> &Self::Output {
        let index = self.index_inner(index);
        &self.data[index]
    }
}

impl Tensor4 {
    /// index = x + y * D1 + z * D1 * D2 + t * D1 * D2 * D3, but use
    /// Horner's rule
    #[inline]
    fn index_inner(&self, index: (usize, usize, usize, usize)) -> usize {
        let (x, y, z, t) = index;
        x + self.d1 * (y + self.d2 * (z + self.d3 * t))
    }

    /// returns the slice of the first two dimensions from `start` to `end` with
    /// fixed final dimensions `k` and `l`
    pub fn submatrix(
        &self,
        start: (usize, usize),
        end: (usize, usize),
        k: usize,
        l: usize,
    ) -> &[f64] {
        let (a, b) = start;
        let (c, d) = end;
        let start = self.index_inner((a, b, k, l));
        let end = self.index_inner((c, d, k, l));
        &self.data[start..end]
    }

    pub fn set_submatrix(
        &mut self,
        start: (usize, usize),
        end: (usize, usize),
        k: usize,
        l: usize,
        data: &[f64],
    ) {
        let (a, b) = start;
        let (c, d) = end;
        let start = self.index_inner((a, b, k, l));
        let end = self.index_inner((c, d, k, l));
        self.data[start..end].copy_from_slice(data);
    }
}

impl IndexMut<(usize, usize, usize, usize)> for Tensor4 {
    fn index_mut(
        &mut self,
        index: (usize, usize, usize, usize),
    ) -> &mut Self::Output {
        let index = self.index_inner(index);
        &mut self.data[index]
    }
}

impl Sub<Tensor4> for Tensor4 {
    type Output = Tensor4;

    fn sub(self, rhs: Tensor4) -> Self::Output {
        if self.shape() != rhs.shape() {
            panic!("Tensor4::sub: dimension mismatch");
        }
        let (a, b, c, d) = self.shape();
        let mut ret = Self::Output::zeros(a, b, c, d);
        for i in 0..self.d1 {
            for j in 0..self.d2 {
                for k in 0..self.d3 {
                    for l in 0..self.d4 {
                        let col = self[(i, j, k, l)];
                        ret[(i, j, k, l)] = col - rhs[(i, j, k, l)];
                    }
                }
            }
        }
        ret
    }
}

impl Display for Tensor4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        let (mi, mj, mk, ml) = self.shape();
        for i in 0..mi {
            writeln!(f, "I = {i:5}")?;
            for j in 0..mj {
                for k in 0..mk {
                    for l in 0..ml {
                        write!(f, "{:12.6}", self[(i, j, k, l)])?;
                    }
                    writeln!(f)?;
                }
                writeln!(f)?;
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl LowerExp for Tensor4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        // for (i, tens) in self.0.iter().enumerate() {
        //     writeln!(f, "I = {i:5}")?;
        //     writeln!(f, "{:e}", Tensor3(tens.clone()))?;
        // }
        Ok(())
    }
}
