use std::{
    fmt::{Display, LowerExp},
    io::BufRead,
    io::BufReader,
    ops::{Index, IndexMut, Neg, Sub},
};

use approx::{abs_diff_ne, AbsDiffEq};

#[derive(Clone, PartialEq)]
pub struct Tensor3<T> {
    pub data: Vec<T>,
    d1: usize,
    d2: usize,
    d3: usize,
}

impl Tensor3<usize> {
    /// return a new i x j x k tensor
    pub fn zeros(i: usize, j: usize, k: usize) -> Self {
        Self {
            data: vec![0; i * j * k],
            d1: i,
            d2: j,
            d3: k,
        }
    }
}

// TODO could probably replace these fields with vectors and fc3 index formula
// if they're always symmetric. Then I don't have to do all the symmetry stuff
// myself, I can just sort the indices when I access them
impl Tensor3<f64> {
    /// return a new i x j x k tensor
    pub fn zeros(i: usize, j: usize, k: usize) -> Self {
        Self {
            data: vec![0.0; i * j * k],
            d1: i,
            d2: j,
            d3: k,
        }
    }

    pub fn print(&self) {
        println!("{}", self);
    }

    pub fn load(filename: &str) -> Self {
        let f =
            std::fs::File::open(filename).expect("failed to open tensor file");
        let lines = BufReader::new(f).lines();
        let mut hold = Vec::new();
        let mut buf = Vec::new();
        for line in lines.flatten() {
            let mut fields = line.split_whitespace().peekable();
            if fields.peek().is_none() {
                // in between chunks
                hold.push(buf);
                buf = Vec::new();
            } else {
                let row = fields
                    .map(|s| s.parse::<f64>().unwrap())
                    .collect::<Vec<_>>();
                buf.push(row);
            }
        }
        hold.push(buf);
        let a = hold.len();
        let b = hold[0].len();
        let c = hold[0][0].len();
        let mut ret = Self::zeros(a, b, c);
        for i in 0..a {
            for j in 0..b {
                for k in 0..c {
                    ret[(i, j, k)] = hold[i][j][k];
                }
            }
        }
        ret
    }

    pub fn equal(&self, other: &Self, eps: f64) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        for i in 0..self.d1 {
            for j in 0..self.d2 {
                for k in 0..self.d3 {
                    if f64::abs(self[(i, j, k)] - other[(i, j, k)]) > eps {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// apply `f` to every element of `self` and return a new Tensor3
    pub fn map<F>(&self, mut f: F) -> Self
    where
        F: FnMut(f64) -> f64,
    {
        let (a, b, c) = self.shape();
        let mut ret = Self::zeros(a, b, c);
        for i in 0..a {
            for j in 0..b {
                for k in 0..c {
                    ret[(i, j, k)] = f(self[(i, j, k)]);
                }
            }
        }
        ret
    }

    /// panics if `self` is empty
    pub fn max(&self) -> f64 {
        let (a, b, c) = self.shape();
        let mut ret = self[(0, 0, 0)];
        for i in 0..a {
            for j in 0..b {
                for k in 0..c {
                    if self[(i, j, k)] > ret {
                        ret = self[(i, j, k)];
                    }
                }
            }
        }
        ret
    }

    /// return the absolute value of self
    pub fn abs(&self) -> Self {
        self.map(|s| s.abs())
    }
}

impl<T> Tensor3<T>
where
    T: Copy,
{
    /// panics if either of the latter two dimensions is empty
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.d1, self.d2, self.d3)
    }

    pub fn set_submatrix(
        &mut self,
        start: (usize, usize),
        end: (usize, usize),
        k: usize,
        data: &[T],
    ) {
        let (a, b) = start;
        let (c, d) = end;
        let start = self.index_inner((a, b, k));
        let end = self.index_inner((c, d, k));
        self.data[start..end].copy_from_slice(data);
    }
}

impl<T> Tensor3<T> {
    pub fn index_inner(&self, index: (usize, usize, usize)) -> usize {
        let (x, y, z) = index;
        let index = x + self.d1 * (y + self.d2 * z);
        index
    }

    /// returns the slice of the first two dimensions from `start` to `end` with
    /// fixed final dimension `k`
    pub fn submatrix(
        &self,
        start: (usize, usize),
        end: (usize, usize),
        k: usize,
    ) -> &[T] {
        let (a, b) = start;
        let (c, d) = end;
        let start = self.index_inner((a, b, k));
        let end = self.index_inner((c, d, k));
        &self.data[start..end]
    }
}

impl<T> Index<(usize, usize, usize)> for Tensor3<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.data[self.index_inner(index)]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for Tensor3<T> {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        let i = self.index_inner(index);
        &mut self.data[i]
    }
}

impl Neg for Tensor3<f64> {
    type Output = Tensor3<f64>;

    fn neg(self) -> Self::Output {
        let mut ret = self;
        for col in &mut ret.data {
            *col *= -1.0;
        }
        ret
    }
}

impl Sub for Tensor3<f64> {
    type Output = Self;

    /// panics if self and rhs are not the same size
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape(), rhs.shape());
        let (a, b, c) = self.shape();
        let mut ret = Self::zeros(a, b, c);
        for i in 0..a {
            for j in 0..b {
                for k in 0..c {
                    ret[(i, j, k)] = self[(i, j, k)] - rhs[(i, j, k)];
                }
            }
        }
        ret
    }
}

impl Display for Tensor3<usize> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        for i in 0..self.d1 {
            for j in 0..self.d2 {
                for k in 0..self.d3 {
                    let col = self[(i, j, k)];
                    write!(f, "{:5}", col)?;
                }
                writeln!(f)?;
            }
            writeln!(f)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Display for Tensor3<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        for i in 0..self.d1 {
            for j in 0..self.d2 {
                for k in 0..self.d3 {
                    let col = self[(i, j, k)];
                    write!(f, "{:12.6}", col)?;
                }
                writeln!(f)?;
            }
            writeln!(f)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

impl LowerExp for Tensor3<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        // for mat in &self.0 {
        //     for row in mat {
        //         for col in row {
        //             write!(f, "{:12.2e}", col)?;
        //         }
        //         writeln!(f)?;
        //     }
        //     writeln!(f)?;
        //     writeln!(f)?;
        // }
        Ok(())
    }
}

impl AbsDiffEq for Tensor3<f64> {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let (a, b, c) = self.shape();
        for i in 0..a {
            for j in 0..b {
                for k in 0..c {
                    if abs_diff_ne!(
                        self[(i, j, k)],
                        other[(i, j, k)],
                        epsilon = epsilon
                    ) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}
