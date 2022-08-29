use std::{
    fmt::{Display, LowerExp},
    io::BufRead,
    io::BufReader,
    ops::{Index, IndexMut, Neg, Sub},
};

use approx::{abs_diff_ne, AbsDiffEq};

#[derive(PartialEq, Clone, Debug)]
pub struct Tensor3<T>(pub(crate) Vec<Vec<Vec<T>>>);

impl Tensor3<usize> {
    /// return a new i x j x k tensor
    pub fn zeros(i: usize, j: usize, k: usize) -> Self {
        Self(vec![vec![vec![0; k]; j]; i])
    }
}

// TODO could probably replace these fields with vectors and fc3 index formula
// if they're always symmetric. Then I don't have to do all the symmetry stuff
// myself, I can just sort the indices when I access them
impl Tensor3<f64> {
    /// return a new i x j x k tensor
    pub fn zeros(i: usize, j: usize, k: usize) -> Self {
        Self(vec![vec![vec![0.0; k]; j]; i])
    }

    pub fn print(&self) {
        println!("{}", self);
    }

    pub fn load(filename: &str) -> Self {
        let f =
            std::fs::File::open(filename).expect("failed to open tensor file");
        let lines = BufReader::new(f).lines();
        let mut ret = Vec::new();
        let mut buf = Vec::new();
        for line in lines.flatten() {
            let mut fields = line.split_whitespace().peekable();
            if fields.peek().is_none() {
                // in between chunks
                ret.push(buf);
                buf = Vec::new();
            } else {
                let row = fields
                    .map(|s| s.parse::<f64>().unwrap())
                    .collect::<Vec<_>>();
                buf.push(row);
            }
        }
        ret.push(buf);
        Self(ret)
    }

    pub fn equal(&self, other: &Self, eps: f64) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        for (i, mat) in self.0.iter().enumerate() {
            for (j, row) in mat.iter().enumerate() {
                for k in 0..row.len() {
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
}

impl<T> Tensor3<T>
where
    T: Copy,
{
    /// panics if either of the latter two dimensions is empty
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.0.len(), self.0[0].len(), self.0[0][0].len())
    }

    /// copy values across the 3D diagonals
    pub fn fill3b(&mut self) {
        for m in 0..3 {
            for n in 0..m {
                for p in 0..n {
                    self[(n, m, p)] = self[(m, n, p)];
                    self[(n, p, m)] = self[(m, n, p)];
                    self[(m, p, n)] = self[(m, n, p)];
                    self[(p, m, n)] = self[(m, n, p)];
                    self[(p, n, m)] = self[(m, n, p)];
                }
                self[(n, m, n)] = self[(m, n, n)];
                self[(n, n, m)] = self[(m, n, n)];
            }
            for p in 0..m {
                self[(m, p, m)] = self[(m, m, p)];
                self[(p, m, m)] = self[(m, m, p)];
            }
        }
    }

    pub fn fill3a(&mut self, nsx: usize) {
        for p in 0..nsx {
            for n in 0..p {
                for m in 0..n {
                    self[(n, m, p)] = self[(m, n, p)];
                    self[(n, p, m)] = self[(m, n, p)];
                    self[(m, p, n)] = self[(m, n, p)];
                    self[(p, m, n)] = self[(m, n, p)];
                    self[(p, n, m)] = self[(m, n, p)];
                }
                self[(n, p, n)] = self[(n, n, p)];
                self[(p, n, n)] = self[(n, n, p)];
            }
            for m in 0..p {
                self[(p, m, p)] = self[(m, p, p)];
                self[(p, p, m)] = self[(m, p, p)];
            }
        }
    }
}

impl<T> Index<(usize, usize, usize)> for Tensor3<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1][index.2]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for Tensor3<T> {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0][index.1][index.2]
    }
}

impl Neg for Tensor3<f64> {
    type Output = Tensor3<f64>;

    fn neg(self) -> Self::Output {
        let mut ret = self;
        for mat in &mut ret.0 {
            for row in mat {
                for col in row {
                    *col *= -1.0;
                }
            }
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

impl Display for Tensor3<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        for mat in &self.0 {
            for row in mat {
                for col in row {
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
        for mat in &self.0 {
            for row in mat {
                for col in row {
                    write!(f, "{:12.2e}", col)?;
                }
                writeln!(f)?;
            }
            writeln!(f)?;
            writeln!(f)?;
        }
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
