use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

use crate::Tensor4;

pub struct Tensor5 {
    pub data: Vec<Tensor4>,
}

impl Tensor5 {
    pub fn zeros(i: usize, j: usize, k: usize, l: usize, m: usize) -> Self {
        Self { data: vec![Tensor4::zeros(j, k, l, m); i] }
    }
}

impl Index<(usize, usize, usize, usize, usize)> for Tensor5 {
    type Output = f64;

    fn index(
        &self,
        index: (usize, usize, usize, usize, usize),
    ) -> &Self::Output {
        let (i, j, k, l, m) = index;
        &self.data[i][(j, k, l, m)]
    }
}

impl IndexMut<(usize, usize, usize, usize, usize)> for Tensor5 {
    fn index_mut(
        &mut self,
        index: (usize, usize, usize, usize, usize),
    ) -> &mut Self::Output {
        let (i, j, k, l, m) = index;
        &mut self.data[i][(j, k, l, m)]
    }
}

impl Display for Tensor5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        for (i, d) in self.data.iter().enumerate() {
            write!(f, "H = {i}")?;
            write!(f, "{}", d)?;
        }
        Ok(())
    }
}
