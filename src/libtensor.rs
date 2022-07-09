// TODO: create a macro tensofr!([4,3,2,1,3])
use std::convert::{From};
use std::iter::FromIterator;
use std::ops::{Add, Mul, Div, Sub, Neg};
use crate::libmap::{Gradient, Identity, ElementWiseMul, Unit};

/// ```
/// let x = Tensor::<Array>::from([4,1,3]);
/// assert_eq!(x.grad, Tensor.val([1,1,1]));
/// 
/// let y = Tensor::<List>::from([4,1,3]);
/// let z = x + y
/// 
/// ```
/// 
/// grad is also type `<T>`, since val is semantically
/// viewed as the resulting computation, hence why the default of grad_fn is also 
/// the identity function
/// 
/// For part of the implementation of the tensor we keep
/// in mind that it as a function evaluated for some value.
/// 

#[derive(Debug, Copy, Clone)]
struct Tensor<T, F>
where
    F: Gradient<T>
{
    val: T,
    grad_fn: F,
    grad: T,
}

///
/// L = [x, y] * [z, w] 
/// B = L + [c,b] = C
/// B/dx = B/L * L/x + B/C * C/x
/// ElementWiseMul
/// d L/dx = dL/dL * d L/dx = [z, 0]
///  


impl<T, F> Tensor<T, F> where F: Gradient<T> {

    fn new(val: T, grad_fn: F, grad: T) -> Self {
        Self {
            val,
            grad_fn,
            grad
        }
    }
    
}

impl<'a, T, F> Mul for Tensor<T, F> where F: Gradient<T> {
    type Output = Tensor<T, ElementWiseMul<'a, T, F>>;
}

impl<T> FromIterator<T> for Tensor<Vec<T>, Identity> where T: Unit<T> + Default + Copy, Identity: Gradient<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut val: Vec<T> = Vec::new();
        let mut grad: Vec<T> = Vec::new(); 

        for i in iter {
            val.push(i);
            grad.push(T::default());
        }

        Self { val, grad_fn: Identity, grad }
    }
}

impl<T> Default for Tensor<T, Identity> where T: Default + Unit<T>, Identity: Gradient<T> {
    fn default() -> Self {
        Self {
           val: T::default(),
           grad_fn: Identity,
           grad: T::default(), 
        }
    } 
}



#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn tensor_from_array() {
        let x = Tensor::from_iter([3.,2.,1.].into_iter());
        dbg!(Identity::grad(x.val));
        dbg!(x);
    }

    #[test]
    fn identity_gradient() {
        panic!("Not implemented")
        //let x = tensor!([3,2,1]);
        //assert_eq!(x.grad, tensor!([1,1,1]))
    }
}
