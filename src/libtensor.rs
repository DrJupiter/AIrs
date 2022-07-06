// TODO: create a macro tensofr!([4,3,2,1,3])
use std::convert::{From};
use std::iter::FromIterator;

#[derive(Debug)]
struct Var<F> 
where
    //F: Fn(f64) -> f64
    F: Fn(f64) -> f64
{
    val: f64,
    grad_fn: Vec<F>, // const N based on parents 
    grad: f64,
}

impl<F> Var<F> where F: Fn(f64) -> f64 {
    fn new(val: f64, grad_fn: Vec<F>, grad: f64) -> Self {
        Self {
            val,
            grad_fn,
            grad
        }
    }
}

trait Gradient<T> {
    fn grad(tensor: T) -> T;
}

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


impl<T, const N: usize> From<[T; N]> for Tensor<[T; N], Identity> where T: Default + Unit<T> + Copy {
    fn from(array: [T;N]) -> Self {
       Self {
       val: array,
       grad_fn: Identity,
       grad: [T::default(); N]
        }
    }
}

impl<T> FromIterator<T> for Tensor<Vec<T>, Identity> where T: Unit<T> + Default, Identity: Gradient<T> {
    fn from_iter<T: IntoIterator<Item = T>>(iter: T) -> Self {
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

#[derive(Debug)]
struct Identity;

trait Unit<U> {
    fn unit() -> U;
}

impl Unit<f64> for f64 {
    fn unit() -> f64 {
        1f64
    }
}

impl<T, const N: usize> Unit<[T; N]> for [T; N] where T: Unit<T> + Copy {
    fn unit() -> [T; N] {
        [T::unit(); N]
    }
}

impl<T, const N: usize> Gradient<[T; N]> for Identity where T: Unit<T> + Copy {
    fn grad(_tensor: [T; N]) -> [T; N] {
       <[T;N]>::unit()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn tensor_from_array() {
        let x = Tensor::from([3.,2.,1.]);
        dbg!(x);
    }

    #[test]
    fn identity_gradient() {
        panic!("Not implemented")
        //let x = tensor!([3,2,1]);
        //assert_eq!(x.grad, tensor!([1,1,1]))
    }
}
