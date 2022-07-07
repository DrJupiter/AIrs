
pub trait Gradient<T> {
    fn grad(tensor: T) -> T;
}


#[derive(Debug)]
pub struct Identity;

/* 
impl<T, const N: usize> Gradient<[T; N]> for Identity where T: Unit<T> + Copy {
    fn grad(_tensor: [T; N]) -> [T; N] {
       <[T;N]>::unit()
    }
}

impl<T> Gradient<Vec<T>> for Identity where T: Unit<T> + Copy {
    fn grad(_tensor: Vec<T>) -> Vec<T> {
       vec![<T>::unit();_tensor.len()]
    }
}
*/

impl<T> Gradient<T> for Identity where T: Unit<T> {
    fn grad(_tensor: T) -> T {
        T::unit()
    }
}

impl<T> Gradient<Vec<T>> for Identity where T: Unit<T> + Copy {
    fn grad(tensor: Vec<T>) -> Vec<T> {
        vec![T::unit();tensor.len()]
    }
}

pub trait Unit<U> {
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

//impl<U> Unit<Vec<U>> for Vec<U> where U: Unit<U> + Copy {
//    fn unit() -> Vec<U> {
//        vec![U::unit();1]
//    }
//}