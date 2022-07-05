
/// Var = struct
/// 




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
    fn grad(&self) -> T;
}

struct Tensor<T, F>
where
    F: Gradient<T>
{
    val: T,
    grad_fn: Option<F>,
    grad: f64,
}

struct Tensor2<T, F>
where
    F: Gradient<T>
{
    val: T,
    grad_fn: F,
    grad: T,
}

impl<T, F> Tensor2<T, F> where F: Gradient<T> {

    fn new(val: T, grad_fn: F, grad: T) -> Self {
        Self {
            val,
            grad_fn,
            grad
        }
    }
}

impl<T, F> Default for Tensor2<T,F> where F: Gradient<T> + Default, T: Default {
    fn default() -> Self{
        Self {
           val: T::default(),
           grad_fn: F::default(),
           grad: T::default(), 
        }
    } 
}

struct IdentityFunction;

trait Identity<T> {

}

impl<T> Gradient<T> for IdentityFunction{
    fn grad(&self) -> T {
        T::identity()
    }
}

fn identity() -> f64 {
    1.
}

// perhaps we have an id for 
// looking up the derivative of a var with
// respect to antoher var
// returns -> Option<f64> where we get None
// when the id does not exist

trait Derivative {
    fn diff(&self) -> f64;
}

// the graph can be a list of length N
// for nn


// idx icr from function
#[derive(Debug)]
struct Test<'a> {
    c: Option<&'a mut Self>
}

impl<'a> Test<'a> {
    fn new(c: Option<&'a mut Self>) -> Self {
        Self{
            c
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    
    #[test]
    fn create_var() {

        let var= Var::new(3.0, vec![|x: f64| -> f64 {x}], 3.0);
        dbg!(var.grad_fn[0](4.0)); 

    }

    #[test]
    fn create_test() {
        let mut test = Test::new(None);
        let test2 = Test::new(Some(&mut test));
        dbg!(test2.c);

    }
}
