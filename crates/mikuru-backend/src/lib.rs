//! Burn integration boundary for Mikuru LM.
//!
//! The initial backend is Burn Flex on CPU. This crate deliberately exposes a
//! small boundary so the model crates do not need to decide the device policy.

use burn::prelude::*;

/// Run a tiny FP32 forward-and-backward calculation on Burn Flex.
///
/// This verifies the backend and autodiff path. It is not a model-training run.
pub fn run_smoke_test() -> Result<(), String> {
    let device = Device::flex().autodiff();

    let input = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device).require_grad();
    let loss = (input.clone() * 3.0).sum();
    let gradients = loss.backward();
    let input_gradient = input
        .grad(&gradients)
        .ok_or_else(|| "Burn did not return the expected input gradient".to_owned())?;

    println!("Burn backend: Flex (CPU)");
    println!("Training precision: FP32");
    println!("Gradient result: {:?}", input_gradient.into_data());
    println!("Burn forward + autodiff smoke test passed.");

    Ok(())
}
