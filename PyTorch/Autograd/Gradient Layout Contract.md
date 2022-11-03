<div align='center'><font size='20'> Gradient Layout Contract </font></div>

# Gradient Layout Contract
- AccumulateGrad tries to stash strided (non-sparse) grads with memory layout (strides) such that variables and grads interact efficiently in later `optimizer` kernels, and grads interact efficiently with `c10d::Reducer.cpp`
- Specifically, AccumulateGrad tries to ensure the following (cf torch/csrc/autograd/utils/grad_layout_contract.h):
    1. if variable.is_non_overlapping_and_dense(), the stashed grad's strides match variable.
    2. else, stashed grad is rowmajor contiguous. If variable's grad does not exist (`!variable_grad.defined()`) AccumulateGrad steals `new_grad` if it's stealable and obeys the contract already, otherwise it deep copies `new_grad` into an obedient clone.
- If variable's grad already exists (`variable_grad.defined()`), new_grad must be added to `variable_grad`.  If we aren't setting up for double backward (`!GradMode::is_enabled()`), AccumulateGrad performs `variable_grad += new_grad` in-place, which keeps variable_grad's layout. We assume (hope) `variable_grad` was created obeying (1) or (2) at some point in the past.
- If we are setting up for double backward, AccumulateGrad updates the grad out-of-place via `variable_grad + new_grad` . TensorIterator `operator+` decides result's layout.  Typically TensorIterator matches strides of the first arg, so we once again assume (hope) `variable_grad` was originally created obeying (1) or (2).
- AccumulateGrad does not enforce the contract with 100% certainty. Examples:
    - If a user manually permutes a param or its grad, then runs a fwd+bwd, `variable_grad += new_grad` keeps `variable_grad`'s layout without
         rechecking the contract.
    - If TensorIterator changes its corner cases about `operator+`'s result (for example, giving more or less priority to channels_last inputs, see https://github.com/pytorch/pytorch/pull/37968) the result may not obey.
- Fortunately, if a given grad doesn't satisfy (1) or (2), the penalty is degraded performance in `Reducer.cpp` or `optimizer` kernels, not death by
  assert or silently bad numerics.

# torch/csrc/autograd/utils/grad_layout_contract.h
  <details> 
    <summary>Code for obeys_layout_contract</summary> 

  ```cpp
  // Checks if grad obeys the contract with variable.
  inline bool obeys_layout_contract(
      const at::Tensor& grad,
      const at::Tensor& variable) {
    TORCH_INTERNAL_ASSERT(!grad.is_sparse());
    TORCH_INTERNAL_ASSERT(!variable.is_sparse());
    TORCH_INTERNAL_ASSERT(!grad.is_sparse_csr());
    TORCH_INTERNAL_ASSERT(!variable.is_sparse_csr());

    if (variable.is_nested()) {
      // TODO: Nested Tensor does not have an implementation of detach. The
      // current implementation of nested tensor likely does obey the gradient
      // contract and should return true, but this would likely change in the
      // future
      return false;
    } else if (variable.is_non_overlapping_and_dense()) {
      // Only look at stride for dimensions that are not of size 1.
      const auto& grad_sizes = grad.sizes();
      const auto& grad_strides = grad.strides();
      const auto& variable_strides = variable.strides();
      for (const auto idx : c10::irange(grad_sizes.size())) {
        if (grad_sizes[idx] != 1) {
          if (grad_strides[idx] != variable_strides[idx]) {
            return false;
          }
        } else {
          // This should not be needed but we don't check if a Tensor has views
          // before stashing it. And 0-strided Tensors of size 1 are actually
          // views for ops like cat.
          // TODO: Actually detect views in the accumulateGrad function so that
          // this Tensor is not considered at all.
          if (grad_strides[idx] == 0) {
            return false;
          }
        }
      }
      return true;
    } else {
      return grad.is_contiguous(at::MemoryFormat::Contiguous);
    }
  }
  ```
  </details>

  <details> 
    <summary>Code for clone_obey_contract</summary> 

  ```cpp
  // Creates a clone of new_grad that obeys the contract with variable.
  // The clone should attach to new_grad's history if GradMode::is_enabled().
  inline at::Tensor clone_obey_contract(
      const at::Tensor& new_grad,
      const at::Tensor& variable) {
    if (variable.is_non_overlapping_and_dense()) {
      // (1)
      // Does this dicey-looking sequence attach the result to new_grad's
      // history if GradMode::is_enabled()?  Yes, and @alband says it should.
      return std::move(new_grad
                           .new_empty_strided(
                               variable.sizes(),
                               variable.strides(),
                               variable.options().memory_format(c10::nullopt))
                           .copy_(new_grad));
    } else {
      // (2)
      return new_grad.clone(at::MemoryFormat::Contiguous);
    }
  }
  ```
  </details>