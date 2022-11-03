<div align='center'><font size='20'> AccumulateGrad </font></div>

# AccumulateGrad
- Subclassing `torch::autograd::Node`
- `grad_fn` for leaf Tensor
- Accumulate gradient from `Autograd Engine` following [Gradient Layout Contract](Gradient%20Layout%20Contract.md)
  <details> 
    <summary>Code for Declaration & accumulateGrad</summary> 

    ```cpp
    struct TORCH_API AccumulateGrad : public Node {
    explicit AccumulateGrad(Variable variable_);

    variable_list apply(variable_list&& grads) override;

    static at::Tensor callHooks(const Variable& variable, at::Tensor new_grad) {
      for (auto& hook : impl::hooks(variable)) {
        new_grad = (*hook)({new_grad})[0];
      }
      return new_grad;
    }

    template <typename T>
    static void accumulateGrad(
        const Variable& variable,
        at::Tensor& variable_grad,
        const at::Tensor& new_grad,
        size_t num_expected_refs,
        const T& update_grad) {
      if (!variable_grad.defined()) {
        if (!GradMode::is_enabled() && !new_grad.is_sparse() &&
            !new_grad.is_sparse_csr() &&
            !(variable.is_sparse_csr() && new_grad.layout() == at::kStrided) &&
            new_grad.use_count() <= num_expected_refs &&
            (new_grad.is_mkldnn() ||
             utils::obeys_layout_contract(new_grad, variable))) {
          // we aren't setting up for double-backward
          // not sparse
          // no other user-visible tensor references new_grad
          // new_grad obeys the "Gradient Layout Contract", there has a special
          // case, For MKLDNN tensor, which is a opaque tensor, assuming it obeys
          // layout_contract. Under these conditions, we can steal new_grad
          // without a deep copy.
          update_grad(new_grad.detach());
        } else if (
            !GradMode::is_enabled() && new_grad.is_sparse() &&
            new_grad._indices().is_contiguous() &&
            new_grad._values().is_contiguous() &&
            // Use count for indices and values should always be <=1 since the
            // SparseTensor should be the only one holding a reference to these.
            new_grad._indices().use_count() <= 1 &&
            new_grad._values().use_count() <= 1 &&
            new_grad.use_count() <= num_expected_refs) {
          // Can't detach sparse tensor (since metadata changes are not allowed
          // after detach), so just create a new one for the grad which is a
          // shallow copy. We need a shallow copy so that modifying the original
          // grad tensor doesn't modify the grad we accumulate.
          // We only skip clone if indices and values themselves are contiguous
          // for backward compatiblity reasons. Since without this optimization,
          // earlier we would clone the entire SparseTensor which cloned indices
          // and values.
          // For details see https://github.com/pytorch/pytorch/issues/34375.
          update_grad(at::_sparse_coo_tensor_unsafe(
              new_grad._indices(),
              new_grad._values(),
              new_grad.sizes(),
              new_grad.options()));
        } else {
          if (new_grad.is_sparse() || new_grad.is_sparse_csr() ||
              new_grad.is_nested()) {
            update_grad(new_grad.clone());
          } else {
            if (new_grad.is_mkldnn()) {
              update_grad(new_grad.clone());
            } else {
              // Deep copies new_grad according to the "Gradient Layout Contract."
              update_grad(utils::clone_obey_contract(new_grad, variable));
            }
          }
        }
      } else if (!GradMode::is_enabled()) {
        // This case is not strictly necessary, but it makes the first-order only
        // case slightly more efficient.
        if (variable_grad.is_sparse() && !new_grad.is_sparse()) {
          // If `variable_grad` is sparse and `new_grad` is not sparse, their
          // sum is not sparse, and we must change the TensorImpl type of
          // `variable_grad` for it to store the result. However, changing the
          // TensorImpl type of a tensor requires changing the tensor itself, and
          // thus in this case we have to change the grad tensor.
          auto result = new_grad + variable_grad;
          CHECK_RESULT(result, variable);
          update_grad(std::move(result));
        } else if (!at::inplaceIsVmapCompatible(variable_grad, new_grad)) {
          // Ideally we'd perform an in-place operation to avoid changing
          // the grad tensor. However, if that's impossible because the grads
          // are vmap-incompatible (See NOTE: [vmap-incompatible in-place
          // operations]), then we just add them out-of-place.
          auto result = variable_grad + new_grad;
          CHECK_RESULT(result, variable);
          update_grad(std::move(result));
        } else {
          // In this case we can avoid changing the grad tensor. There are three
          // scenarios when we'll hit this case:
          //
          // 1. `variable_grad` is sparse, and `new_grad` is sparse.
          // 2. `variable_grad` is dense, and `new_grad` is sparse.
          // 3. `variable_grad` is dense, and `new_grad` is dense.
          // 4. `variable_grad` is mkldnn, and `new_grad` is mkldnn.
          //
          // In all of these four cases, `variable_grad += new_grad` is a
          // valid operation which adds `new_grad` to `variable_grad` in
          // place. `variable_grad` is thus still referring to the same tensor
          // after the operation.
          // Also DistributedDataParallel(DDP) package relies on grad being
          // mutated in place for saving peak memory usage. DDP will still
          // work correctly if it is mutated out of place here, but DDP will
          // maintain one extra copy of grad tensors in buffer and thus
          // increase peak memory usage.
          variable_grad += new_grad;
          CHECK_RESULT(variable_grad, variable);
          // ^ We could enforce the contract more aggressively here by writing:
          // if (variable_grad.is_sparse() || new_grad.is_sparse()) {
          //   variable_grad += new_grad;
          // } else if (obeys_layout_contract(variable_grad, variable)) {
          //   variable_grad += new_grad;
          // } else {
          //   result = at::empty_strided(variable.sizes(), variable.strides(),
          //                              variable.options().memory_format(c10::nullopt));
          //   update_grad(at::native::add_out(result, variable_grad,
          //   new_grad, 1.0);
          // }
          // However, that accumulation is sometimes in place and sometimes not,
          // which may break user code.
        }
      } else {
        at::Tensor result;
        if (variable_grad.is_sparse() && !new_grad.is_sparse()) {
          // CPU backend throws an error on sparse + dense, so prefer dense +
          // sparse here.
          result = new_grad + variable_grad;
        } else {
          // Assumes operator+ result typically matches strides of first arg,
          // and hopes variable_grad was originally created obeying layout
          // contract.
          result = variable_grad + new_grad;
        }
        CHECK_RESULT(result, variable);
        update_grad(std::move(result));
        // ^ We could enforce the contract more aggressively here by saying
        // if (obeys_layout_contract(new_grad, variable)) {
        //   update_grad(new_grad + variable_grad);
        // } else {
        //   update_grad(variable_grad + new_grad);
        // }
        // such that the stashed grad is likely to have the right strides if
        // either variable_grad or new_grad already has the right strides.
        // We could enforce the contract with certainty by saying
        // auto result = variable_grad + new_grad (or vice versa), checking
        // result's layout, and copying to an obedient clone if necessary before
        // update_grad. The copy would require another gmem pass.  We can't create
        // empty result with the right layout then add_out into it with a single
        // kernel, because GradMode is enabled in this branch, and add_out isn't
        // differentiable. Maybe more trouble than it's worth.
      }
    }

    Variable variable;
  };
    ```
  </details>

  <details> 
    <summary>Code for auto AccumulateGrad::apply(variable_list&& grads) -> variable_list</summary> 

    ```cpp
    auto AccumulateGrad::apply(variable_list&& grads) -> variable_list {
    check_input_variables("AccumulateGrad", grads, 1, 0);

    if (!grads[0].defined())
      return {};
    if (variable.grad_fn())
      throw std::logic_error(
          "leaf variable has been moved into the graph interior");
    if (!variable.requires_grad())
      return {};

    // std::move(grads[0]) to avoid bumping up refcount
    at::Tensor new_grad = callHooks(variable, std::move(grads[0]));

    // Acquire lock to here protect thread safety on variable, this ensures
    // AccumulateGrad does not race to shared variable from different threads
    // when updating the gradients. We don't ensure thread safety on hooks
    // and rely on user to provide thread safe hooks
    // see Note [Thread Safety on Autograd Node]
    std::lock_guard<std::mutex> lock(mutex_);

    at::Tensor& grad = variable.mutable_grad();

    // If the function has post hooks (for example, a DDP allreduce hook),
    // call_function in Engine.cpp will temporarily bump the expected refcount
    // by one, hence the addition of !post_hooks().empty() for 'num_expected_refs'
    // in addition to the one reference that we're holding.
    // 'num_expected_refs' is used to determine whether or not we should clone
    // the grad or can steal the grad.
    accumulateGrad(
        variable,
        grad,
        new_grad,
        1 + !post_hooks().empty() /* num_expected_refs */,
        [&grad](at::Tensor&& grad_update) { grad = std::move(grad_update); });

    return variable_list();
  }
    ```
  </details>