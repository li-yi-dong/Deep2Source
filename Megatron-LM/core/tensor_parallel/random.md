<div align='center'><font size='20'> random </font></div>

- rng and CheckpointFunction

# CudaRNGStatesTracker

# CheckpointFunction
## forward
- Save cpu and gpu rng
- Run forward function with `torch.no_grad`
- If `distribute_saved_activations`
  - Split args[0].data across tensor parallel group
- Save all args for backward

## backward
- If `distribute_saved_activations`
  - Gather args[0].data across tensor parallel group
- Resume forward rng
- Detach all saved_tensor
- Run forward function with `torch.enable_grad`
- Get backward rng
- Run backward on output with `torch.autograd.backward`

  <details> 
      <summary>Code for CheckpointFunction</summary>  

  ```Python
  class CheckpointFunction(torch.autograd.Function):
      """This function is adapted from torch.utils.checkpoint with
         two main changes:
             1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
             2) the states in the model parallel tracker are also properly
                tracked/set/reset.
      """
      @staticmethod
      def forward(ctx, run_function, distribute_saved_activations, *args):
          ctx.run_function = run_function
          ctx.distribute_saved_activations \
              = distribute_saved_activations

          # Copy the rng states.
          ctx.fwd_cpu_rng_state = torch.get_rng_state()
          ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
          ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

          with torch.no_grad():
              outputs = run_function(*args)

          # Divide hidden states across model parallel group and only keep
          # the chunk corresponding to the current rank.
          if distribute_saved_activations:
              ctx.input_0_shape = args[0].data.shape
              safely_set_viewless_tensor_data(
                  args[0],
                  split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True))

          # Store everything.
          ctx.save_for_backward(*args)

          return outputs

      @staticmethod
      def backward(ctx, *args):
          if not torch.autograd._is_checkpoint_valid():
              raise RuntimeError("Checkpointing is not compatible with .grad(), "
                                 "please use .backward() if possible")
          inputs = ctx.saved_tensors
          if ctx.distribute_saved_activations:
              safely_set_viewless_tensor_data(
                  inputs[0],
                  gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape))

          # Store the current states.
          bwd_cpu_rng_state = torch.get_rng_state()
          bwd_cuda_rng_state = torch.cuda.get_rng_state()
          bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

          # Set the states to what it used to be before the forward pass.
          torch.set_rng_state(ctx.fwd_cpu_rng_state)
          _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
          get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

          # Compute the forward pass.
          detached_inputs = detach_variable(inputs)
          with torch.enable_grad():
              outputs = ctx.run_function(*detached_inputs)

          # Set the states back to what it was at the start of this function.
          torch.set_rng_state(bwd_cpu_rng_state)
          _set_cuda_rng_state(bwd_cuda_rng_state)
          get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

          if isinstance(outputs, torch.Tensor):
              outputs = (outputs,)
          torch.autograd.backward(outputs, args)
          grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                        for inp in detached_inputs)
          return (None, None) + grads
  ```
  </details>