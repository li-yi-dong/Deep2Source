<div align='center'><font size='20'> mappings </font></div>

- megatron/core/tensor_parallel/mappings.py
- Implementations collective utils for tensor parallel

# _reduce
- all_reduce with tenor_parallel_group
    <details> 
        <summary>Code for _reduce</summary>  

    ```Python
    def _reduce(input_):
        """All-reduce the input tensor across model parallel group."""

        # Bypass the function if we are using only 1 GPU.
        if get_tensor_model_parallel_world_size()==1:
            return input_

        # All-reduce.
        torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

        return input_
    ```
    </details>

# _ReduceFromModelParallelRegion
- all_reduce with tenor_parallel_group
- Pass through gradient on backward
    <details> 
        <summary>Code for _ReduceFromModelParallelRegion</summary>  

    ```Python
    class _ReduceFromModelParallelRegion(torch.autograd.Function):
        """All-reduce the input from the model parallel region."""

        @staticmethod
        def symbolic(graph, input_):
            return _reduce(input_)
        
        @staticmethod
        def forward(ctx, input_):
            return _reduce(input_)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output
    ```
    </details>