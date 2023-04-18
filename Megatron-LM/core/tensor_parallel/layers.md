<div align='center'><font size='20'> layers </font></div>

- megatron/core/tensor_parallel/layers.py
- Tensor-parallel word embedding and row/column parallel linear

# VocabParallelEmbedding
- Each rank only store a partition of weights of vocab_size
- Each rank only process input within its own partition of vocab_size
- all_reduce embed output to have full embed output
## initialization
- Calculate `vocab_start_index` and `vocab_end_index` according to vocab_size and tensor parallel config
- Initialize weight for own partition (on cpu or gpu according to `use_cpu_initialization`)
    <details> 
        <summary>Code for VocabParallelEmbedding.__init__</summary>  

    ```Python
    def __init__(self, num_embeddings: int, embedding_dim: int, *,
                    init_method=init.xavier_normal_,
                    params_dtype: torch.dtype=torch.float32,
                    use_cpu_initialization: bool=False,
                    perform_initialization: bool=True):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight, self.num_embeddings, self.embedding_dim,
                    self.num_embeddings_per_partition, 0, init_method,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                                partition_dim=0, stride=1)
    ```
    </details>

## forward
- Mask out input out of range (idx beyond `vocab_start_index` and `vocab_end_index`)
- Embed with local weight
- Mask out output out of range
- all_reduce to get the whole embeddings by [`reduce_from_tensor_model_parallel_region(output_parallel)`](mappings.md#_reducefrommodelparallelregion)

# LinearWithGradAccumulationAndAsyncCommunication
## forward
- If not sequence_parallel:
  - Simply `output = torch.matmul(total_input, weight.t())`
- If sequence_parallel:
  - ...
- Add bias
## backward
- If not sequence_parallel:
  - `total_input = input`
- If sequence_parallel:
  - ...
- Calculate gradient of input by `grad_input = grad_output.matmul(weight)`
- If async_grad_allreduce:
  - Async all_reduce gradient of input
- If not async_grad_allreduce:
  - gradient would be all_reduced by caller after `LinearWithGradAccumulationAndAsyncCommunication.backward`
- View `grad_output` and  `total_input` to deal with batch
- If gradient_accumulation_fusion:
  - ...
- If not gradient_accumulation_fusion
  - Calculate `grad_weight` by `grad_weight = grad_output.t().matmul(total_input)`
- Calculate `grad_bias` by `grad_bias = grad_output.sum(dim=0) if use_bias else None`
# ColumnParallelLinear
- Partition weights along output_size
## initialization
- Initialize weight and bias (partitioned along output_size)
    <details> 
        <summary>Code for ColumnParallelLinear.__init__</summary>  

    ```Python
    def __init__(self, num_embeddings: int, embedding_dim: int, *,
                    init_method=init.xavier_normal_,
                    params_dtype: torch.dtype=torch.float32,
                    use_cpu_initialization: bool=False,
                    perform_initialization: bool=True):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight, self.num_embeddings, self.embedding_dim,
                    self.num_embeddings_per_partition, 0, init_method,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                                partition_dim=0, stride=1)
    ```
    </details>

## forward
- If not `async_tensor_model_parallel_allreduce` or not `sequence_parallel_enabled`, all_reduce gradient of input on backward
- Calculate output by [LinearWithGradAccumulationAndAsyncCommunication](#linearwithgradaccumulationandasynccommunication)
- Gather output according to gather_output
    <details> 
        <summary>Code for ColumnParallelLinear.forward</summary>  

    ```Python 
    def forward(self, input_):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or \
                self.sequence_parallel_enabled:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
        )
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
    ```
    </details>

# RowParallelLinear
- Partition weights along input_size
- Partition input along its second dim
## initialization
- Initialize weight (partitioned along input_size)
- Initialize bias
## forward
- Scatter input if not `input_is_parallel`
- Calculate partial output by [LinearWithGradAccumulationAndAsyncCommunication](#linearwithgradaccumulationandasynccommunication)
- all_reduce partial output
- Add bias
    <details> 
        <summary>Code for RowParallelLinear.forward</summary>  

    ```Python 
    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel_enabled
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
        )

        # All-reduce across all the partitions.
        if self.sequence_parallel_enabled:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
    ```
    </details>