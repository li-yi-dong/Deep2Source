# mpu (model parallel utility)
* Utility for managing world info and Tensor/Pipeline parallel

## Initialization
`megatron/mpu/initialize.py`

### initialize_model_parallel
Prepare `process_group`s for collective communication
#### _DATA_PARALLEL_GROUP
* `data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size)`
* Ranks are grouped with stride of `tensor_model_parallel_size` 
* Initialize a `process_group` for each `data group `
* Set global variable `_DATA_PARALLEL_GROUP` to the `process_group` contain current rank
  <details> 
      <summary>Code for initialize _DATA_PARALLEL_GROUP</summary>  

  ```Python
  all_data_parallel_group_ranks = []
  for i in range(pipeline_model_parallel_size):
      start_rank = i * num_pipeline_model_parallel_groups
      end_rank = (i + 1) * num_pipeline_model_parallel_groups
      for j in range(tensor_model_parallel_size):
          ranks = range(start_rank + j, end_rank,
                        tensor_model_parallel_size)
          all_data_parallel_group_ranks.append(list(ranks))
          group = torch.distributed.new_group(ranks)
          if rank in ranks:
              _DATA_PARALLEL_GROUP = group
  ```
  </details>

#### _MODEL_PARALLEL_GROUP
* i<sub>th</sub> ranks in each `data_parallel_group` are grouped
* Initialize a `process_group` for each `model group `
* Set global variable `_MODEL_PARALLEL_GROUP` to the `process_group` contain current rank
  <details> 
      <summary>Code for initialize _MODEL_PARALLEL_GROUP</summary>  

  ```Python
  for i in range(data_parallel_size):
      ranks = [data_parallel_group_ranks[i]
               for data_parallel_group_ranks in all_data_parallel_group_ranks]
      group = torch.distributed.new_group(ranks)
      if rank in ranks:
          _MODEL_PARALLEL_GROUP = group
  ```
  </details>

#### _TENSOR_MODEL_PARALLEL_GROUP
* Group ranks in sequence with size of `num_tensor_model_parallel_groups`
* Set global variable `_TENSOR_MODEL_PARALLEL_GROUP` to the `process_group` contain current rank
  <details> 
      <summary>Code for initialize _TENSOR_MODEL_PARALLEL_GROUP</summary>  

  ```Python
  for i in range(num_tensor_model_parallel_groups):
      ranks = range(i * tensor_model_parallel_size,
                    (i + 1) * tensor_model_parallel_size)
      group = torch.distributed.new_group(ranks)
      if rank in ranks:
          _TENSOR_MODEL_PARALLEL_GROUP = group
  ```
  </details>

#### _PIPELINE_MODEL_PARALLEL_GROUP
* Ranks are grouped with stride of `num_pipeline_model_parallel_groups`
* Set global variable `_PIPELINE_MODEL_PARALLEL_GROUP` to the `process_group` contain current rank
* Set global variable `_PIPELINE_GLOBAL_RANKS` to the ranks in the `process_group` contain current rank
  <details> 
      <summary>Code for initialize _PIPELINE_MODEL_PARALLEL_GROUP</summary>  

  ```Python
  for i in range(num_pipeline_model_parallel_groups):
    ranks = range(i, world_size,
                  num_pipeline_model_parallel_groups)
    group = torch.distributed.new_group(ranks)
    if rank in ranks:
        _PIPELINE_MODEL_PARALLEL_GROUP = group
        _PIPELINE_GLOBAL_RANKS = ranks
  ```
  </details>

#### _EMBEDDING_GROUP
* The first and last ranks in `_PIPELINE_MODEL_PARALLEL_GROUP` together with ranks in `pipeline_model_parallel_split_rank_` 

#### _POSITION_EMBEDDING_GROUP
* The first rank in `_PIPELINE_MODEL_PARALLEL_GROUP` together with ranks in `pipeline_model_parallel_split_rank_` 

<details> 
    <summary>Code for initialize_model_parallel</summary>  

```Python
def initialize_model_parallel(tensor_model_parallel_size_=1,
                              pipeline_model_parallel_size_=1,
                              virtual_pipeline_model_parallel_size_=None,
                              pipeline_model_parallel_split_rank_=None):
    """
    Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model parallelism.
        virtual_pipeline_model_parallel_size: number of virtual stages (interleaved
                                              pipeline).
        pipeline_model_parallel_split_rank: for models with both encoder and decoder,
                                            rank in pipeline with split point.


    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if torch.distributed.get_rank() == 0:
        print('> initializing tensor model parallel with size {}'.format(
            tensor_model_parallel_size_))
        print('> initializing pipeline model parallel with size {}'.format(
            pipeline_model_parallel_size_))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = min(tensor_model_parallel_size_, world_size)
    pipeline_model_parallel_size = min(pipeline_model_parallel_size_, world_size)
    ensure_divisibility(world_size,
                        tensor_model_parallel_size * pipeline_model_parallel_size)
    data_parallel_size = world_size // (tensor_model_parallel_size *
                                        pipeline_model_parallel_size)

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    num_data_parallel_groups = world_size // data_parallel_size

    if virtual_pipeline_model_parallel_size_ is not None:
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size_

    if pipeline_model_parallel_split_rank_ is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank_

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank,
                          tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i]
                 for data_parallel_group_ranks in all_data_parallel_group_ranks]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, \
        'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, \
        'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, \
        'position embedding group is already initialized'
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size,
                      num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank_ is not None:
                if ranks[pipeline_model_parallel_split_rank_] not in embedding_ranks:
                    embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank_],
                                       ranks[-1]]
                if ranks[pipeline_model_parallel_split_rank_] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank_]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(position_embedding_ranks)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks
```
</details>

# get_num_layers
Compute the number of transformer layers resident on the current rank.
* For not using pipeline
  - Number of layers equals to `args.num_layers`
* For using pipeline
  - For not `is_encoder_and_decoder_model`
    - 0 if `standalone_embedding_stage` and `rank==0`
    - Else `args.num_layers // args.transformer_pipeline_model_parallel_size`
  - For `is_encoder_and_decoder_model`
    - `args.pipeline_model_parallel_split_rank` designate which rank has encoder and which rank has decoder
    - If `is_pipeline_stage_before_split()==True`, this rank has encoder
    - Compute `num_ranks_in_encoder` and `num_ranks_in_decoder`
    - Compute `num_layers` according to whether current should have encoder or decoder

<details> 
    <summary>Code for initialize _MODEL_PARALLEL_GROUP</summary>  

```Python
def get_num_layers(args, is_encoder_and_decoder_model):
    """Compute the number of transformer layers resident on the current rank."""
if get_pipeline_model_parallel_world_size() > 1:
    if is_encoder_and_decoder_model:
        assert args.pipeline_model_parallel_split_rank is not None

        # When a standalone embedding stage is used, a rank is taken from
        # the encoder's ranks, to be used for the encoder's embedding
        # layer. This way, the rank referenced by the 'split rank' remains
        # the same whether or not a standalone embedding stage is used.
        num_ranks_in_encoder = (
            args.pipeline_model_parallel_split_rank - 1
            if args.standalone_embedding_stage else
            args.pipeline_model_parallel_split_rank
        )
        num_ranks_in_decoder = args.transformer_pipeline_model_parallel_size - num_ranks_in_encoder
        assert args.num_layers % num_ranks_in_encoder == 0, \
                'num_layers (%d) must be divisible by number of ranks given to encoder (%d)' % (args.num_layers, num_ranks_in_encoder)
        assert args.num_layers % num_ranks_in_decoder == 0, \
                'num_layers (%d) must be divisible by number of ranks given to decoder (%d)' % (args.num_layers, num_ranks_in_decoder)
        if is_pipeline_stage_before_split():
            num_layers = (
                0
                if args.standalone_embedding_stage
                and get_pipeline_model_parallel_rank() == 0 else
                args.num_layers // num_ranks_in_encoder
            )
        else:
            num_layers = args.num_layers // num_ranks_in_decoder
    else:
        assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
            'num_layers must be divisible by transformer_pipeline_model_parallel_size'

        # When a standalone embedding stage is used, all transformer layers
        # are divided among pipeline rank >= 1, while on pipeline rank 0,
        # ranks either contain the input embedding layer (virtual pp rank 0),
        # or no layers at all (virtual pp rank >= 1).
        num_layers = (
            0
            if args.standalone_embedding_stage
            and get_pipeline_model_parallel_rank() == 0 else
            args.num_layers // args.transformer_pipeline_model_parallel_size
        )
else:
    num_layers = args.num_layers
return num_layers
```
</details>