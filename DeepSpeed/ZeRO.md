# Zero Redundancy Optimizer (ZeRO)
- [Zero Redundancy Optimizer (ZeRO)](#zero-redundancy-optimizer-zero)
  - [Official Website](#official-website)
  - [Paper](#paper)
  - [Overview](#overview)
  - [Stage 1 & 2](#stage-1--2)
    - [DeepSpeedZeroOptimizer](#deepspeedzerooptimizer)
      - [Initialization](#initialization)
      - [reduce_ready_partitions_and_remove_grads](#reduce_ready_partitions_and_remove_grads)
        - [reduce_independent_p_g_buckets_and_remove_grads](#reduce_independent_p_g_buckets_and_remove_grads)
        - [reduce_ipg_grads](#reduce_ipg_grads)
        - [buffered_reduce_fallback](#buffered_reduce_fallback)
        - [allreduce_no_retain](#allreduce_no_retain)
        - [allreduce_and_copy](#allreduce_and_copy)
        - [allreduce_bucket](#allreduce_bucket)
      - [backward](#backward)
      - [step](#step)
        - [has_overflow](#has_overflow)
        - [has_overflow_partitioned_grads_serial & has_overflow_serial](#has_overflow_partitioned_grads_serial--has_overflow_serial)
        - [_has_inf_or_nan](#_has_inf_or_nan)
      - [contiguous_gradients](#contiguous_gradients)
      - [cpu_offload](#cpu_offload)
## Official Website
https://www.deepspeed.ai/tutorials/zero/
## Paper
https://arxiv.org/abs/1910.02054v3
## Overview
TODO
## Stage 1 & 2
* Stage 1 partitions `Optimizer`'s states across ranks
* Stage 2 partitions `Optimizer`'s states and gradients across ranks
* Both stage 1 & 2 introducing no communication overhead(in terms of volume). Reducing memory consumption and computation for updating parameters on a single rank.
### DeepSpeedZeroOptimizer
`deepspeed/runtime/zero/stage_1_and_2.py`
* `DeepSpeedZeroOptimizer` is the core of ZeRO stage 1 & 2
#### Initialization
* Initialize `round_robin_gradients` for load balancing across ranks via `_round_robin_reorder`
* Pad (to 4 * world_size byte) and flatten parameters in `param_group` via `flatten_dense_tensors_aligned`
* Update `bit16_groups` by `round_robin_bit16_groups` via `_update_model_bit16_weights`
* Partition parameters to be updated via `get_data_parallel_partitions` (view `bit16_groups_flat` as `dp` tensors with size `tensor.numel() // dp`)
* Clone self partition into `single_partition_of_fp32_groups`
* Replace original `Optimizer`'s `param_group['params']` with partitioned parameters
* Initialize all `is_partition_reduced` and `is_grad_computed` to `False`
* For stage 2, register [`reduce_ready_partitions_and_remove_grads`](#reducereadypartitionsandremovegrads) to each parameters' `grad_accumulator` just like `torch.nn.DistributedDataParallel` <!-- TODO: link to ddp-->
* Initialize `LossScaler`
* Initialize original `Optimizer`'s state by a fake `step` via `initialize_optimizer_states`

<details> 
    <summary>Code</summary>  

```Python
def __init__(self,
             init_optimizer,
             timers,
             static_loss_scale=1.0,
             dynamic_loss_scale=False,
             dynamic_loss_args=None,
             verbose=True,
             contiguous_gradients=True,
             reduce_bucket_size=500000000,
             allgather_bucket_size=5000000000,
             dp_process_group=None,
             expert_parallel_group=None,
             expert_data_parallel_group=None,
             reduce_scatter=True,
             overlap_comm=False,
             cpu_offload=False,
             mpu=None,
             clip_grad=0.0,
             communication_data_type=torch.float16,
             postscale_gradients=True,
             gradient_predivide_factor=1.0,
             gradient_accumulation_steps=1,
             ignore_unused_parameters=True,
             partition_grads=True,
             round_robin_gradients=False,
             has_moe_layers=False,
             fp16_master_weights_and_gradients=False,
             elastic_checkpoint=False):

    if dist.get_rank() == 0:
        logger.info(f"Reduce bucket size {reduce_bucket_size}")
        logger.info(f"Allgather bucket size {allgather_bucket_size}")
        logger.info(f"CPU Offload: {cpu_offload}")
        logger.info(f'Round robin gradient partitioning: {round_robin_gradients}')
    # The fused optimizer does all the work. We need this layer for two reason:
    # 1. maintain same user API from apex.fp16_utils
    # 2. keep common stuff here in case we need to add ne552w fused optimizer later

    self.elastic_checkpoint = elastic_checkpoint

    # differences from apex.fp16_utils:
    # - assume all model params in fp16
    # - assume all params requires grad
    # - flat by groups, not keeping state. TODO: remove state explicitly?
    # - master grad and unflat master weight never exist. TODO: a way to save out unflat master?
    if not torch.cuda.is_available:
        raise SystemError("Cannot use fp16 without CUDA.")
    self.optimizer = init_optimizer

    # Load pre-built or JIT compile (un)flatten ops
    util_ops = UtilsBuilder().load()
    self.flatten = util_ops.flatten
    self.unflatten = util_ops.unflatten

    # ZeRO stage 1 (False) or 2 (True)
    self.partition_gradients = partition_grads

    self.timers = timers

    self.reduce_scatter = reduce_scatter

    self.overlap_comm = overlap_comm

    self.cpu_offload = cpu_offload

    self.deepspeed_adam_offload = cpu_offload

    self.device = torch.cuda.current_device() if not self.cpu_offload else 'cpu'

    self.dp_process_group = dp_process_group

    #expert parallel group
    self.ep_process_group = expert_parallel_group

    #data parallel group for experts
    self.expert_dp_process_group = expert_data_parallel_group

    #data parallel size for non-experts
    dp_size = dist.get_world_size(group=self.dp_process_group)

    #For MoE models this maybe different for different param group
    #It will be modified during MoE setup later in the init
    self.real_dp_process_group = [
        dp_process_group for i in range(len(self.optimizer.param_groups))
    ]
    self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]

    self.is_gradient_accumulation_boundary = True

    # CPU-Offload requires contiguous gradients
    self.contiguous_gradients = contiguous_gradients or cpu_offload

    self.has_moe_layers = has_moe_layers
    if self.has_moe_layers:
        self._configure_moe_settings()
    self._global_grad_norm = 0.

    if mpu is None:
        self.model_parallel_group = None
        self.model_parallel_rank = 0
    else:
        self.model_parallel_group = mpu.get_model_parallel_group()
        self.model_parallel_rank = bwc_tensor_model_parallel_rank(mpu)

    self.overflow = False
    self.clip_grad = clip_grad
    self.communication_data_type = communication_data_type
    self.gradient_predivide_factor = gradient_predivide_factor
    self.postscale_gradients = postscale_gradients
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.micro_step_id = 0
    self.ignore_unused_parameters = ignore_unused_parameters
    self.round_robin_gradients = round_robin_gradients

    self.extra_large_param_to_reduce = None
    self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients

    if self.fp16_master_weights_and_gradients:
        assert self.cpu_offload and type(self.optimizer) in [DeepSpeedCPUAdam], f"fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32. Currently only supported using ZeRO-Offload with DeepSpeedCPUAdam. But current setting is ZeRO-Offload:{self.cpu_offload} and optimizer type {type(self.optimizer)}. Either disable fp16_master_weights_and_gradients or enable ZeRO-2 Offload with DeepSpeedCPUAdam"

    if self.reduce_scatter:
        assert self.communication_data_type in (torch.float16, torch.bfloat16), f"ZeRO-2 supports only float16 or bfloat16 communication_data_type with reduce scatter enabled. Got: '{self.communication_data_type}'"
        assert self.gradient_predivide_factor == 1.0, "gradient_predivide_factor != 1.0 is not yet supported with ZeRO-2 with reduce scatter enabled"
        assert self.postscale_gradients, "pre-scale gradients is not yet supported with ZeRO-2 with reduce scatter enabled"

    # param flattened by groups
    self.bit16_groups = []
    self.bit16_groups_flat = []

    # param partitioned by data parallel degree
    # this will contain a list of equal sized tensors
    # each of which will be updated by a different process
    self.parallel_partitioned_bit16_groups = []

    # a single 32-bit partition of the parallel partitioned parameters
    # that this process will update
    self.single_partition_of_fp32_groups = []

    # param partition info

    # These are the parameters in each group that will not be updated by this process directly
    self.params_not_in_partition = []

    # These are the parameters that will be updated by this process directly
    self.params_in_partition = []

    # Offset from the first parameter in the the self.params_in_partition
    # the parameter boundaries may not align with partition boundaries
    # so we need to keep track of the offset
    self.first_offset = []

    # number of elements per partition in each group
    self.partition_size = []

    #align nccl all-gather send buffers to 4-bye boundary
    self.nccl_start_alignment_factor = 2  # 4-byte alignment/sizeof(fp16) = 2

    assert (allgather_bucket_size % self.nccl_start_alignment_factor == 0), f"allgather_bucket_size must be a multiple of nccl_start_alignment_factor, {self.nccl_start_alignment_factor} "

    self.all_reduce_print = False
    self.dtype = self.optimizer.param_groups[0]['params'][0].dtype

    self.round_robin_bit16_groups = []
    # NOTE: ith tensor's index(before reorder) is round_robin_bit16_indices[i](after reorder)
    self.round_robin_bit16_indices = []

    # Use different parallel to do all_to_all_reduce related things
    # padding on each partition for alignment purposes
    self.groups_padding = []
    # loop to deal with groups
    for i, param_group in enumerate(self.optimizer.param_groups):
        partition_id = dist.get_rank(group=self.real_dp_process_group[i])

        # push this group to list before modify
        # TODO: Explore simplification that avoids the extra book-keeping by pushing the reordered group
        trainable_parameters = [
            param for param in param_group['params'] if param.requires_grad
        ]
        self.bit16_groups.append(trainable_parameters)

        # not sure why apex was cloning the weights before flattening
        # removing cloning here

        see_memory_usage(f"Before moving param group {i} to CPU")
        # move all the parameters to cpu to free up GPU space for creating flat buffer
        move_to_cpu(self.bit16_groups[i])
        see_memory_usage(f"After moving param group {i} to CPU", force=False)

        # Reorder group parameters for load balancing of gradient partitioning during backward among ranks.
        # This ensures that gradients are reduced in a fashion such that ownership round robins among the ranks.
        # For example, rather than 3 gradients (g_n+2, g_n+1, g_n) that are reduced consecutively belonging
        # to the same rank, instead they will belong to 3 ranks (r_m+2, r_m+1, r_m).
        if self.round_robin_gradients:
            round_robin_tensors, round_robin_indices = self._round_robin_reorder(
                self.bit16_groups[i],
                dist.get_world_size(group=self.real_dp_process_group[i])
            )
        else:
            round_robin_tensors = self.bit16_groups[i]
            round_robin_indices = list(range(len(self.bit16_groups[i])))

        self.round_robin_bit16_groups.append(round_robin_tensors)
        self.round_robin_bit16_indices.append(round_robin_indices)

        # create flat buffer in CPU and move to GPU
        self.bit16_groups_flat.append(
            self.flatten_dense_tensors_aligned(
                self.round_robin_bit16_groups[i],
                self.nccl_start_alignment_factor *
                dist.get_world_size(group=self.real_dp_process_group[i])).cuda(
                    torch.cuda.current_device()))
        see_memory_usage(f"After flattening and moving param group {i} to GPU",
                            force=False)

        # Record padding required for alignment
        if partition_id == dist.get_world_size(
                group=self.real_dp_process_group[i]) - 1:
            padding = self.bit16_groups_flat[i].numel() - sum(
                [t.numel() for t in self.round_robin_bit16_groups[i]])
        else:
            padding = 0
        self.groups_padding.append(padding)

        if dist.get_rank(group=self.real_dp_process_group[i]) == 0:
            see_memory_usage(
                f"After Flattening and after emptying param group {i} cache",
                force=False)

        # set model bit16 weight to slices of flattened buffer
        self._update_model_bit16_weights(i)

        # divide the flat weights into near equal partition equal to the data parallel degree
        # each process will compute on a different part of the partition
        data_parallel_partitions = self.get_data_parallel_partitions(
            self.bit16_groups_flat[i],
            i)
        self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)

        # verify that data partition start locations are 4-byte aligned
        for partitioned_data in data_parallel_partitions:
            assert (partitioned_data.data_ptr() %
                    (2 * self.nccl_start_alignment_factor) == 0)

        # verify that data partition start locations are 4-byte aligned
        for partitioned_data in data_parallel_partitions:
            assert (partitioned_data.data_ptr() %
                    (2 * self.nccl_start_alignment_factor) == 0)

        # A partition of the fp32 master weights that will be updated by this process.
        # Note that the params in single_partition_of_fp32_groups is cloned and detached
        # from the origin params of the model.
        if not fp16_master_weights_and_gradients:
            self.single_partition_of_fp32_groups.append(
                self.parallel_partitioned_bit16_groups[i][partition_id].to(
                    self.device).clone().float().detach())
        else:
            self.single_partition_of_fp32_groups.append(
                self.parallel_partitioned_bit16_groups[i][partition_id].to(
                    self.device).clone().half().detach())

        # Set local optimizer to have flat params of its own partition.
        # After this, the local optimizer will only contain its own partition of params.
        # In that case, the local optimizer only saves the states(momentum, variance, etc.) related to its partition's params(zero stage1).
        self.single_partition_of_fp32_groups[
            i].requires_grad = True  # keep this in case internal optimizer uses it
        param_group['params'] = [self.single_partition_of_fp32_groups[i]]

        partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(
            group=self.real_dp_process_group[i])
        params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(
            self.round_robin_bit16_groups[i],
            partition_size,
            partition_id)

        self.partition_size.append(partition_size)
        self.params_in_partition.append(params_in_partition)
        self.params_not_in_partition.append(params_not_in_partition)
        self.first_offset.append(first_offset)

    for rank in range(dist.get_world_size()):
        if dist.get_rank() == rank:
            print(
                f"Rank: {rank} partition count {self.partition_count} and sizes{[(p.numel(), self.is_moe_param_group[i] if hasattr(self, 'is_moe_param_group') else False) for i,p in enumerate(self.single_partition_of_fp32_groups)]} "
            )
            dist.barrier()
    #exit(0)
    self.reduce_bucket_size = int(reduce_bucket_size)
    self.allgather_bucket_size = int(allgather_bucket_size)

    self.reduction_event = torch.cuda.Event(enable_timing=False, blocking=False)
    self.reduction_stream = torch.cuda.Stream()
    self.cpu_computation_stream = torch.cuda.Stream()
    self.copy_grad_stream = torch.cuda.Stream()
    self.callback_queued = False

    self.param_dict = {}

    # map between param_id and bool to specify if a param is in this partition
    self.is_param_in_current_partition = {}

    self.grads_in_ipg_bucket = []
    self.params_in_ipg_bucket = []
    self.elements_in_ipg_bucket = 0
    self.params_already_reduced = []
    self._release_ipg_buffers()
    self.previous_reduced_grads = None
    self.ipg_bucket_has_moe_params = False

    # simplified param id
    self.param_id = {}

    #interesting code: unique ids being assigned to individual parameters
    largest_param_numel = 0
    count = 0
    for i, params_group in enumerate(self.bit16_groups):
        for param in params_group:
            unique_id = id(param)
            self.param_id[unique_id] = count
            self.param_dict[count] = param
            self.params_already_reduced.append(False)
            if param.numel() > largest_param_numel:
                largest_param_numel = param.numel()
            count = count + 1

    for param_group in self.params_in_partition:
        for param in param_group:
            self.is_param_in_current_partition[self.get_param_id(param)] = True

    for param_group in self.params_not_in_partition:
        for param in param_group:
            self.is_param_in_current_partition[self.get_param_id(param)] = False

    if self.cpu_offload:
        self.accumulated_grads_in_cpu = {}
        self.norm_for_param_grads = {}
        self.local_overflow = False
        self.grad_position = {}
        self.temp_grad_buffer_for_cpu_offload = torch.zeros(
            largest_param_numel,
            device=self.device,
            dtype=self.dtype).pin_memory()
        self.temp_grad_buffer_for_gpu_offload = torch.zeros(
            largest_param_numel,
            device=torch.cuda.current_device(),
            dtype=self.dtype)
        for i, params_group in enumerate(self.bit16_groups):
            self.get_grad_position(i,
                                    self.params_in_partition[i],
                                    self.first_offset[i],
                                    self.partition_size[i])

    # mapping from parameter to partition that it belongs to
    self.param_to_partition_ids = {}

    # stores if a partition has been reduced in this step
    self.is_partition_reduced = {}

    # number of grads in partition that still need to be computed
    self.remaining_grads_in_partition = {}

    # total number of grads in partition
    self.total_grads_in_partition = {}

    # stores if a grad in a partition has been computed or not
    self.is_grad_computed = {}

    # stores the offset at which a parameter gradient needs to be inserted in a partition
    self.grad_partition_insertion_offset = {}

    # the offset in the gradient at which it must be inserted at the beginning of the partition
    self.grad_start_offset = {}

    # will store the averaged gradients required by this partition
    self.averaged_gradients = {}

    # store index of first parameter in each partition
    self.first_param_index_in_partition = {}

    # initializes all data structures for implementing gradient partitioning
    self.initialize_gradient_partitioning_data_structures()

    # resets the data structure value for the next backward propagation
    self.reset_partition_gradient_structures()

    # creates backward hooks for gradient partitioning
    if self.partition_gradients or self.overlap_comm:
        self.create_reduce_and_remove_grad_hooks()

    self.custom_loss_scaler = False
    self.external_loss_scale = None

    # we may have a way of fusing dynamic scale. Do not support for now
    if self.dtype == torch.float or self.dtype == torch.bfloat16 or not dynamic_loss_scale:
        loss_scale_value = 1.0 if (
            (self.dtype == torch.float) or
            (self.dtype == torch.bfloat16)) else static_loss_scale

        self.dynamic_loss_scale = False
        self.loss_scaler = LossScaler(scale=loss_scale_value)
        cur_iter = 0
    else:
        if dynamic_loss_args is None:
            self.loss_scaler = DynamicLossScaler()
        else:
            self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)

        self.dynamic_loss_scale = True

    see_memory_usage("Before initializing optimizer states", force=True)
    self.initialize_optimizer_states()
    see_memory_usage("After initializing optimizer states", force=True)

    if dist.get_rank() == 0:
        logger.info(f"optimizer state initialized")

    if dist.get_rank(group=self.dp_process_group) == 0:
        see_memory_usage(f"After initializing ZeRO optimizer", force=True)
```
</details> 

#### reduce_ready_partitions_and_remove_grads
* Core of `partition_gradients`(stage 2) and `overlap_comm` 
* Be registered to each parameter's `grad_accumulator`
* Call `reduce_independent_p_g_buckets_and_remove_grads`
##### reduce_independent_p_g_buckets_and_remove_grads
* Call [`reduce_ipg_grads`](#reduceipggrads) if accumulated `bucket size + current gradient size > reduce_bucket_size` 
* Dealing with `contiguous_gradients`
* Increase `elements_in_ipg_bucket` by the numel of gradient
* Append gradient to `grads_in_ipg_bucket`, parameter to `params_in_ipg_bucket`

<details> 
    <summary>Code for reduce_independent_p_g_buckets_and_remove_grads</summary>  

```Python
def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
    if self.elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size:
        self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads",
                                        param.numel())
        self.reduce_ipg_grads()
        if self.contiguous_gradients and self.overlap_comm:
            # Swap ipg_index between 0 and 1
            self.ipg_index = 1 - self.ipg_index
        self.report_ipg_memory_usage("In ipg_remove_grads after reduce_ipg_grads",
                                        param.numel())

    param_id = self.get_param_id(param)
    assert self.params_already_reduced[param_id] == False, \
        f"The parameter {param_id} has already been reduced. \
        Gradient computed twice for this partition. \
        Multiple gradient reduction is currently not supported"

    if param.numel() > self.reduce_bucket_size:
        self.extra_large_param_to_reduce = param

    elif self.contiguous_gradients:
        # keeping the gradients contiguous to prevent memory fragmentation, and avoid flattening
        new_grad_tensor = self.ipg_buffer[self.ipg_index].narrow(
            0,
            self.elements_in_ipg_bucket,
            param.numel())
        new_grad_tensor.copy_(param.grad.view(-1))
        param.grad.data = new_grad_tensor.data.view_as(param.grad)

    self.elements_in_ipg_bucket += param.numel()

    assert param.grad is not None, f"rank {dist.get_rank()} - Invalid to reduce Param {param_id} with None gradient"

    self.grads_in_ipg_bucket.append(param.grad)
    self.params_in_ipg_bucket.append((i, param, param_id))

    #make sure the average tensor function knows how to average the gradients
    if is_moe_param(param):
        self.ipg_bucket_has_moe_params = True

    self.report_ipg_memory_usage("End ipg_remove_grads", 0)
```
</details> 

##### reduce_ipg_grads
* For `contiguous_gradients` <!--TODO: -->
* For not `contiguous_gradients`
    - Call [`buffered_reduce_fallback`](#bufferedreducefallback)
    - For `overlap_comm`
        - Set stream to `reduction_stream`
        - Mark `params_already_reduced[param_id] = True`
        - Append parameter to `previous_reduced_grads` if it is not belong to the partition of this rank (Will be discarded in [`allreduce_and_copy`](#allreduceandcopy))
    - For not `overlap_comm`
        - Mark `params_already_reduced[param_id] = True`
        - Set `parameter.grad` to `None` if it is not belong to the partition of this rank (discard gradient)
* Empty `grads_in_ipg_bucket`, `params_in_ipg_bucket` and `elements_in_ipg_bucket`
<details> 
    <summary>Code for reduce_ipg_grads</summary>  

```Python
def reduce_ipg_grads(self):
    if self.contiguous_gradients:
        if self.extra_large_param_to_reduce is not None:
            assert len(self.params_in_ipg_bucket) == 1, "more than 1 param in ipg bucket, this shouldn't happen"
            _, _, param_id = self.params_in_ipg_bucket[0]
            assert self.get_param_id(
                self.extra_large_param_to_reduce) == param_id, "param in ipg bucket does not match extra-large param"
            self.average_tensor(self.extra_large_param_to_reduce.grad.view(-1))
            self.extra_large_param_to_reduce = None
        else:
            self.average_tensor(self.ipg_buffer[self.ipg_index])
    else:
        self.buffered_reduce_fallback(
            None,
            self.grads_in_ipg_bucket,
            elements_per_buffer=self.elements_in_ipg_bucket)

    if self.overlap_comm:
        stream = self.reduction_stream
    elif self.cpu_offload:
        # TODO: copy_grad_stream is disabled because of race with reduce. This hurts perf and should be fixed.
        #            torch.cuda.synchronize()
        #            stream = self.copy_grad_stream
        stream = torch.cuda.current_stream()
    else:
        stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream):
        for _, param, param_id in self.params_in_ipg_bucket:

            assert self.params_already_reduced[param_id] == False, \
                f"The parameter {param_id} has already been reduced. \
                Gradient computed twice for this partition. \
                Multiple gradient reduction is currently not supported"

            self.params_already_reduced[param_id] = True

            if self.partition_gradients:
                if not self.is_param_in_current_partition[param_id]:
                    if self.overlap_comm and self.contiguous_gradients is False:
                        # Clear grads of other partitions during the next reduction
                        # to avoid clearing them before the reduction is complete.
                        if self.previous_reduced_grads is None:
                            self.previous_reduced_grads = []
                        self.previous_reduced_grads.append(param)
                    else:
                        param.grad = None  #only if self.partition_gradients
                elif self.contiguous_gradients:
                    self.copy_grads_in_partition(param)
            else:  # zero stage 1 - partition only optimizer state
                if self.contiguous_gradients and self.is_param_in_current_partition[
                        param_id]:
                    self.copy_grads_in_partition(param)

    self.grads_in_ipg_bucket = []
    self.params_in_ipg_bucket = []
    self.ipg_bucket_has_moe_params = False
    self.elements_in_ipg_bucket = 0
```
</details>

##### buffered_reduce_fallback
* Split grads in bucket according to its dtype
* Call [`allreduce_no_retain`](#allreducenoretain) on each dtype
<details> 
    <summary>Code for buffered_reduce_fallback</summary>  

```Python
def buffered_reduce_fallback(self,
                             rank,
                             grads,
                             elements_per_buffer=500000000,
                             log=None):
    split_buckets = split_half_float_double(grads)

    for i, bucket in enumerate(split_buckets):
        self.allreduce_no_retain(bucket,
                                    numel_per_bucket=elements_per_buffer,
                                    rank=rank,
                                    log=log)
```
</details>

##### allreduce_no_retain
* Split bucket into small bucket according to `numel_per_bucket`
* Call [`allreduce_and_copy`] on each small bucket

##### allreduce_and_copy
* For `overlap_comm`
    - Call `torch.cuda.synchronize()`
    - Set gradients of parameters in `previous_reduced_grads` to `None` (discard gradient)
    - Change stream to `reduction_stream`
* Call [`allreduce_bucket`] 
* Unflatten `allreduce`ed flattened tensor and copy back to origin tensor in bucket

<details> 
    <summary>Code for buffered_reduce_fallback</summary>  

```Python
# if rank is specified do a reduction instead of an allreduce
def allreduce_and_copy(self, small_bucket, rank=None, log=None):
    if self.overlap_comm:
        torch.cuda.synchronize()
        # It is safe to clear the previously reduced grads of other partitions
        self._clear_previous_reduced_grads()
        stream = self.reduction_stream
    else:
        stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream):
        allreduced = self.allreduce_bucket(small_bucket, rank=rank, log=log)
        if rank is None or rank == dist.get_rank(group=self.dp_process_group):
            for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
                buf.copy_(synced)
```
</details>

##### allreduce_bucket
* Flatten bucket
* Cast tensor to be `allreduce`ed to proper dtype
* Divide grad by `world_size`
* `reduce` or `allreduce`
* Cast back
<details> 
    <summary>Code for buffered_reduce_fallback</summary>  

```Python
def allreduce_bucket(self, bucket, rank=None, log=None):
    rank = None
    tensor = self.flatten(bucket)

    tensor_to_allreduce = tensor

    if pg_correctness_test:
        communication_data_type = torch.float32
    else:
        communication_data_type = self.communication_data_type

    if communication_data_type != tensor.dtype:
        tensor_to_allreduce = tensor.to(communication_data_type)

    tensor_to_allreduce.div_(dist.get_world_size(group=self.dp_process_group))

    if rank is None:
        #    "All Reducing"
        dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
    else:
        global_rank = dist.get_global_rank(self.dp_process_group, rank)
        dist.reduce(tensor_to_allreduce, global_rank, group=self.dp_process_group)

    if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
        if rank is None or rank == dist.get_rank(group=self.dp_process_group):
            tensor.copy_(tensor_to_allreduce)

    return tensor
```
</details>

#### backward
* Would be called by `DeepSpeedEngine.backward()`
* Handle `contiguous_gradients`
* Call `loss_scaler.backward()`

#### step
* Call [`has_overflow`](#hasoverflow)
* Update scale according to `overflow`
* If `overflow`, skipping update parameters
* Normalize gradients
* For `cpu_offload` <!--TODO: -->
* For not `cpu_offload`
    - Free parameters not belongs to current rank
    - Copy desired gradients to `single_grad_partition`
    - Unscale `single_grad_partition`
    - Call origin `Optimizer`'s step
* `all_gather` partitioned parameters
* Update model's parameters by copy `all_gather`ed tensor

<details> 
    <summary>Code for buffered_reduce_fallback</summary>  

```Python
def step(self, closure=None):
    """
    Not supporting closure.
    """
    self.micro_step_id = -1

    see_memory_usage(f"In step before checking overflow")

    # First compute norm for all group so we know if there is overflow
    self.check_overflow()
    OPTIMIZER_ALLGATHER = 'optimizer_allgather'
    OPTIMIZER_GRADIENTS = 'optimizer_gradients'
    OPTIMIZER_STEP = 'optimizer_step'
    timer_names = [OPTIMIZER_ALLGATHER, OPTIMIZER_GRADIENTS, OPTIMIZER_STEP]

    prev_scale = self.loss_scale
    self._update_scale(self.overflow)
    if self.overflow:
        if dist.get_rank() == 0:
            logger.info(
                "[deepspeed] OVERFLOW! Rank {} Skipping step. Attempted loss scale: {}, "
                "reducing to {}".format(dist.get_rank(),
                                        prev_scale,
                                        self.loss_scale))

        see_memory_usage('After overflow before clearing gradients')
        self.zero_grad()
        if self.cpu_offload:
            self.reset_cpu_buffers()
        else:
            self.averaged_gradients = {}

        see_memory_usage('After overflow after clearing gradients')

        self.start_timers(timer_names)
        self.stop_timers(timer_names)
        return

    # Step 1:- Calculate gradient norm using fp-16 grads
    see_memory_usage('Before norm calculation')
    scaled_global_grad_norm = self.scaled_global_norm()
    self._global_grad_norm = scaled_global_grad_norm / self.loss_scale

    see_memory_usage('After norm before optimizer')
    # Step 2:- run optimizer and upscaling simultaneously
    for i, group in enumerate(self.bit16_groups):
        self.start_timers([OPTIMIZER_GRADIENTS])
        partition_id = dist.get_rank(group=self.real_dp_process_group[i])
        if self.cpu_offload:
            single_grad_partition = self.single_partition_of_fp32_groups[i].grad
            self.unscale_and_clip_grads([single_grad_partition],
                                        scaled_global_grad_norm)
            self.stop_timers([OPTIMIZER_GRADIENTS])
            self.start_timers([OPTIMIZER_STEP])
            self._optimizer_step(i)

            from deepspeed.ops.adam import DeepSpeedCPUAdam
            if not (type(self.optimizer) == DeepSpeedCPUAdam
                    and self.dtype == torch.half):
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)

            self.stop_timers([OPTIMIZER_STEP])
        else:
            # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
            self.free_grad_in_param_list(self.params_not_in_partition[i])

            # create a flat gradients for parameters updated by this process
            # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
            if partition_id == dist.get_world_size(
                    group=self.real_dp_process_group[i]) - 1:
                single_grad_partition = self.flatten_dense_tensors_aligned(
                    self.averaged_gradients[i],
                    int(self.partition_size[i])).to(
                        self.single_partition_of_fp32_groups[i].dtype)
            else:
                single_grad_partition = self.flatten(self.averaged_gradients[i]).to(
                    self.single_partition_of_fp32_groups[i].dtype)
            assert single_grad_partition.numel() == self.partition_size[i], \
                "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                    single_grad_partition.numel(), self.partition_size[i], i, partition_id)

            self.single_partition_of_fp32_groups[i].grad = single_grad_partition
            # release all the gradient since we have already created a necessary copy in dp_grad_partition(ZeRO stage2)
            self.free_grad_in_param_list(self.params_in_partition[i])

            self.averaged_gradients[i] = None

            self.unscale_and_clip_grads([single_grad_partition],
                                        scaled_global_grad_norm)
            self.stop_timers([OPTIMIZER_GRADIENTS])

            # Step 3:- run the optimizer if no offloading
            self.start_timers([OPTIMIZER_STEP])
            self._optimizer_step(i)
            # Step 4:- get rid of the fp32 gradients. Not needed anymore
            self.single_partition_of_fp32_groups[i].grad = None
            del single_grad_partition
            bit16_partitions = self.parallel_partitioned_bit16_groups[i]
            fp32_partition = self.single_partition_of_fp32_groups[i]
            bit16_partitions[partition_id].data.copy_(fp32_partition.data)
            self.stop_timers([OPTIMIZER_STEP])

    see_memory_usage('After optimizer before all-gather')
    if self.cpu_offload:
        self.reset_cpu_buffers()

    self.start_timers([OPTIMIZER_ALLGATHER])
    # Gather the updated weights from everyone.
    # Then all partitions of the model parameters are updated and ready for next round forward.
    all_gather_dp_groups(
        partitioned_param_groups=self.parallel_partitioned_bit16_groups,
        dp_process_group=self.real_dp_process_group,
        start_alignment_factor=self.nccl_start_alignment_factor,
        allgather_bucket_size=self.allgather_bucket_size)

    self.stop_timers([OPTIMIZER_ALLGATHER])

    # TODO: we probably don't need this? just to be safe
    for i in range(len(self.bit16_groups)):
        self._update_model_bit16_weights(i)

    self.log_timers(timer_names)
    see_memory_usage('After zero_optimizer step')

    return
```
</details>

##### has_overflow
* For `partition_gradients` call [`has_overflow_partitioned_grads_serial`](#hasoverflowpartitionedgradsserial--hasoverflowserial)
* For not `partition_gradients` call [`has_overflow_serial`](#hasoverflowpartitionedgradsserial--hasoverflowserial)
* `allreduce` result

<details> 
    <summary>Code for buffered_reduce_fallback</summary>  

```Python
def has_overflow(self, partition_gradients=True):
    if partition_gradients:
        overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial(
        )
        overflow_gpu = torch.cuda.ByteTensor([overflow])
        '''This will capture overflow across all data parallel and expert parallel process
        Since expert parallel process are a subset of data parallel process'''
        dist.all_reduce(overflow_gpu,
                        op=dist.ReduceOp.MAX,
                        group=self.dp_process_group)

    else:
        params = []
        for group in self.bit16_groups:
            for param in group:
                params.append(param)

        overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
        overflow_gpu = torch.cuda.ByteTensor([overflow])

    # Since each model parallel GPU carries only part of the model,
    # make sure overflow flag is synced across all the model parallel GPUs
    self._model_parallel_all_reduce(tensor=overflow_gpu, op=dist.ReduceOp.MAX)

    overflow = overflow_gpu[0].item()
    return bool(overflow)
```
</details>

##### has_overflow_partitioned_grads_serial & has_overflow_serial
* Call [`_has_inf_or_nan`](#hasinfornan)

##### _has_inf_or_nan
* Sum
```Python
a = torch.Tensor([1, float('nan'), 1]) 
a.sum()  # tensor(nan)

a = torch.Tensor([1, float('inf'), 1]) 
a.sum()  # tensor(inf)

a = torch.Tensor([1, float('nan'), float('inf')])
a.sum()  # tensor(nan)

a = float('nan')
a != a  # True!!!!!!
```

<details> 
    <summary>Code for buffered_reduce_fallback</summary>  

```Python
# `x` is a torch.Tensor
@staticmethod
def _has_inf_or_nan(x, j=None):
    try:
        # if x is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as x
        # (which is true for some recent version of pytorch).
        cpu_sum = float(x.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # cpu_sum = float(x.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True
        return False
```
</details>

#### contiguous_gradients <!--TODO: -->

#### cpu_offload <!--TODO: -->