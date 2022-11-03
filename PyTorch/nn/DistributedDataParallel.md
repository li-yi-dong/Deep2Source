<div align='center'><font size='20'> Distributed Data Parallel </font></div>

- [torch.distributed](#torchdistributed)
- [torch.nn.parallel](#torchnnparallel)
- [torch.nn.parallel.DistributedDataParallel（DDP）](#torchnnparalleldistributeddataparallelddp)
  - [Initialization](#initialization)
  - [forward](#forward)
  - [no_sync](#no_sync)
  - [join](#join)
  - [join_hook](#join_hook)
- [_DDPJoinHook](#_ddpjoinhook)
- [_DDPSink](#_ddpsink)
- [Reducer](#reducer)
- [Communication hook](#communication-hook)
  - [Paper](#paper)

# torch.distributed
`torch.nn.parallel` heavily relies on `torch.distributed` for operations across nodes. See:
[Distributed](../Distributed/README.md)

# torch.nn.parallel
`torch.nn.parallel` implements`DataParallel` and`DistributedDataParallel` （`DataParallel` is obsoleted）

# torch.nn.parallel.DistributedDataParallel（DDP）
`torch/nn/parallel/distributed.py`  
`DistributedDataParallel` implements distributed data parallel with simple interfaces. 
https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html

- DDP achieves distributed training by average gradients across all nodes(equivalent to enlarge batch_size on a single node)
- DDP performs forward/backward and updating parameters(optimizer) independently across nodes, but `all_reduce` gradients once independent backward done. Thus, it requires the model (parameters and buffers) is the same at the beginning of each step.  Furthermore, the optimizer on each node should have the same result for same gradients(If there is random procedural in optimizer, the `seed` on each node should be the same).
- DDP `bucketing` parameters for overlapping the computation of backward and `all_reduce` of gradients. One `Bucket` contains a part of the whole parameters. Once all parameters in a `Bucket` obtain their gradients from backward(autograd engine), the `all_reduce` on gradients will be triggered.
- DDP bucketing the parameters in a reversed order against the order of parameters declared in the model (In most cases, the outer layer of model would get gradients earlier). Therefore, to achieve better performance, construct models accordingly when possible.
- The actual `all_reduce` of gradients is implemented in [Reducer](../Distributed/Reducer.md)

## Initialization
- Subclassing `torch.nn.Module`
- Check inputs
- Prepare parameters via `_build_params_for_reducer()` and `_verify_param_shape_across_processes()`
- Broadcast rank0’s parameters to other ranks
- Set up `reducer` via `_ddp_init_helper()`
- Set up static graph
  <details> 
    <summary>__init__</summary>  

    ```Python        
    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        static_graph=False,
    ):
    
        super(DistributedDataParallel, self).__init__()
        Joinable.__init__(self)
        self.logger = None
        if not any((p.requires_grad for p in module.parameters())):
            self._log_and_throw(
                RuntimeError,
                "DistributedDataParallel is not needed when a module "
                "doesn't have any parameter that requires a gradient.",
            )
    
        if device_ids is not None and len(device_ids) > 1:
            self._log_and_throw(
                ValueError, "device_ids can only be None or contain a single element."
            )
    
        self.is_multi_device_module = len({p.device for p in module.parameters()}) > 1
        distinct_device_types = {p.device.type for p in module.parameters()}
        if len(distinct_device_types) != 1:
            self._log_and_throw(
                ValueError,
                "DistributedDataParallel's input module must be on "
                "the same type of devices, but input module parameters locate in {}.".format(
                    distinct_device_types
                ),
            )
    
        self.device_type = list(distinct_device_types)[0]
    
        if (
            device_ids is None
            or len(device_ids) == 0  # For backward compatibility.
            or self.device_type == "cpu"
            or self.is_multi_device_module
        ):
            if device_ids or output_device:
                self._log_and_throw(
                    ValueError,
                    "DistributedDataParallel device_ids and output_device arguments "
                    "only work with single-device/multiple-device GPU modules or CPU modules, "
                    "but got device_ids {}, output_device {}, and module parameters {}.".format(
                        device_ids,
                        output_device,
                        {p.device for p in module.parameters()},
                    ),
                )
    
            self.device_ids = None
            self.output_device = None
        else:
            self.device_ids = [_get_device_index(x, True) for x in device_ids]
    
            if output_device is None:
                output_device = device_ids[0]
    
            self.output_device = _get_device_index(output_device, True)
    
        if process_group is None:
            self.process_group = _get_default_group()
        else:
            self.process_group = process_group
    
        self.static_graph = False
        self.dim = dim
        self.module = module
        self.device = list(self.module.parameters())[0].device
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.gradient_as_bucket_view = gradient_as_bucket_view
        if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = module._ddp_params_and_buffers_to_ignore
        else:
            self.parameters_to_ignore = []
    
        self._use_replicated_tensor_module = _ddp_with_replicated_tensor_enabled()
        self._build_replicated_tensor_module()
    
        if check_reduction:
            # This argument is no longer used since the reducer
            # will ensure reduction completes even if some parameters
            # do not receive gradients.
            warnings.warn(
                "The `check_reduction` argument in `DistributedDataParallel` "
                "module is deprecated. Please avoid using it."
            )
    
        # Check that a module does not have Uninitialized parameters
        for param in module.parameters():
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                self._log_and_throw(
                    RuntimeError,
                    "Modules with uninitialized parameters can't be used with `DistributedDataParallel`. "
                    "Run a dummy forward pass to correctly initialize the modules",
                )
        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = int(250 * 1024 * 1024)
    
        # reduction bucket size
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)
        # Whether to perform input tensor CPU to GPU copies on a side-stream
        self.use_side_stream_for_tensor_copies = (
            os.environ.get("PYTORCH_DDP_USE_SIDE_STREAM", "1") == "1"
        )
    
        # Build parameters for reducer.
        parameters, expect_sparse_gradient = self._build_params_for_reducer()
        # Verify model equivalence.
        _verify_param_shape_across_processes(self.process_group, parameters)
        # Sync params and buffers. Ensures all DDP models start off at the same value.
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=0,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
        )
        # In debug mode, build a mapping of parameter index -> parameter.
        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)
        # Builds reducer.
        self._ddp_init_helper(
            parameters, expect_sparse_gradient, param_to_name_mapping, static_graph
        )
        self._has_rebuilt_buckets = False
    
        if static_graph:
            self._set_static_graph()
    ```
  </details>

  <details> 
    <summary>_build_params_for_reducer</summary> 

    Generate two lists like `[(m1, p1), (m1, p2), (m2, p3), ...]` and `[False, False, True, ...]`
    ```Python
    def _build_params_for_reducer(self):
        # Build tuple of (module, parameter) for all parameters that require grads.
    		# NOTE: A list like [(m1, p1), (m1, p2), (m2, p2), (m2, p3), ...]
        modules_and_parameters = [
            (module, parameter)
            for module_name, module in self.module.named_modules()
            for parameter in [
                param
                # Note that we access module.named_parameters instead of
                # parameters(module). parameters(module) is only needed in the
                # single-process multi device case, where it accesses replicated
                # parameters through _former_parameters.
                for param_name, param in module.named_parameters(recurse=False)
                if param.requires_grad
                and f"{module_name}.{param_name}" not in self.parameters_to_ignore
            ]
        ]
    
        # Deduplicate any parameters that might be shared across child modules.
        memo = set()
    		# NOTE: Actually removed duplicated parameters
    		# NOTE: The example list ahead becomes [(m1, p1), (m1, p2), (m2, p3), ...]
        modules_and_parameters = [
            # "p not in memo" is the deduplication check.
            # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
            (m, p) for m, p in modules_and_parameters
            if p not in memo and not memo.add(p)
        ]
    
        # Build list of parameters.
        parameters = list(parameter for _, parameter in modules_and_parameters)
    
        # Checks if a module will produce a sparse gradient.
        def produces_sparse_gradient(module):
            if isinstance(module, torch.nn.Embedding) or isinstance(
                module, torch.nn.EmbeddingBag
            ):
                return module.sparse
            return False
    
        # Build list of booleans indicating whether or not to expect sparse
        # gradients for the corresponding parameters.
        expect_sparse_gradient = list(produces_sparse_gradient(module) for module, _ in modules_and_parameters)
    
        self._assign_modules_buffers()
    
        return parameters, expect_sparse_gradient
    ```
  </details>
        
  <details> 
    <summary>_ddp_init_helper</summary>

    Initialize `Reducer` and `Logger`
    ```Python
    def _ddp_init_helper(
        self, parameters, expect_sparse_gradient, param_to_name_mapping,
        static_graph
    ):
        """
        Initialization helper function that does the following:
        (1) bucketing the parameters for reductions
        (2) resetting the bucketing states
        (3) registering the grad hooks
        (4) Logging construction-time DDP logging data
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """
        self.num_iterations = 0
        # Notice, the parameters order is not in the order in which they are used,
        # especially in models with control flow.
        #
        # Alongside parameters are not presented in the real execution order,
        # if a certain model happens to also
        #   1) have other collectives comm ops in its backward graph.
        #   2) have unused parameter in subset ranks of the whole world.
        # bucketing could insert ALL-REDUCE comm op too early on the rank with unused parameter,
        # matching up with other collectives comm ops on other ranks unexpectedly.
        #
        # In order to handle this corner case, when the parameters are not in the real execution order,
        # we don't do bucketing, thus only one ALL-REDUCE is inserted after all the gradients
        # of the whole graph are computed.
        #
        # Notice, here we only disable bucketing for the first iteration.
        # After the first iteration, it's OK to rebuild buckets,
        # because "bucket rebuild" bucketizes parameters based on its real execution order in backward graph.
    
        # Can remove this branching once #73732 is landed.
        if static_graph is True or self.find_unused_parameters is False:
            bucket_size_limits = [sys.maxsize]
        else:
            bucket_size_limits = [dist._DEFAULT_FIRST_BUCKET_BYTES, self.bucket_bytes_cap]
    
    		# NOTE: link to **[compute_bucket_assignment_by_size](Distributed%20Data%20Parallel%20398e5c2564604fd6a2d67eadb811e6ca.md)**
        bucket_indices, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            parameters,
            bucket_size_limits,
            expect_sparse_gradient,
        )
    
        # Note: reverse list of buckets because we want to approximate the
        # order in which their gradients are produced, and assume they
        # are used in the forward pass in the order they are defined.
        self.reducer = dist.Reducer(
            parameters,
            list(reversed(bucket_indices)),
            list(reversed(per_bucket_size_limits)),
            self.process_group,
            expect_sparse_gradient,
            # The bucket size limit is specified in the constructor.
            # Additionally, we allow for a single small bucket for parameters
            # that are defined first, such that their gradients don't spill into
            # a much larger bucket, adding unnecessary latency after gradient
            # computation finishes. Experiments showed 1MB is a reasonable value.
            self.bucket_bytes_cap,
            self.find_unused_parameters,
            self.gradient_as_bucket_view,
            param_to_name_mapping,
            # User can set dist._DEFAULT_FIRST_BUCKET_BYTES to tune DDP first
            # bucket.
            dist._DEFAULT_FIRST_BUCKET_BYTES
        )
    
        self.logger = dist.Logger(self.reducer)
        # Set as a weak reference to avoid reference cycle between
        # logger and reducer.
        self.reducer.set_logger(self.logger)
    
        has_sync_bn = False
        for submodule in self.module.modules():
            if isinstance(submodule, torch.nn.SyncBatchNorm):
                has_sync_bn = True
                break
    
        # Set logging data that can be got during construction time.
        self.logger.set_construction_data_and_log(
            self.module.__class__.__name__,
            [] if self.device_ids is None else self.device_ids,
            -1 if self.output_device is None else self.output_device,
            self.broadcast_buffers,
            has_sync_bn,
            static_graph,
        )
    
        # passing a handle to torch.nn.SyncBatchNorm layer
        self._passing_sync_batchnorm_handle(self.module)
    ```
  </details>
        

## forward
- Call [`Join.notify_join_context`](../Distributed/Joinable.md#notifyjoincontext)
- Call [`Reducer._rebuild_buckets`](../Distributed/Reducer.md#reducerrebuildbuckets)
- Sync buffers if needed
- Call original model and get `output`
- Sync buffers if needed
- If `is_grad_enabled` and not in [gradient accumulation](#nosync), call [`Reducer.prepare_for_backward`](../Distributed/Reducer.md#reducerprepareforbackward)(enable [`autograd_hook`](../Distributed/Reducer.md#reducerautogradhook))
- Apply [_DDPSink](#ddpsink) to each `output` if needed
  <details> 
    <summary>Code for forward</summary> 
    
    ```Python
    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
            if torch.is_grad_enabled() and self.require_backward_grad_sync:
                self.logger.set_runtime_stats_and_log()
                self.num_iterations += 1
                self.reducer.prepare_for_forward()
    
            # Notify the join context that this process has not joined, if
            # needed
            work = Join.notify_join_context(self)
            if work:
                self.reducer._set_forward_pass_work_handle(
                    work, self._divide_by_initial_world_size
                )
    
            # Calling _rebuild_buckets before forward compuation,
            # It may allocate new buckets before deallocating old buckets
            # inside _rebuild_buckets. To save peak memory usage,
            # call _rebuild_buckets before the peak memory usage increases
            # during forward computation.
            # This should be called only once during whole training period.
            if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
                logger.info("Reducer buckets have been rebuilt in this iteration.")
                self._has_rebuilt_buckets = True
    
            # sync params according to location (before/after forward) user
            # specified as part of hook, if hook was specified.
            buffer_hook_registered = hasattr(self, 'buffer_hook')
            if self._check_sync_bufs_pre_fwd():
                self._sync_buffers()
    
            if self._join_config.enable:
                # Notify joined ranks whether they should sync in backwards pass or not.
                self._check_global_requires_backward_grad_sync(is_joined_rank=False)
    
            output = self._run_ddp_forward(*inputs, **kwargs)
    
            # sync params according to location (before/after forward) user
            # specified as part of hook, if hook was specified.
            if self._check_sync_bufs_post_fwd():
                self._sync_buffers()
    
            if torch.is_grad_enabled() and self.require_backward_grad_sync:
                self.require_forward_param_sync = True
                # We'll return the output object verbatim since it is a freeform
                # object. We need to find any tensors in this object, though,
                # because we need to figure out which parameters were used during
                # this forward pass, to ensure we short circuit reduction for any
                # unused parameters. Only if `find_unused_parameters` is set.
                if self.find_unused_parameters and not self.static_graph:
                    # Do not need to populate this for static graph.
                    self.reducer.prepare_for_backward(list(_find_tensors(output)))
                else:
                    self.reducer.prepare_for_backward([])
            else:
                self.require_forward_param_sync = False
    
        # TODO: DDPSink is currently enabled for unused parameter detection and
        # static graph training for first iteration.
        if (self.find_unused_parameters and not self.static_graph) or (
            self.static_graph and self.num_iterations == 1
        ):
            state_dict = {
                'static_graph': self.static_graph,
                'num_iterations': self.num_iterations,
            }
    
            output_tensor_list, treespec, output_is_rref = _tree_flatten_with_rref(
                output
            )
            output_placeholders = [None for _ in range(len(output_tensor_list))]
            # Do not touch tensors that have no grad_fn, which can cause issues
            # such as https://github.com/pytorch/pytorch/issues/60733
            for i, output in enumerate(output_tensor_list):
                if torch.is_tensor(output) and output.grad_fn is None:
                    output_placeholders[i] = output
    
            # When find_unused_parameters=True, makes tensors which require grad
            # run through the DDPSink backward pass. When not all outputs are
            # used in loss, this makes those corresponding tensors receive
            # undefined gradient which the reducer then handles to ensure
            # param.grad field is not touched and we don't error out.
            passthrough_tensor_list = _DDPSink.apply(
                self.reducer,
                state_dict,
                *output_tensor_list,
            )
            for i in range(len(output_placeholders)):
                if output_placeholders[i] is None:
                    output_placeholders[i] = passthrough_tensor_list[i]
    
            # Reconstruct output data structure.
            output = _tree_unflatten_with_rref(
                output_placeholders, treespec, output_is_rref
            )
        return output
    ```
    
## no_sync
- Set `require_backward_grad_sync` to False on `__enter__` to skip [`Reducer.prepare_for_backward`](../Distributed/Reducer.md#reducerprepareforbackward), therefore disable [`autograd_hook`](../Distributed/Reducer.md#reducerautogradhook)
- Restore `require_backward_grad_sync` on `__exit__`
  <details> 
    <summary>Code for forward</summary> 

    ```Python
    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.
    
        Example::
    
            >>> ddp = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            >>>   for input in inputs:
            >>>     ddp(input).backward()  # no synchronization, accumulate grads
            >>> ddp(another_input).backward()  # synchronize grads
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync
    ```
    

## join
https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join

- Context manager to handle uneven inputs across nodes
- Cannot handle additional distributed collective operations(such as `SyncBatchNorm`) other than trigged by DDP
- Return [`Join`](../Distributed/Joinable.md#join)

## join_hook
- Implement base class [`Joinable`](../Distributed/Joinable.md#joinable)’s interface
- Return [`_DDPJoinHook`](#_ddpjoinhook)

# _DDPJoinHook
- Subclass of [`JoinHook`](../Distributed/Joinable.md#joinhook)
- `main_hook`: to fake data(`buffer`, `gradient` and `unused_parameters`) and `all_reduce` to avoid other nodes to be hanged
- `post_hook`: `_sync_final_model`
  <details> 
    <summary>Code for _DDPJoinHook</summary> 
    
    ```Python
    class _DDPJoinHook(JoinHook):
        def __init__(self, ddp, divide_by_initial_world_size):
            """
            Sets config variables for internal usage.
            """
            assert isinstance(ddp, DistributedDataParallel), (
                "DDP join hook requires passing in a DistributedDataParallel "
                "instance as the state"
            )
            ddp.logger._set_uneven_input_join()
            self.ddp = ddp
            self.ddp._divide_by_initial_world_size = divide_by_initial_world_size
            super().__init__()
    
        def main_hook(self):
            """
            Shadows the DDP collective communication operations in the forward and
            backward passes.
            """
            ddp = self.ddp
            # Buckets are rebuilt only once during a training period
            ddp.reducer._rebuild_buckets()
    
            # Schedule a broadcast if we are syncing module buffers in the
            # forward pass
            # TODO: make DDP uneven inputs context manager support buffer
            # comm hook (https://github.com/pytorch/pytorch/issues/65436)
            ddp._check_and_sync_module_buffers()
    
            # Check if need to sync in the backward pass
            work = ddp._check_global_requires_backward_grad_sync(is_joined_rank=True)
            work.wait()
            should_sync_backwards = work.result()[0].item() != 0
            # Forward parameter sync is disabled in the next iteration if we
            # are skipping gradient sync this iteration, so set
            # `require_forward_param_sync` accordingly
            ddp.require_forward_param_sync = should_sync_backwards
            if not should_sync_backwards:
                return
    
            # Schedule one allreduce per gradient bucket to match the backward
            # pass allreduce
            ddp._match_all_reduce_for_bwd_pass()
    
            # Check if we need to allreduce locally unused parameters
            if ddp.find_unused_parameters:
                ddp._match_unused_params_allreduce()
    
            # Rebuilt parameters are pushed only once during a training period
            ddp.reducer._push_all_rebuilt_params()
    
        def post_hook(self, is_last_joiner: bool):
            """
            Syncs the final model to ensure that the model is the same across all
            processes.
            """
            self.ddp._sync_final_model(is_last_joiner)
    ```
  </details>
  

# _DDPSink
- Applied to the outputs of `DistributedDataParallel.forward`
- If this is first iteration for static graph
    - Register [`Reducer._delay_all_reduce`](../Distributed/Reducer.md#reducerdelayallreduce) as callback to autograd engine
- Else, pass though
  <details> 
    <summary>Code for _DDPSink</summary> 
    
    ```Python
    # Add a DDPSink to run various functions when backwards starts, such as
    # queueing call back of out-most backward/graph task,
    # this helps call back is fired after all gradients' calculation
    # is completed.
    class _DDPSink(Function):
        @staticmethod
        def forward(ctx, reducer, state_dict, *inputs):
            # set_materialize_grads(False) will ensure that None gradients stay as
            # None and are not filled with zeros.
            ctx.set_materialize_grads(False)
            ctx.reducer = reducer
            ctx.state_dict = state_dict
            ret = tuple(
                inp.clone()
                if isinstance(inp, torch.Tensor)
                else inp
                for inp in inputs
            )
            return ret
    
        @staticmethod
        def backward(ctx, *grad_outputs):
            state_dict = ctx.state_dict
            # Enqueue delay allreduce for static graph training on the first
            # iteration.
            if ctx.state_dict['static_graph'] and ctx.state_dict['num_iterations'] == 1:
                Variable._execution_engine.queue_callback(ctx.reducer._delay_all_reduce)
    
            return (None, None, *grad_outputs)
    ```
  </details>
    
  

# Reducer
See link to [Reducer](../Distributed/Reducer.md)

# Communication hook
See link to *[Communication hook](https://www.notion.so/Communication-hook-bc42de8bbd2b444f8597c9033633067b)*

https://pytorch.org/docs/stable/ddp_comm_hooks.html

## Paper
[PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704)
