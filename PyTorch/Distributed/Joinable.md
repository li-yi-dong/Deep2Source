<div align='center'><font size='20'> Joinable </font></div>

`torch/distributed/algorithms/join.py`

# Joinable
- Base class and interface of joinable class
- Class member `join_hook` for [`Join`](#join)

## Definition
* <details> 
    <summary>Code for Joinable</summary>  

    ```Python
    class Joinable(ABC):
        r"""
        This defines an abstract base class for joinable classes. A joinable class
        (inheriting from :class:`Joinable`) should implement :meth:`join_hook`,
        which returns a :class:`JoinHook` instance, in addition to
        :meth:`join_device` and :meth:`join_process_group` that return device and
        process group information, respectively.
        """
        @abstractmethod
        def __init__(self):
            super(Joinable, self).__init__()
            self._join_config = _JoinConfig.construct_disabled_join_config()
    
        @abstractmethod
        def join_hook(self, **kwargs) -> JoinHook:
            r"""
            Returns a :class:`JoinHook` instance for the given :class:`Joinable`.
    
            Arguments:
                kwargs (dict): a :class:`dict` containing any keyword arguments
                    to modify the behavior of the join hook at run time; all
                    :class:`Joinable` instances sharing the same join context
                    manager are forwarded the same value for ``kwargs``.
            """
            ...
    
        @property
        @abstractmethod
        def join_device(self) -> torch.device:
            r"""
            Returns the device from which to perform collective communications
            needed by the join context manager implementation itself.
            """
            ...
    
        @property
        @abstractmethod
        def join_process_group(self) -> Any:
            r"""
            Returns the process group for the collective communications needed by
            the join context manager itself.
            """
            ...
    ```
  </details> 
    

# JoinHook

## Definition
* <details> 
    <summary>Code for JoinHook</summary>
    
    ```python
    class JoinHook():
        r"""
        This defines a join hook, which provides two entry points in the join
        context manager: a main hook, which is called repeatedly while there exists
        a non-joined process, and a post-hook, which is called once all processes
        have joined.
    
        To implement a join hook for the generic join context manager, define a
        class that inherits from :class:`JoinHook` and override ``main_hook()`` and
        ``post_hook()`` as appropriate.
        """
        def main_hook(self) -> None:
            r"""
            This hook is called repeatedly while there exists a non-joined process
            to shadow collective communications in one training iteration (i.e. in
            one forward pass, backward pass, and optimizer step).
            """
            ...
    
        def post_hook(self, is_last_joiner: bool) -> None:
            r"""
            This hook is called after all processes have joined. It is passed an
            additional ``bool`` argument ``is_last_joiner``, which indicates if the
            rank is one of the last to join.
    
            Arguments:
                is_last_joiner (bool): ``True`` if the rank is one of the last to
                    join; ``False`` otherwise.
            """
            ...
    ```
  </details> 
    

# Join
- A context manager of which trigger dummy `all_reduce` when needed
- Trigger an `all_reduce` with 1 in `notify_join_context`
- Trigger an `all_reduce` with 0 in `__exit__`
- If any process reach `__exit__` early than others, `all_reduce` triggered in `__exit__` will get result > 0, then
    - It knows that it finishes earlier that others (be assigned with fewer data for DDP process)
    - Into a loop until all processes reach `__exit__` , in the loop
        - Trigger `Joinable.join_hook.main_hook` to fake any collective operation to avoid hanging other processes
        - Trigger an `all_reduce` with 0, if the result is 0 (all processes reach `__exit__`), break the loop
    - Trigger `Joinable.join_hook.post_hook` to do some finish up

## \_\_exit__
* <details> 
    <summary>Code for __exit__</summary>
    
    ```Python
    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType]
    ):
        r"""
        Repeatedly runs the main hooks until all processes join; then, runs
        the post-hooks.
    
        Raises:
            RuntimeError
                If ``throw_on_early_termination=True``.
        """
        if not self._enable or type:
            return  # propagate the exception directly if one was raised
    
        all_procs_joined = False
        is_last_joiner = True
    
        i = 0
        WARN_THRESHOLD = 1000
        warnings.simplefilter("once")
    
        while not all_procs_joined:
            if i > WARN_THRESHOLD:
                warnings.warn(
                    "Detected uneven input skew of greater than "
                    f"{WARN_THRESHOLD}. This means that rank "
                    f"{self._rank} has at least {WARN_THRESHOLD} "
                    f"fewer inputs than other currently-active ranks. "
                    "This level of skew could lead to performance "
                    "degradation during training."
                )
            # Shadow the all-reduce in non-joined processes
            num_nonjoined_procs = self._get_num_nonjoined_procs()
            if num_nonjoined_procs == 0:
                all_procs_joined = True
            else:
                if self._throw_on_early_termination:
                    self._notify_procs_to_terminate()
    
                # Run main hooks
                for join_hook in self._join_hooks:
                    join_hook.main_hook()
    
                is_last_joiner = False
                i += 1
    
        # Run post-hooks
        for join_hook in self._join_hooks:
            join_hook.post_hook(is_last_joiner)
    ```
    </details>
    

## notify_join_context
- Trigger an `all_reduce` with 1 and return its `work`
  <details> 
    <summary>Code for notify_join_context</summary>
    
    ```Python
    @staticmethod
    def notify_join_context(joinable: Joinable):
        r"""
        Notifies the join context manager that the calling process has not yet
        joined; then, if ``throw_on_early_termination=True``, checks if uneven
        inputs have been detected (i.e. if one process has already joined) and
        throws an exception if so.
    
        This method should be called from a :class:`Joinable` object before
        its per-iteration collective communications. For example, this should
        be called at the beginning of the forward pass in
        :class:`DistributedDataParallel`.
    
        Only the first :class:`Joinable` object passed into the context
        manager performs the collective communications in this method, and
        for the others, this method is vacuous.
    
        Arguments:
            joinable (Joinable): the :class:`Joinable` object calling this
                method.
    
        Returns:
            An async work handle for the all-reduce meant to notify the context
            manager that the process has not yet joined if ``joinable`` is the
            first one passed into the context manager; ``None`` otherwise.
        """
        assert hasattr(joinable, "_join_config"), \
            f"Check that the {type(joinable)} constructor calls the " \
            "``Joinable`` constructor"
    
        join_config = joinable._join_config
        # First joinable is responsible for the collective communications
        if not join_config.is_first_joinable or not join_config.enable:
            return None
    
        device = joinable.join_device
        process_group = joinable.join_process_group
    
        # Schedule an all-reduce to indicate that the caller has not yet joined
        ones = torch.ones(1, device=device)
        work = dist.all_reduce(ones, group=process_group, async_op=True)
    
        if join_config.throw_on_early_termination:
            # Check if uneven inputs have been detected
            zeros = torch.zeros(1, device=device)
            dist.all_reduce(zeros, group=process_group)
            should_throw = zeros.item()
            if should_throw:
                raise RuntimeError(
                    "Detected at least one rank that exhausted inputs. "
                    "Throwing across all ranks."
                )
        return work
    ```
  </details>