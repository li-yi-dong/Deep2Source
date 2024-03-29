<div align='center'><font size='40'> Call stack of xla::gpu::GpuCompiler</font></div>

# Call stack
- xla::Service::Compile()
    - xla::Service::CreateModuleConfig
    - xla::Service::BuildExecutable()
    - xla::CreateModuleFromProto()
    - xla::UpdateEntryComputationLayout()
    - xla::DumpHloModuleIfEnabled()
    - if !run_backend_only
        - backend->compiler()->Compile()
            - for each module:
                - xla::gpu::GpuCompiler::RunHloPasses()
                    - tsl::profiler::TraceMe
                    - xla::gpu::GpuCompiler::OptimizeHloModule()
                        - if hlo_module->config().num_partitions() > 1
                            - SPMD
                        - else
                            - sharding_removal
                        - more passes
                    - xla::gpu::GpuCompiler::PrepareHloModuleForIrEmitting()
                - xla::gpu::GpuCompiler::RunBackend()
                    - Profile
                    - xla::gpu::CompileModuleToLlvmIrImpl()
                        - xla::gpu::ScheduleGpuModule()
                            - xla::gpu::ScheduleGpuModuleWithMemoryScheduler()
                                - if not LatencyHidingScheduler enabled
                                    - enable PostprocessorToScheduleAsEarlyOrLateAsPossible
                                - try List, DFS and Postorder then pick the one with lower memory usage
                                - Execute LatencyHidingScheduler as a pass
                        - OptimizationBarrierExpander pass
                        - xla::BufferAssigner::Run()
                        - mlir stuff
                        - IrEmitter stuff
  - ......