<div align='center'><font size='20'> Combiner Passes </font></div>

- Combine small collective communications into big one
- Save the overhead for establishing the communications

# Supported primitives
- AllGather
- AllReduce
- ReduceScatter
- A stand alone pass for each primitive

# CombineInstructionsByKey
`tensorflow/compiler/xla/service/collective_combiner_utils.h`
- Combines instructions with matching keys together
- Instructions are combined in topological post-order
## Procedure
- Find all instructions with target key (`keys`)
- Find all keys and tracking all instructions of each key (`groups`)
- For each key
  - Recompute reachability of the computation
  - For each instruction in the computation with post order
    - If not target instruction with target key, continue
    - If the instruction is not in the `group` of the key, continue
    - If the instruction with more than 1 operand, continue
    - If the size of the instruction is bigger than threshold, continue
    - If the size of the current instruction + previous > threshold, break
    - If the instruction is dependent with any of previous, break
    - Push the instruction to `to_combine` and tracking the size of `to_combine`
    - If number of instructions in `to_combine` reach the threshold, break
  - Call `combine_fn` on `to_combine` and set `changed` to `true`
  <details> 
      <summary>CombineInstructionsByKey</summary>  

    ```C++
    // Combines instructions with matching keys together.
    //
    // Instructions are combined in topological post-order.
    //
    // `key_fn` should return equal keys for two instructions that might be combined
    // together. Instructions will be combined until the threshold for output byte
    // size or instruction count is reached.
    template <typename K>
    StatusOr<bool> CombineInstructionsByKey(
        HloComputation* computation,
        absl::FunctionRef<std::optional<K>(const HloInstruction*)> key_fn,
        absl::FunctionRef<Status(absl::Span<HloInstruction* const>)> combine_fn,
        int64_t combine_threshold_bytes, int64_t combine_threshold_count) {
    // Cache keys for each instruction and build sets of instructions with the
    // same key that might be combined together.
    absl::flat_hash_map<HloInstruction*, K> keys;
    absl::flat_hash_map<K, absl::flat_hash_set<HloInstruction*>> groups;

    for (HloInstruction* instruction : computation->instructions()) {
        std::optional<K> key = key_fn(instruction);
        if (key) {
        keys.insert({instruction, *key});
        groups[*key].insert(instruction);
        }
    }

    bool changed = false;

    // Keys are removed after the instruction is combined (or never will be).
    while (!keys.empty()) {
        std::vector<HloInstruction*> to_combine;
        int64_t to_combine_bytes = 0;
        absl::flat_hash_set<HloInstruction*>* group = nullptr;

        // Recompute reachability after every combine group because we can't
        // maintain a cross group topological order to be able to rely on the
        // transitive dependencies to detect cycles.
        std::unique_ptr<HloReachabilityMap> reachability =
            HloReachabilityMap::Build(computation);

        for (HloInstruction* instruction :
            computation->MakeInstructionPostOrder()) {
        auto it = keys.find(instruction);
        if (it == keys.end()) continue;

        // If this is the first instruction, set the active group.
        if (to_combine.empty()) {
            group = &groups.find(it->second)->second;
        }

        // Check instruction is in the active group.
        if (group->find(instruction) == group->end()) {
            continue;
        }

        VLOG(1) << "Considering HLO " << instruction->ToString()
                << " with current set size of " << to_combine_bytes
                << " and current operand count of " << to_combine.size();

        // We do not handle ops that have more than one operand since that is
        // simpler and this pass is the only way to generate such ops.
        if (instruction->operands().size() != 1) {
            VLOG(1) << "Skipping due to " << instruction->operands().size()
                    << " operands";
            keys.erase(it);
            continue;
        }

        TF_RET_CHECK(instruction->shape().IsArray());
        int64_t instruction_bytes = ShapeUtil::ByteSizeOf(instruction->shape());

        // If the instruction is greater than the threshold, then we can never
        // combine it with anything.
        if (instruction_bytes > combine_threshold_bytes) {
            VLOG(1) << "Size " << instruction_bytes << " above threshold.";
            keys.erase(it);
            continue;
        }

        if (to_combine_bytes + instruction_bytes > combine_threshold_bytes) {
            VLOG(1) << "Combined size threshold exceeded.";
            break;
        }

        // We can't combine dependent instructions.
        bool is_reachable =
            absl::c_any_of(to_combine, [&](HloInstruction* to_combine_inst) {
                return reachability->IsReachable(to_combine_inst, instruction);
            });
        if (is_reachable) {
            VLOG(1) << "Instruction is reachable.";
            break;
        }

        VLOG(1) << "Adding instruction to set.";
        to_combine.push_back(instruction);
        to_combine_bytes += instruction_bytes;
        keys.erase(it);

        if (to_combine.size() >= combine_threshold_count) {
            VLOG(1) << "Combined count threshold reached.";
            break;
        }
        }

        if (to_combine.size() > 1) {
        TF_RETURN_IF_ERROR(combine_fn(to_combine));
        changed = true;
        }
    }

    return changed;
    }
    ```
  </details>

# AllReduce
`tensorflow/compiler/xla/service/all_reduce_combiner.h`
## Initialization of the pass
- 2 threshold
  - threshold in bytes
    - Combine small AllReduces to a big one up to this limit size
    - less or equal to 0 to disable this pass
  - threshold count
    - Combine up to this limit number of small AllReduces to a big one
    - less or equal to 0 to disable this pass

## Run of the pass
- Return if disabled
- Return if the module has layout constrained AllReduces
  <details> 
      <summary>LayoutConstrainedCollective</summary>  

    ```C++
    bool ContainsLayoutConstrainedCollective(const HloModule& module,
                                             HloOpcode op) {
      CHECK(IsCollectiveCommunicationOp(op));

      for (auto computation : module.computations()) {
        for (auto hlo : computation->instructions()) {
          if (hlo->opcode() == op &&
              DynCast<HloCollectiveInstruction>(hlo)->constrain_layout()) {
            return true;
          }
        }
      }
      return false;
    }
    ```
  </details>
- For all non-fusion computations
  - Construct an `HloDomainMap`
  - Define `key_fn`
    - Return a key by applying [GetAllReduceKey](#getallreducekey) to the instruction
  - Apply [CombineInstructionsByKey](#combineinstructionsbykey) to the computation

# AllReduceKey
`tensorflow/compiler/xla/service/all_reduce_key.h`
## AllReduceKey
```C++
// Encapsulates all of the properties which must match for two all-reduce
// instructions to be compatible with each other (and hence be possible to
// combine the instructions).
using AllReduceKey =
    std::tuple<HloOpcode, PrimitiveType,
               /*domain metadata id*/ int64_t,
               /*has channel id*/ bool,
               /*use_global_device_ids*/ bool,
               /*replica_groups*/ std::vector<std::vector<int64_t>>>;
```
## GetAllReduceKey
- Return equal key for AllReduce instructions that can be combined
  <details> 
      <summary>GetAllReduceKey</summary>  

    ```C++
    // Returns a key that will be equal for all-reduce instructions that are
    // compatible with each other, and hence might be combined, or different if not.
    std::optional<AllReduceKey> GetAllReduceKey(const HloInstruction* instruction,
                                                const HloDomainMap* domain_map,
                                                bool ignore_replica_groups) {
    if (instruction->opcode() != HloOpcode::kAllReduce &&
        instruction->opcode() != HloOpcode::kReduceScatter) {
        return std::nullopt;
    }

    if (instruction->to_apply()->instruction_count() != 3 ||
        instruction->to_apply()->num_parameters() != 2) {
        VLOG(1) << "Skipping due to non-trivial reduction function.";
        return std::nullopt;
    }

    const auto* ar = Cast<HloAllReduceInstructionBase>(instruction);

    std::vector<std::vector<int64_t>> replica_groups;
    if (!ignore_replica_groups) {
        replica_groups.reserve(ar->replica_groups().size());
        for (const ReplicaGroup& replica_group : ar->replica_groups()) {
        replica_groups.push_back(
            std::vector<int64_t>(replica_group.replica_ids().begin(),
                                replica_group.replica_ids().end()));
        }
    }

    const HloInstruction* to_apply_root = ar->to_apply()->root_instruction();
    // Domain metadata id returned by `GetDomainMetadataId` is guaranteed to be >=
    // 0, so use -1 when we don't need to track domain metadata id.
    int64_t domain_metadata_id =
        domain_map ? domain_map->GetDomainMetadataId(ar) : -1;
    return AllReduceKey{
        to_apply_root->opcode(),     to_apply_root->shape().element_type(),
        domain_metadata_id,          ar->channel_id().has_value(),
        ar->use_global_device_ids(), replica_groups};
    }
    ```
  </details>