# Cost
2 kinds: inter-nodes and intra-nodes
## Latency
- inter-nodes
- Only the start and end OP for the same collective has KHighLatency of 5000
- Otherwise with KLowLatency of 1
## NodeCost
- intra-nodes
- LoopFusion instruction has KMediumCost
- OutPutFusion and Convolution has KHightCost
- Otherwise has KLowCost

# Resource
## AsyncTracker::ResourcesPerInstruction
- How many SupportedAsyncDone recursively in the Instruction
  <details> 
      <summary>Resource definition</summary>  

  ```C++
    enum class ResourceType {
    kNoResource = 0,
    kAllToAll = 1,
    kAllGather = 2,
    kAllReduce = 3,
    kCollectivePermute = 4,
    kSendRecv = 5,
    kSendHost = 6,
    kRecvHost = 7,
    kNumResources = 8,
    };
  ```
  </details>

# Policy (ReadySetLT)
Priority from top to bottom
## Delay nodes with `ForceDelay` tag as far as possible
## Schedule `NOP` first
- `NOP` including:
  - `GetTupleElement`
  - `BitCast`
  - `Transpose` && `TransposeIsBitCast`
## memory usage first if mem tracking is enabled and peak mem usage > mem limit / 2
- If current mem usage >= limit
  - Schedule node with smallest mem pressure (release more or use less)
- Else
  - Just ensure not to exceed the limit
## Prioritize certain `AsyncDone`
- Node of which occupy some resource
- Delay `SendDone` or occupy `SendHost`
- Delay estimated not ready node:
  <details> 
    <summary>Estimation for ready or not</summary>  

    ```C++
      const HloGraphNode& start =
          sched_state_.sched_graph.GetNode(gn.GetInstr().operand(0));
      const LatencyEstimator::TimeCost latency =
          sched_state_.latency_estimator->GetLatencyBetween(start, gn);
      if (start.GetReadyTime() - sched_state_.current_time <= latency) {
        return false;
      }
    ```
  </details>
## Schedule node with less ready interval (ReadyTime - CurrentTime)
## If aggressive
- Delay node that is not resource constrained && `AsyncDepth` is 0
  <details> 
    <summary>AsyncDepth</summary>  

    ```C++
    while (!stack.empty()) {
      auto* node = stack.back();
      stack.pop_back();
      if (async_tracker->IsSupportedAsyncDone(node->GetInstr())) {
        for (auto& pred : node->GetPredecessors()) {
          node->SetAsyncDepth(
              std::max(pred.Target().GetAsyncDepth() + pred.Latency(),
                      node->GetAsyncDepth()));
        }
      } else {
        for (auto& pred : node->GetPredecessors()) {
          node->SetAsyncDepth(
              std::max(pred.Target().GetAsyncDepth(), node->GetAsyncDepth()));
        }
      }
      for (auto& succ : node->GetSuccessors()) {
        if (--current_rank[&succ.Target()] == 0) {
          stack.push_back(&succ.Target());
        }
      }
    }
    ```
  </details>
## Prioritize node that would release occupied constrained resource
## If aggressive
- Prioritize node with greater `AsyncDepth`
## If aggressive
- Prioritize node that cost closest to the estimated latency it may hide
  <details> 
    <summary>Closest to estimated latency</summary>  

    ```C++
      if (!sched_state_.next_ready_stack.empty()) {
        HloGraphNode::TimeCost latest_ready =
            sched_state_.next_ready_stack.front()->GetReadyTime();
        HloGraphNode::TimeCost a_cost_diff = std::abs(
            latest_ready - sched_state_.current_time - a.node->GetCost());
        HloGraphNode::TimeCost b_cost_diff = std::abs(
            latest_ready - sched_state_.current_time - b.node->GetCost());
        if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
                a_cost_diff < b_cost_diff, a, b_cost_diff < a_cost_diff, b,
                "kAvoidWaste")) {
          return *value;
        }
      }
    ```
  </details>
## Prioritize the node with `AsyncDone` in its operands
## Obey `target_scheduling_rule`
## Prioritize the node would unlock more nodes
## Memory pressure
## Original order

# Remaining questions
## Schedule root first?