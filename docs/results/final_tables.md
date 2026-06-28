# Final Result Tables

The following tables summarize the accepted non-figure evidence for Cube/Object MorphTAMP-X v2. They are generated from JSON result files and should be used together with `results/final_claims.md`.

## Table 1. Benchmark scope

| Item | Value |
|---|---:|
| Object types | 10 |
| Task scenes | 11 |
| Object-task cases | 110 |
| Arm morphology designs | 6 |
| Morphology evaluations | 660 |

## Table 2. Object-task feasibility

| Metric | Value |
|---|---:|
| Successful cases | 110 / 110 |
| Success rate | 1.000 |
| Maximum position error (m) | 0.036162 |
| Failed cases after geometry fix | 0 |

## Table 3. Task split coverage

| Split | Tasks |
|---|---|
| development | `tabletop_easy`, `over_barrier` |
| main | `shelf_place`, `shelf_pick`, `narrow_slot`, `folded_transfer`, `far_corner`, `around_wall`, `over_barrier` |
| heldout | `compound_shelf_barrier`, `diagonal_reach_around`, `under_bridge` |
| unassigned | — |

## Table 4. Robust morphology benchmark (`minimum_reach_margin = 0.03` m)

| Rank | Design | Successes / Runs | Success Rate | Morphology Cost | Min Reach Margin (m) | Mean Path Length | Interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `compact_base_shift` | 110 / 110 | 1.000 | 0.805030 | 0.091555 | 2.239634 | best feasible-cost trade-off |
| 2 | `nominal_panda` | 110 / 110 | 1.000 | 0.860000 | 0.154025 | 2.239634 | feasible but higher cost |
| 3 | `long_wrist` | 110 / 110 | 1.000 | 0.906420 | 0.200445 | 2.239634 | feasible but higher cost |
| 4 | `long_forearm` | 110 / 110 | 1.000 | 0.916880 | 0.210905 | 2.239634 | feasible but higher cost |
| 5 | `high_reach_arm` | 110 / 110 | 1.000 | 0.948430 | 0.242455 | 2.239634 | feasible but higher cost |
| 6 | `short_arm` | 93 / 110 | 0.845 | 0.722080 | 0.016105 | 2.119706 | lowest cost but not robust |

## Table 5. Failure distribution under robust reach margin

| Category | Count |
|---|---:|
| design:`short_arm` | 17 |
| task:`compound_shelf_barrier` | 7 |
| task:`diagonal_reach_around` | 10 |
| object:`bowl_proxy` | 2 |
| object:`capsule` | 2 |
| object:`cube` | 2 |
| object:`cylinder` | 2 |
| object:`flat_box` | 1 |
| object:`mug_proxy` | 2 |
| object:`plate` | 1 |
| object:`ring` | 1 |
| object:`sphere` | 2 |
| object:`tall_box` | 2 |

## Table 6. Representative `short_arm` failures

| Object | Task | Reach Margin (m) | Required Margin (m) | Workspace Demand | Failure Reason |
|---|---|---:|---:|---:|---|
| `capsule` | `compound_shelf_barrier` | 0.016105 | 0.030000 | 0.705975 | `reach_margin:0.016105` |
| `tall_box` | `compound_shelf_barrier` | 0.016105 | 0.030000 | 0.705975 | `reach_margin:0.016105` |
| `capsule` | `diagonal_reach_around` | 0.019869 | 0.030000 | 0.702211 | `reach_margin:0.019869` |
| `tall_box` | `diagonal_reach_around` | 0.019869 | 0.030000 | 0.702211 | `reach_margin:0.019869` |
| `mug_proxy` | `compound_shelf_barrier` | 0.020297 | 0.030000 | 0.701783 | `reach_margin:0.020297` |
| `mug_proxy` | `diagonal_reach_around` | 0.021081 | 0.030000 | 0.700999 | `reach_margin:0.021081` |
| `cylinder` | `diagonal_reach_around` | 0.021634 | 0.030000 | 0.700446 | `reach_margin:0.021634` |
| `bowl_proxy` | `diagonal_reach_around` | 0.022151 | 0.030000 | 0.699929 | `reach_margin:0.022151` |
| `cylinder` | `compound_shelf_barrier` | 0.022348 | 0.030000 | 0.699732 | `reach_margin:0.022348` |
| `sphere` | `diagonal_reach_around` | 0.022906 | 0.030000 | 0.699174 | `reach_margin:0.022906` |
| `cube` | `diagonal_reach_around` | 0.023081 | 0.030000 | 0.698999 | `reach_margin:0.023081` |
| `ring` | `diagonal_reach_around` | 0.023868 | 0.030000 | 0.698212 | `reach_margin:0.023868` |

## Table 7. Claim boundaries

| Statement Type | Safe Wording | Avoid |
|---|---|---|
| Feasibility | The implemented MuJoCo benchmark reports 110/110 object-task successes. | The robot is guaranteed to succeed in the real world. |
| Morphology | `compact_base_shift` is the best tested design under the current cost proxy and 3 cm reach margin. | `compact_base_shift` is globally optimal. |
| Short arm | `short_arm` is lowest cost but fails robust reach constraints on selected long/compound tasks. | Short arms are generally bad. |
| Evidence scope | Results are simulation-stage evidence with simplified geometric objects and scripted phase plans. | Results prove hardware-level grasping or manufacturing feasibility. |

## Table 8. Source files

| Artifact | Path |
|---|---|
| Object-task benchmark | `results/fresh_object_task_benchmark_after_slot_fix.json` |
| Object-task summary | `results/fresh_benchmark_summary_after_slot_fix.json` |
| Object-task summary CSV | `results/fresh_benchmark_summary_after_slot_fix.csv` |
| Robust morphology benchmark | `results/morphology_benchmark_after_slot_fix_margin003.json` |
| Result claims | `results/final_claims.md` |
| Result tables | `results/final_tables.md` |
