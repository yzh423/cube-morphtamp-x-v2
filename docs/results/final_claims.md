# Final Results and Claim Boundaries

This section reports the current evidence for Cube/Object MorphTAMP-X v2. The evaluation uses MuJoCo simulation with a Franka/Panda arm model, geometric object proxies, scripted pick-place phase plans, collision-aware replay checks, grasp-strategy selection, and an equivalent morphology-cost model. The results support claims about simulated task feasibility and morphology trade-offs under the implemented benchmark. They do not establish real-hardware performance, force-controlled grasp retention, or manufactured redesign feasibility.

## Result 1: Object-task feasibility after task-geometry correction

After correcting the narrow-slot target geometry so that the placed object volume no longer overlaps the obstacle, the object-task benchmark completed 110 successful cases out of 110 evaluated cases. The benchmark includes 10 object types and 11 task scenes. The measured success rate is 1.000, and the maximum reported position error is 0.036162 m under a 0.05 m tolerance. The analyzed result file reports zero failed cases. This establishes that the current scripted planner, grasp-selection layer, scene construction, and replay evaluator can complete the fixed benchmark suite in simulation.

## Result 2: Robust reach margin exposes a morphology trade-off

A permissive morphology comparison ranked `short_arm` first because every tested morphology completed the benchmark when near-boundary reach was accepted. This was not a useful optimization result because it rewarded the lowest-cost design even when that design had very small reach margin on the hardest tasks. The robust morphology benchmark therefore used `minimum_reach_margin = 0.03` m. Under this stricter reachability condition, the benchmark completed 643 successful design-task-object evaluations out of 660. The best design changed from the shortest arm to `compact_base_shift`.

## Result 3: Compact base shift is the best current feasible-cost design

The `compact_base_shift` design completed 110 of 110 cases under the 0.03 m reach-margin constraint. Its morphology cost is 0.805030, and its minimum reach margin is 0.091555 m. Compared with `nominal_panda`, this corresponds to an approximate 6.39% reduction in the morphology-cost proxy while preserving full benchmark success. Among the full-success designs, `compact_base_shift` has the lowest morphology cost, which makes it the current best cost-feasible morphology in this benchmark.

## Result 4: The shortest arm is not robust across the full task suite

The `short_arm` design has the lowest morphology cost (0.722080), but it succeeds on only 93 of 110 cases when a 0.03 m reach margin is required. Its 17 failures all occur in the `short_arm` condition and are concentrated in long-reach or compound scenes, especially `diagonal_reach_around` (10 failures) and `compound_shelf_barrier` (7 failures). This result supports the design principle that minimizing arm length alone is insufficient; morphology optimization must include workspace demand, task geometry, and robust reachability margins.

## Claim-safe interpretation

The current evidence supports the following bounded claim: in the implemented MuJoCo benchmark, with geometric object proxies and a 0.03 m minimum reach margin, `compact_base_shift` is the lowest-cost morphology among the tested designs that maintains full object-task success. This claim is stronger than a toy reach-only result because it is conditioned on grasp-strategy selection, object-task variation, collision-aware replay, and task-specific workspace demand. However, it remains a simulation claim. A report should avoid saying that the design is physically optimal for a real Panda robot or that it guarantees real-world manipulation success.

## Recommended wording for the report

A concise report sentence is: "Under a 3 cm robust reachability margin, the compact base-shift morphology achieved full simulated object-task success with a lower morphology-cost proxy than the nominal Panda, whereas the shortest arm failed on long-reach and compound tasks." A stronger but still safe version is: "These results indicate that constrained morphology optimization should not minimize arm length alone; the best design in this benchmark balances compactness with task-dependent reachability margin."

## Limitations and next evidence gap

The current benchmark uses simplified object geometries, phase-level scripted motion, and an equivalent morphology-cost proxy. It does not model a physically manufactured Franka/Panda redesign, nor does it validate force-closure, tactile feedback, contact-rich grasp retention, or sim-to-real transfer. The next evidence gap is robustness: repeated seeds, object/target perturbations, fixed-scene evaluation without auto-fitting, and sensitivity to the reach-margin threshold should be evaluated before making a stronger research-paper claim.

## Source files

The object-task benchmark evidence is stored in `results/fresh_object_task_benchmark_after_slot_fix.json` and summarized in `results/fresh_benchmark_summary_after_slot_fix.json`. The robust morphology benchmark is stored in `results/morphology_benchmark_after_slot_fix_margin003.json`. The companion table file is `results/final_tables.md`.
