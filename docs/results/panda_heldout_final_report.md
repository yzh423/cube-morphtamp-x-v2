# Panda-Aware Held-Out Morphology Optimization Report

## Abstract

This report summarizes the current final evidence for Cube MorphTAMP-X v2, a MuJoCo-based research prototype for task-aware robot arm morphology optimization. The system evaluates object-conditioned pick-and-place tasks with a 7-DoF Franka/Panda MuJoCo model, multi-candidate grasp planning, collision-aware replay checks, inverse-kinematics quality metrics, and an equivalent arm morphology cost proxy. The central objective is to search for a lower-cost arm morphology while maintaining task reachability, robust workspace margin, and singularity safety. In the held-out Panda-aware search, the optimizer evaluated 375 continuous morphology candidates and found 234 feasible designs. The best feasible design was `opt_u0.900_f0.900_w0.820_bx0.000`, corresponding to upper-arm scale 0.90, forearm scale 0.90, wrist scale 0.82, and no base shift. This design achieved full held-out success, a morphology cost of 0.75712, a minimum reach margin of 0.05115 m, a minimum singular value of 0.24051, and a maximum Jacobian condition number of 4.09616. A subsequent Panda-aware robustness evaluation over 150 perturbed held-out runs achieved 150/150 successes with zero failure-analysis entries. These results support the bounded claim that, within the implemented MuJoCo benchmark and cost model, a compact equivalent Panda morphology can reduce morphology cost while preserving reachability, singularity margin, and task feasibility under the tested perturbations.

## Method Overview

Cube MorphTAMP-X v2 is organized around a task-and-motion-inspired evaluation loop. Each benchmark case begins with an object type and a task scene, such as a compound shelf-and-barrier transfer, a diagonal reach-around task, or an under-bridge manipulation task. The planner generates grasp candidates conditioned on object geometry and task constraints, ranks them using kinematic and geometric feasibility metrics, builds a MuJoCo scene containing the Franka/Panda robot and task objects, and solves a Panda joint replay for the selected candidate. The replay evaluation records position error, orientation quality where applicable, joint margin, joint-path length, singular-value statistics, condition number, collision-related failure reasons, and task success.

The morphology optimizer searches over an equivalent arm design space rather than manufacturing a new physical Panda robot. A design is parameterized by upper-arm, forearm, and wrist scale factors, together with an optional base translation. Each design receives a morphology cost and an equivalent reach proxy. Candidate designs are accepted only if all selected object-task cases remain successful, the minimum reach margin exceeds the configured threshold, and the Panda-aware Jacobian metrics satisfy the configured singularity constraints. This means the reported design should be interpreted as a task-aware equivalent reach-length optimization result, not as a physically validated mechanical redesign.

## Experimental Setup

The final held-out optimization used the `heldout_fixed` protocol. This protocol evaluates three stress tasks: `compound_shelf_barrier`, `diagonal_reach_around`, and `under_bridge`. The Panda-aware evaluation used the Franka/Panda MuJoCo model located at `/home/yzh/robocasa/mujoco_menagerie/franka_emika_panda/scene.xml`. The morphology search considered 375 continuous grid candidates from upper, forearm, and wrist scale values `[0.70, 0.82, 0.90, 1.00, 1.10]` and base-x values `[0.00, 0.03, 0.05]`. The optimizer enforced a minimum reach margin of 0.05 m, a minimum singular value threshold of 0.08, a maximum condition number threshold of 30, and a path-cost weight of 0.02.

The full robustness evaluation used the same three held-out tasks and all ten object types in the current object library: `cube`, `sphere`, `cylinder`, `plate`, `mug_proxy`, `bowl_proxy`, `capsule`, `tall_box`, `flat_box`, and `ring`. For each object-task pair, five perturbation trials were generated using seed 42, object/target position noise of 0.01 m, and obstacle-position noise of 0.01 m. The final robustness experiment therefore contained 150 Panda-aware validation runs. The experiment used `full_candidate_limit = 1` and `position_tolerance = 0.05` m. The robustness implementation writes incremental cache files, so interrupted runs can resume without recomputing already completed Panda validations.

## Morphology Optimization Results

The held-out Panda-aware search evaluated 375 morphology candidates and found 234 feasible candidates. The selected design was `opt_u0.900_f0.900_w0.820_bx0.000`. This design scaled the upper arm and forearm to 90% of the nominal equivalent length and scaled the wrist to 82%, with no base translation. Its morphology cost was 0.75712 and its final objective value was 0.81197. The design achieved a minimum held-out reach margin of 0.05115 m, slightly above the required 0.05 m threshold. The Panda-aware singularity metrics were well within the configured constraints: the minimum singular value was 0.24051, compared with the threshold of 0.08, and the maximum condition number was 4.09616, compared with the threshold of 30.

These results support a useful design conclusion. The optimizer did not simply select the shortest possible arm; instead, it selected the lowest-cost design that still preserved enough reach margin on the held-out task distribution. The wrist segment was shortened more aggressively than the upper arm and forearm, which is consistent with a task set dominated by gross reach and obstacle clearance rather than extreme distal dexterity. The selected design is therefore a compact but not degenerate morphology: it reduces the equivalent arm length while preserving the reach and Jacobian quality needed for the evaluated Panda tasks.

| Metric | Value |
|---|---:|
| Search candidates | 375 |
| Feasible candidates | 234 |
| Best design | `opt_u0.900_f0.900_w0.820_bx0.000` |
| Upper-arm scale | 0.90 |
| Forearm scale | 0.90 |
| Wrist scale | 0.82 |
| Base shift | `[0.0, 0.0, 0.0]` |
| Morphology cost | 0.75712 |
| Objective | 0.81197 |
| Mean path length | 2.74255 |
| Minimum reach margin | 0.05115 m |
| Minimum singular value | 0.24051 |
| Maximum condition number | 4.09616 |
| Singularity constraints passed | true |

## Full Robustness Results

The robustness experiment evaluated the selected Panda-aware pipeline under small perturbations to object, target, and obstacle positions. Across 150 held-out perturbation runs, all 150 runs succeeded. The maximum Panda replay position error was 0.02449 m, which remained below the 0.05 m position tolerance, and the mean position error was 0.00112 m. The worst singularity metric over all robustness runs remained safe relative to the optimizer constraints: the minimum singular value over the robustness set was 0.23390, and the maximum condition number was 4.21181. The mean condition number was 3.55907, and the mean joint-path length was 5.74200.

The robustness results cover all ten object proxies and all three held-out stress tasks. Every object type completed 15/15 perturbed runs. Every held-out task completed 50/50 perturbed runs. This does not prove real-world contact robustness, but it does show that the current simulated solution is not a single-case artifact. The selected design and grasp-planning pipeline remained feasible across a controlled perturbation set containing object variation, task variation, and pose noise.

| Metric | Value |
|---|---:|
| Robustness runs | 150 |
| Successful runs | 150 |
| Success rate | 1.000 |
| Position/obstacle noise | 0.01 m |
| Max position error | 0.02449 m |
| Mean position error | 0.00112 m |
| Minimum singular value | 0.23390 |
| Mean singular value | 0.26141 |
| Maximum condition number | 4.21181 |
| Mean condition number | 3.55907 |
| Mean joint-path length | 5.74200 |

| Object type | Successes / Runs |
|---|---:|
| `bowl_proxy` | 15 / 15 |
| `capsule` | 15 / 15 |
| `cube` | 15 / 15 |
| `cylinder` | 15 / 15 |
| `flat_box` | 15 / 15 |
| `mug_proxy` | 15 / 15 |
| `plate` | 15 / 15 |
| `ring` | 15 / 15 |
| `sphere` | 15 / 15 |
| `tall_box` | 15 / 15 |

| Held-out task | Successes / Runs |
|---|---:|
| `compound_shelf_barrier` | 50 / 50 |
| `diagonal_reach_around` | 50 / 50 |
| `under_bridge` | 50 / 50 |

## Failure Analysis

The full robustness failure analysis reported zero failed runs. There were no failure entries by design, no failures by task, no failures by object, and no failure reason codes. This is useful for the current report because it allows the robustness claim to be stated cleanly: under the implemented perturbation settings, all evaluated Panda-aware held-out cases remained feasible. At the same time, the absence of failures should not be overinterpreted as global robustness. The tested perturbation radius was 0.01 m, the object models were simplified geometric proxies, and the replay remains a simulation-stage evaluation.

## Interpretation

The final evidence supports the conclusion that task-aware morphology optimization must be constrained by reachability, singularity, and task feasibility rather than minimizing length alone. The best design was compact, but it was not the shortest possible morphology in the search space. The selected morphology preserved just over 5 cm of minimum held-out reach margin while maintaining a comfortable Jacobian singularity margin. The full robustness run further indicates that this design is stable under small object, target, and obstacle perturbations across all current object proxies and held-out tasks.

The strongest claim supported by the current evidence is therefore bounded but meaningful: within the Cube MorphTAMP-X v2 MuJoCo benchmark, the selected compact Panda-equivalent morphology achieves lower cost than a nominal full-scale equivalent arm while preserving full held-out task success, robust reach margin, and singularity-safe Panda replay metrics under the tested perturbation distribution. This claim is stronger than a reach-only toy problem because it combines object-specific grasp selection, obstacle/task variation, Panda inverse kinematics, Jacobian singularity checks, and robustness perturbations. It is still a simulation claim and should not be presented as hardware validation.

## Limitations

The morphology variables are equivalent scale factors and base shifts, not a manufacturable redesign of a Franka/Panda robot. The object models are simplified geometric proxies rather than high-fidelity deformable or textured objects. The planner evaluates phase-level pick-and-place behavior with replay-level checks, but it does not yet prove force-closure, tactile feedback, contact-rich grasp retention, actuator torque feasibility, or real-world sim-to-real transfer. The robustness experiment used five perturbation trials per object-task pair and 0.01 m pose/obstacle noise, which is appropriate for a first full robustness evidence set but not a complete uncertainty characterization. Future work should add stronger perturbation sweeps, contact-dynamics grasp validation, torque/energy constraints, and hardware or high-fidelity RoboCasa task validation.

## Reproducibility Pointers

The optimization evidence is stored in `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_singularity_morphology.json`, and its design-analysis report is stored in `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_design_analysis.md`. The full robustness evidence is stored in `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_robustness_full.json`, and the corresponding failure-analysis report is stored in `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_robustness_full_failure_analysis.md`. A compact Git-tracked summary of these final numbers is stored in `evidence/heldout_panda_final_summary.json`.

The key commands are documented in the project README. The most important commands are the held-out Panda-aware morphology optimization with `--base-results-cache` and the full Panda-aware robustness benchmark with `--results-cache`. The cache flags are part of the experimental protocol because the real Panda IK and Jacobian validations are expensive and long-running experiments should be resumable.
