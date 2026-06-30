# Claim Audit

This table maps report-level claims to concrete result fields. It is intentionally conservative: simulation-supported claims are separated from hardware or sim-to-real claims.

| Claim ID | Supported | Evidence path | Observed | Threshold | Caveat |
|---|---:|---|---:|---:|---|
| `optimization_search` | Yes | `optimization.total_candidates` | 375 | 1 | This checks search execution, not global optimality. |
| `feasible_design_found` | Yes | `optimization.feasible_candidates` | 234 | 0 | Feasibility is defined by the implemented simulation constraints. |
| `best_design_selected` | Yes | `optimization.best_design` | "opt_u0.900_f0.900_w0.820_bx0.000" | null | Selection depends on the configured cost and constraint model. |
| `heldout_success` | Yes | `optimization.best_success_rate` | 1.0 | 1.0 | Held-out success is bounded to the encoded object/task protocol. |
| `reach_margin` | Yes | `optimization.best_minimum_reach_margin` | 0.05114549621108844 | 0.05 | Reach margin is an equivalent morphology proxy, not hardware clearance. |
| `singularity_margin` | Yes | `optimization.best_minimum_sigma` | 0.2405079904507164 | 0.08 | The metric is computed on Panda replay states in simulation. |
| `condition_number` | Yes | `optimization.best_maximum_condition_number` | 4.096159158473751 | 30.0 | Condition number is a kinematic quality metric, not a force guarantee. |
| `robustness_runs` | Yes | `robustness.total_runs` | 150 | 150 | The count should be interpreted with the listed perturbation settings. |
| `robustness_success` | Yes | `robustness.success_rate` | 1.0 | 1.0 | This does not imply unbounded robustness outside the perturbation radius. |
| `failure_free_summary` | Yes | `failure_analysis.failed_runs` | 0 | 0 | Only failures recorded by the current evaluator are counted. |
| `hardware_transfer` | No | `unsupported.hardware_transfer` | null | true | No hardware, force-closure, tactile, actuator-torque, or sim-to-real evidence is present; this remains a simulation claim. |

## Safe wording

The current evidence supports a bounded MuJoCo/Panda simulation claim about task-aware equivalent morphology optimization. It does not support claims of hardware validation, force-closure grasping, or physically manufacturable Panda redesign.
