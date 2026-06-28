# Research State

## Project

- Research question: How can a task-aware robot-arm morphology optimizer find lower-cost arm designs while preserving reachability, grasp/place task feasibility, collision safety proxies, and Panda/Franka singularity margins in MuJoCo?
- Primary domain: Robotics simulation, manipulation planning, morphology optimization.
- Research archetype: Empirical robotics.
- Intended contribution: A reproducible Cube/Object-Based MorphTAMP-X benchmark and optimizer that couples object/task constraints with equivalent arm morphology design search.
- Target audience or venue: Robotics research prototype / master's project / possible IEEE robotics-style artifact.
- Communication language: Chinese
- Manuscript language: English

## Current Stage

- Stage: Panda-aware heldout morphology evidence and full robustness evidence generated; cache/resume support implemented.
- Active route: empirical robotics evidence -> Panda-aware optimization -> design/failure/robustness reports.
- Last verified milestone: Windows and WSL full tests passed with 136 tests; Panda-aware heldout optimization completed; 150-run full robustness completed with 150/150 success and failure analysis generated.
- Next decision: Freeze current evidence and write the final result/method chapter, or run optional ablations on sigma threshold, reach margin, and candidate limit.
- Blockers: No physical robot validation; MuJoCo results remain simulation evidence.

## Artifacts

- Code repository and revision: `cube_morphtamp_x_v2_github`, base revision `0e36353` plus local uncommitted cache/optimizer updates.
- Environment or lockfile: WSL conda env `robocasa`; `PYTHONPATH=src:tools`; Panda XML `/home/yzh/robocasa/mujoco_menagerie/franka_emika_panda/scene.xml`.
- Data or benchmark paths: generated object/task definitions in `src/morphtamp_x_v2/`; benchmark protocols in `src/morphtamp_x_v2/protocols.py`.
- Experiment configuration paths: CLI command in `README.md` section "Continuous morphology optimization".
- Raw result paths: `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_singularity_morphology.json`; cache `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_base_cache.json`; robustness smoke `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_robustness_smoke.json`; full robustness cache `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_robustness_full_cache.json`; full robustness output `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_robustness_full.json`.
- Analysis paths: `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_design_analysis.json`; `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_design_analysis.md`; `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_robustness_full_failure_analysis.md`; report draft `docs/results/panda_heldout_final_report.md`; compact evidence summary `evidence/heldout_panda_final_summary.json`.
- Figure paths: README screenshots under `docs/assets/readme/`; result figures intentionally de-emphasized after empty-chart issue.
- Manuscript paths: none yet.

## Claims and Evidence

| Claim | Evidence path or citation | Status | Verification |
|---|---|---|---|
| The optimizer can search continuous equivalent arm morphology scales and base shifts under reachability and singularity constraints. | `src/morphtamp_x_v2/morphology_optimizer.py`; `tests/test_continuous_morphology_optimizer.py` | supported | `python -m pytest -q tests` -> 131 passed |
| Panda-aware heldout search found a feasible lower-cost design `opt_u0.900_f0.900_w0.820_bx0.000`. | `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_singularity_morphology.json` | supported | completed CLI run; feasible `234/375`; best design recorded |
| The recommended design satisfies the configured singularity constraints. | Same result JSON; design report Markdown | supported | best min sigma `0.2405079904507164`; max condition number `4.096159158473751`; constraints were `minimum_sigma=0.08`, `maximum_condition_number=30` |
| The design has at least 5 cm heldout reach margin under current proxy demand model. | Same result JSON | supported | best minimum reach margin `0.05114549621108844` |
| Full Panda-aware robustness passes under 1 cm pose/obstacle perturbations for all object types and heldout tasks. | `/home/yzh/robocasa/cube_morphtamp_x_v2/results/heldout_panda_robustness_full.json` | supported | `150/150` successful; max position error `0.0244885693312509`; mean position error `0.0011229728864130936`; min sigma `0.23390447283229004`; max condition `4.211812901125048`; failure analysis reports `0` failures |
| Simulation evidence implies real-world manipulation success. | none | retired | not supported; report must state simulation-stage limitation |

Status values: `planned`, `partial`, `supported`, `contradicted`, `retired`.

## Engineering Quality

- Tests: Windows `python -m pytest -q tests` -> 136 passed; WSL `python -m pytest -q tests` -> 136 passed.
- Reproducibility smoke test: Panda-aware heldout optimization is resumable via `--base-results-cache`.
- Code review: completed for robustness summary/cache behavior, Panda perturbed-scenario evaluation, README/report sync notes, and scenario override cleanup.
- Security review, if applicable: not applicable.
- Performance review, if applicable: cache/resume added because Panda base evaluations are expensive.

## Research Quality

- Baselines or comparison class: Pareto design set and preset morphology benchmark exist; final baseline comparison should include nominal Panda-style design.
- Seeds, repetitions, or sample rationale: current heldout result is deterministic protocol-level evidence; full robustness used seed 42, 5 perturbation trials, 10 object types, and 3 heldout tasks.
- Statistical analysis: not yet required for deterministic grid; robustness summary should report dispersion.
- Ablations or sensitivity analysis: planned for reach margin, sigma threshold, path-cost weight, and candidate limit.
- Failure cases: planned via `failure-analysis`.
- Limitations and threats to validity: equivalent morphology cost, simulation-only evidence, simplified object/contact grasp model, no physical hardware validation.

## Publication Checks

- Citation audit: pending.
- Claim audit: pending.
- Independent research review: pending.
- Venue-aware peer review: pending.
- PDF compilation and visual inspection: not started.

## Decision Log

| Date | Decision | Evidence or rationale | Consequence |
|---|---|---|---|
| 2026-06-28 | Add incremental base-result cache to `optimize-morphology`. | Panda-aware heldout runs exceeded 10 minutes before finishing; base evaluation is expensive but reusable. | Long Panda optimization can resume after timeout and sweep different design grids without recomputing completed base cases. |
| 2026-06-28 | Use Panda-aware heldout result as the next primary design evidence. | Completed run found `234/375` feasible candidates and recommended `opt_u0.900_f0.900_w0.820_bx0.000`. | Next stage should focus on robustness/failure analysis and final report claims. |
| 2026-06-28 | Fix Panda robustness to evaluate perturbed scenarios, not nominal scenarios with replaced metadata. | Regression test `test_panda_robustness_evaluates_perturbed_scenario` reproduces the bug and passes after fix. | Robustness results now correspond to the actual perturbed scene used for Panda validation. |
| 2026-06-28 | Run a representative Panda robustness smoke before scaling. | Full robustness with real Panda is expensive; 18-run smoke completed successfully. | Current robustness claim should be labelled `partial`, not full distributional robustness. |
| 2026-06-29 | Add incremental cache/resume for robustness and launch full Panda robustness. | 150 real Panda validations are too expensive to risk without checkpointing. | Background run PID `5955`; cache path `results/heldout_panda_robustness_full_cache.json`; output path `results/heldout_panda_robustness_full.json`. |
| 2026-06-29 | Complete full Panda robustness and failure analysis. | Completed `150/150` runs with zero failures across 10 objects and 3 heldout tasks. | Robustness claim upgraded from `partial` to `supported`; next step is final report or ablation package. |
| 2026-06-29 | Draft final Panda-aware heldout report chapter. | Report numbers cross-checked against `evidence/heldout_panda_final_summary.json`; tests passed. | `docs/results/panda_heldout_final_report.md` is ready for review and can seed a manuscript/results chapter. |
| 2026-06-29 | Strengthen robustness benchmark outputs and cache semantics. | `robustness-benchmark` now reports by-object/by-task success, error, sigma, condition, path, and failed-run summaries; cache reuse now survives increasing trial count. | Result JSONs are more self-contained for reports, and expensive robustness runs can be extended without recomputing completed trials. |
| 2026-06-29 | Align failure analysis with Panda joint-replay success criteria. | Regression test added for rows where static task success is true but `joint_metrics.success` is false. | `failure-analysis` now reports `joint:*` failures instead of silently treating Panda IK/collision failures as successes. |
