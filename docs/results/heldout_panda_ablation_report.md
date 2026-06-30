# Ablation and Failure-Boundary Report

This report summarizes which method components are supported by the current evidence.

## Optimized design

- Design: `opt_u0.900_f0.900_w0.820_bx0.000`
- Success rate: 1.0
- Morphology cost: 0.75712

## Component evidence

| Component | Active | Key evidence |
|---|---:|---|
| `reach_margin_constraint` | True | `{"active": true, "evidence": 0.05114549621108844, "threshold": 0.05}` |
| `singularity_constraint` | True | `{"active": true, "max_condition_number": 4.096159158473751, "min_sigma": 0.2405079904507164, "thresholds": {"maximum_condition_number": 30.0, "minimum_sigma": 0.08}}` |
| `multi_grasp_selection` | True | `{"active": true, "mean_coarse_candidates": 2.4, "mean_full_candidates": 2.0, "rows_with_grasp_data": 110}` |

## Baseline and failure boundary

| Design | Success rate | Morphology cost | Min reach margin |
|---|---:|---:|---:|
| `compact_base_shift` | 1.0 | 0.80503 | 0.09155549621108838 |
| `high_reach_arm` | 1.0 | 0.9484300000000001 | 0.24245549621108853 |
| `long_forearm` | 1.0 | 0.91688 | 0.21090549621108834 |
| `long_wrist` | 1.0 | 0.90642 | 0.20044549621108843 |
| `nominal_panda` | 1.0 | 0.86 | 0.1540254962110884 |
| `short_arm` | 0.8454545454545455 | 0.7220799999999999 | 0.016105496211088366 |

A useful morphology benchmark should expose a failure boundary: a too-short design should become cheaper but fail reach or stress constraints. If no baseline fails, the benchmark may be too easy for a strong optimization claim.
