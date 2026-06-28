# Reproducibility Notes

This repository is a clean, runnable release of `cube_morphtamp_x_v2`.

## What is included

- `src/morphtamp_x_v2/`: project source code.
- `tests/`: unit and CLI tests.
- `tools/`: README generation helper.
- `docs/results/final_claims.md`: report-ready result claims and boundaries.
- `docs/results/final_tables.md`: report-ready result tables.
- `evidence/`: compact JSON evidence files for the accepted benchmark summaries.

## What is intentionally excluded

The large local `results/` directory is excluded from Git because it contains replay files, debug scenes, temporary visualizations, and machine-specific artifacts. Recreate fresh outputs with the CLI commands documented in `README.md`.

## External dependency

For Panda/Franka MuJoCo validation, pass a local MuJoCo XML path, for example:

```bash
--panda-xml ~/robocasa/mujoco_menagerie/franka_emika_panda/scene.xml
```

This path is intentionally not hard-coded into the repository because it is machine-specific.
