> 历史设计记录：当前四路径 CLI、固定参考面板和统一 lognorm 协议以根目录 README.md 为准；GEARS PKL 已移至 baseline/gears/export.py 转换入口。

# H5AD Independent Control Evaluation Implementation Plan

> **For agentic workers:** Execute inline using executing-plans and test-driven-development. Review the completed change with requesting-code-review.

**Goal:** Evaluate six VCC2026 raw metrics with truth, prediction, real-control H5AD files and an output directory.

**Architecture:** Add a counts H5AD loader separate from the existing GEARS loader. Align all inputs to the prediction gene panel, select matching truth perturbations, and attach only the independently supplied real controls to both sides. Use cell-eval2's counts public API while preserving the existing GEARS lognorm path.

**Tech Stack:** Python 3.11+, AnnData, NumPy/SciPy, cell-eval2 0.18.0, pdex 0.3.0, pytest.

## Task 1: Input contract and counts computation

**Files:** Create `h5ad.py`, `tests/test_h5ad.py`; modify `metrics.py`.

- [x] Write a synthetic three-H5AD fixture: 3 single-gene perturbations, 40 genes, counts matrices, unequal truth/prediction cell counts, reversed gene order in truth, independent controls.
- [x] Write a test calling `load_h5ad(pred_path, truth_path, control_path)` and comparing the resulting matrices with independently constructed expected AnnData objects, then comparing `compute_counts_six` with `aggregate_metrics(compute_metrics(..., config=cfg))` for all six metrics.
- [x] Run `.venv/Scripts/python.exe -m pytest tests/test_h5ad.py::test_h5ad_aligns_genes_selects_truth_and_uses_independent_control -q`; confirm failure because the new loader does not exist.
- [x] Implement the loader and counts metric function with `EvalConfig.from_preset("vcc2026")`, CPU/pdex and an explicit single-gene target map. Preserve gene panel target exclusions and the preset's counts normalization/filtering.
- [x] Test gene order, metadata gene aliases, sparse/dense input, label aliases, control-only unlabeled files, ignoring embedded controls, and incomplete/invalid input. Run the tests until passing.

## Task 2: Four-path command

**Files:** Modify `run.py`, `tests/test_h5ad.py`.

- [x] Add a subprocess test for `python run.py --truth-path truth.h5ad --pred-path pred.h5ad --control-path control.h5ad --output-dir results` from an unrelated working directory; assert six metric keys, counts config, control path, cell counts and both output files.
- [x] Add CLI rejection tests for omitted `--control-path` in H5AD mode and for providing `--control-path` with GEARS PKL input.
- [x] Run these tests and observe the current CLI reject the new argument.
- [x] Extend `evaluate(pred_path, truth_path, output_dir, control_path=None)` and the parser. Require control_path for H5AD and reject it for PKL, so no supplied argument is silently ignored. Keep all entry points working.
- [x] Re-run H5AD and existing GEARS tests.

## Task 3: Documentation and verification

**Files:** Modify `README.md`, `pyproject.toml`, `__init__.py`; review the spec.

- [x] Document the four-path H5AD command first; explain counts `X`, control-only or labeled control inputs, label and gene alignment, context restrictions, raw metrics, output provenance and GEARS compatibility.
- [x] Update package description/version consistently for H5AD support.
- [x] Install with `uv pip install --python .venv/Scripts/python.exe -e . --no-deps` (this virtualenv has no pip); run all tests and installed CLI help.
- [x] Run `git diff --check`; inspect changed files and confirm the other repositories are untouched.
- [x] Request focused review against the user-approved four-parameter requirement, fix actionable findings, and repeat only affected verification.

All file paths above are relative to the repository root. The user subsequently authorized publishing on 2026-10-17: commit the verified changes, fast-forward main, and push origin/main.

## Verification results

- Full suite: 31 passed in 36.03 seconds, including all 8 existing GEARS tests.
- Dense and sparse counts match independently prepared cell-eval2 inputs for all six metrics.
- Installed package: perturbation-eval 0.5.0; CLI and module help expose the independent control input.
- Git whitespace checks passed; Lingshu-Cell remains clean.
- Independent review was requested but could not run because the reviewer hit its usage limit. Completed a direct self-review of data selection, gene alignment, control provenance, counts configuration, CLI compatibility and documentation; no remaining actionable findings.
- Actual user VCC files were not provided; end-to-end scoring used synthetic H5AD files.
