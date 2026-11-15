<p align="center"><img src=".github/assets/repository-cover.jpg" alt="CEDAR" width="100%"></p>

# CEDAR

**Core model implementations, experiment scripts and recorded result summaries.**

## Repository map

- [`results_v3/`](results_v3/) — experiment code and selected artifacts
- [`results/`](results/) — experiment code and selected artifacts
- [`scripts/`](scripts/) — experiment code and selected artifacts
- [`results_e2/`](results_e2/) — experiment code and selected artifacts
- [`src/`](src/) — core implementation

## Use

Inspect the relevant script’s module docstring and argument parser before running it. Experiments may require project-specific datasets, model weights and hardware. Recreate those dependencies locally; this release does not bundle them.

The model implementation starts in [`src/cedar_model.py`](src/cedar_model.py) and [`src/cedar_v2_model.py`](src/cedar_v2_model.py); experiment orchestration is in [`scripts/`](scripts/).

## Scope and status

This is a research-code snapshot. Model weights, raw datasets, private operational notes and manuscript files are not included. Recorded results are provided as artifacts, not a claim of independent reproduction or production readiness.

Project artwork was generated with AI for repository presentation.
