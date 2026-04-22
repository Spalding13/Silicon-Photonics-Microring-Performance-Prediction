# Changelog

## 2026-04-21

### Simplification pass

- Rewrote `README.md` to match the actual code and explain the project in simpler terms.
- Reduced scaffolding in `src/models.py` to the pieces the project really uses.
- Cleaned `src/physics.py` and `src/generator.py` so the core ideas are easier to follow.
- Removed unused imports and unused helper functions from `src/utils.py`.
- Updated the notebook to use fewer imports, clearer labels, and correct summary statistics.
- Kept the leakage protections and group-aware evaluation logic.

### Validation

- `.venv\Scripts\python.exe -m pytest -q`
- `.venv\Scripts\python.exe -m compileall src tests`
