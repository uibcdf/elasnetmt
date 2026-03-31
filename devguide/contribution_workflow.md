# Contribution Workflow

All contributions to **ElasNetMT** must follow this standardized path to ensure stability and architectural integrity.

## Step 1: Branch and Smoke Test
- Create a branch from `master` or `develop`.
- Before any significant change, run:
  ```bash
  bash devtools/tests/run_tiers.sh smoke
  ```

## Step 2: Implementation Checklist
- [ ] Use `_private/` modules for shared logic.
- [ ] Use `pyunitwizard` for physical quantities.
- [ ] Use `arg_digest` for public methods.
- [ ] Follow `api_conventions.md` (naming and behavior).
- [ ] Implement at least one test in `tests/smoke/`.

## Step 3: Verification
- Run all test tiers:
  ```bash
  bash devtools/tests/run_tiers.sh all
  ```
- Ensure docstrings follow the `documentation_standards.md`.

## Step 4: Pull Request
- Submit a PR with a clear description of the **problem** and the **solution**.
- Use the provided PR template.
- Wait for a maintainer review (typically within 48 hours).

---

## Technical Maintenance
If you add a new third-party library, update:
- `pyproject.toml`
- `elasnetmt/_depdigest.py`
- `devtools/requirements.yaml`
