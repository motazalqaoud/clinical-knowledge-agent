# Contributing

Thanks for your interest in improving the Clinical RAG Assistant.

## Setup

```bash
pip install -r requirements-dev.txt
```

## Before opening a PR

```bash
pytest -q --cov=src
ruff check src tests app.py scripts
black --check src tests app.py scripts
```

The test suite runs fully offline — no Hugging Face model downloads are
required to run it (see `tests/conftest.py` for the fake embedder/generator
used in place of real models).

## Code style

- Python 3.11+, type hints on all functions.
- Google-style docstrings with `Args`, `Returns`, and a `Clinical note`
  section where the reasoning behind a design choice isn't obvious from
  the code (see `.github/CODING_GUIDELINES.md` for the full house style).
- Formatted with Black (line length 88); linted with Ruff.
- Never hardcode drug dosages without a physician-confirmation note;
  always preserve safety disclaimers in patient-facing output; prefer
  explicit errors over silent failures.

## Pull requests

Keep PRs focused on one change. Include or update tests for any
behavior change — the chunker's protected-pattern guarantees in
particular should never regress silently. Describe *why* a change is
needed, not just what it does.

This project doesn't have a separate Code of Conduct document; standard
GitHub community norms (be respectful, assume good faith) apply.
