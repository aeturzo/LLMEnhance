# Phase B test report

Run date: 2026-07-20

Command: `pytest -q`

Result: **20 passed, 3 failed**.

The frozen-result integrity validator passed independently with zero issues; see
`tables/VALIDATION.md` and `tables/validation.json`.

## Pre-existing repository test failures

- `tests/test_ab_memory.py::test_ab_memory_verbose` expects the removed
  `search_service.documents` module attribute.
- `tests/test_full_pipeline.py::test_full_pipeline` expects `doc_id` in a fixture
  whose current rows use a different schema.
- `tests/test_memory_ab.py::test_memory_beats_baseline` expects legacy
  `session`/`content`/`query`/`relevant` fixture fields.

These failures occur outside the frozen evaluation path. No Phase B result or
source artifact was modified to make a legacy test pass.
