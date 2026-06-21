<!-- status: active · updated: YYYY-MM-DD · class: disposable -->

# SPRINT-000 — <slug>

- **base:** main
- **DoD:** <behavioral acceptance criteria — what proves this is done>

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set: the files the agent should load for this sprint, nothing else -->
- path/to/read_one.py
- docs/decisions/ADR-00X-relevant.md

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- path/to/changed.py
- path/to/migrations/0*.py
- path/to/test_feature.py

## contracts
<!-- type: target [ | when: <glob> ]
       test = run in the verify loop (advisory reminder)
       snap = a snapshot file that MUST also change if a `when` file changed
       map  = a pinned map/guard that MUST also change if a `when` file changed -->
- test: path/to/test_feature.py::parity
- snap: docs/openapi.json | when: **/api.py
- map: config/tests/test_gates.py | when: **/api.py

## docs
<!-- must be touched when this lands (the docs gate enforces these appear in the diff) -->
- docs/DEVLOG.md
- docs/CHECKLIST.md
