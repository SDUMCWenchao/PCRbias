SHELL := /usr/bin/env bash

.PHONY: smoke audit consistency synthetic-smoke test inventory ci

smoke:
	bash tools/smoke_test.sh

audit:
	python tools/audit_hardcoded_paths.py --legacy-mode report

consistency:
	python tools/check_repository_consistency.py

synthetic-smoke:
	python tools/run_synthetic_smoke.py

test:
	python -m pytest

inventory:
	python tools/make_inventory.py

ci: smoke consistency audit synthetic-smoke test
