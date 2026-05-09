SHELL := /usr/bin/env bash

.PHONY: smoke audit inventory

smoke:
	bash tools/smoke_test.sh

audit:
	python tools/audit_hardcoded_paths.py --legacy-mode report

inventory:
	python tools/make_inventory.py
