SHELL := /usr/bin/env bash

KAGGLE_USER ?= yehiasamir
KERNEL_SLUG ?= isic2018-xai-evaluation
KERNEL_ID ?= $(KAGGLE_USER)/$(KERNEL_SLUG)

.PHONY: help setup run push status wait pull log-error compare

help:
	@echo "Targets:"
	@echo "  make setup      # create scripts/pipeline.env from example"
	@echo "  make run        # full cycle: push, wait, pull"
	@echo "  make push       # push notebook to Kaggle"
	@echo "  make status     # kaggle kernel status"
	@echo "  make wait       # wait until run completes"
	@echo "  make pull       # pull outputs to outputs/<timestamp>/"
	@echo "  make log-error  # write errors/latest_error.txt"
	@echo "  make compare    # compare output folders"

setup:
	@test -f scripts/pipeline.env || cp scripts/pipeline.env.example scripts/pipeline.env
	@echo "scripts/pipeline.env is ready"

run:
	bash scripts/run.sh

push:
	bash scripts/push_to_kaggle.sh

status:
	kaggle kernels status "$(KERNEL_ID)"

wait:
	bash scripts/wait_for_kernel.sh "$(KERNEL_ID)"

pull:
	@ts=$$(date +%Y-%m-%d_%H-%M); \
	mkdir -p "outputs/$$ts"; \
	kaggle kernels output "$(KERNEL_ID)" -p "outputs/$$ts" --quiet; \
	echo "Saved to outputs/$$ts"

log-error:
	bash scripts/log_error.sh "$(KERNEL_ID)"

compare:
	bash scripts/compare.sh
