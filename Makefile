JULIA ?= julia
RUFF ?= ruff

.PHONY: setup format format-check lint lint-julia lint-python test check

setup:
	$(JULIA) --startup-file=no --project=quality -e 'using Pkg; Pkg.instantiate()'

format:
	$(JULIA) --startup-file=no --project=quality -e 'using JuliaFormatter; format(".")'
	$(RUFF) format .

format-check:
	$(JULIA) --startup-file=no --project=quality -e 'using JuliaFormatter; format(".", overwrite=false) || exit(1)'
	$(RUFF) format --check .

lint-julia:
	$(JULIA) --startup-file=no --project=quality quality/quality.jl

lint-python:
	$(RUFF) check .

lint: format-check lint-julia lint-python

test:
	$(JULIA) --startup-file=no --project=. -e 'using Pkg; Pkg.test()'

check: lint test
