JULIA ?= julia
RUFF ?= ruff

.PHONY: setup format format-check lint lint-julia lint-python test check

# `resolve` before `instantiate`: the quality environment dev-depends on the
# package, so its manifest goes stale whenever the package gains a dependency,
# and `instantiate` alone will not pick that up.
setup:
	$(JULIA) --startup-file=no --project=quality -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

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

# -O1: the suite is compilation-bound, so this cuts wall clock ~23% while still
# running every assertion. Matches CI.
# Focus on one or more categories while iterating: make test CATEGORIES=tsp,knapsack
CATEGORIES ?=

test:
	$(JULIA) --startup-file=no --project=. -e 'using Pkg; Pkg.test(; julia_args=["-O1"], test_args=ARGS)' $(CATEGORIES)

check: lint test
