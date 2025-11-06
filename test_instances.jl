#!/usr/bin/env julia

"""
Convenient wrapper for running comprehensive problem instance tests.

Usage:
    julia --project=@. test_instances.jl [options] [problem_types...]

Options:
    --verbose, -v     Show detailed output for each test
    --help, -h        Show this help message

Examples:
    julia --project=@. test_instances.jl
        Test all problem types

    julia --project=@. test_instances.jl transportation
        Test only transportation problem type

    julia --project=@. test_instances.jl transportation knapsack portfolio
        Test multiple specific problem types

    julia --project=@. test_instances.jl --verbose
        Test all problem types with verbose output

    julia --project=@. test_instances.jl -v transportation
        Test transportation with verbose output
"""

if "--help" in ARGS || "-h" in ARGS
    println(__doc__)
    exit(0)
end

# Include the actual test file
include("test/test_problem_instances.jl")
