#!/usr/bin/env julia

# This script checks if all problem generators consistently use the new standardized interface

using Pkg
Pkg.activate(".")

# Function to check a problem type file
function check_problem_file(file_path)
    problem_name = splitext(basename(file_path))[1]
    println("Checking $problem_name problem...")
    
    content = read(file_path, String)
    
    # Check using direct string searches instead of complex regex
    
    # Check for generator function
    if occursin("function generate_", content)
        println("  ✓ Has generate function")
    else
        println("  ❌ Missing generate function")
    end
    
    # Check for sampler function
    if occursin("function sample_", content)
        println("  ✓ Has sample function")
    else
        println("  ❌ Missing sample function")
    end
    
    # Check for register_problem call
    if occursin("register_problem", content)
        println("  ✓ Has register_problem call")
    else
        println("  ❌ Missing register_problem call")
    end
    
    println()
end

# List all problem type files
problem_dir = joinpath(@__DIR__, "src", "problem_types")
problem_files = filter(file -> endswith(file, ".jl") && 
                             !startswith(basename(file), "template") && 
                             !startswith(basename(file), "lib/"), 
                      readdir(problem_dir, join=true))

# Check each file
for file in sort(problem_files)
    check_problem_file(file)
end

println("Verification complete.")