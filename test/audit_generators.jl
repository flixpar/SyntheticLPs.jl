module AuditGeneratorTests

using Test

include(joinpath(@__DIR__, "..", "scripts", "audit_generators.jl"))

function runtests()
    @testset "Audit report solver outcomes" begin
        args = Dict{String, Any}(
            "size-tol" => 0.1,
            "max-build-time" => 30.0,
            "min-presolve-ratio" => 0.6,
            "flagged-only" => true,
        )
        base = Dict{String, Any}(
            "variant" => "blending/robust",
            "status" => "infeasible",
            "target" => 1000,
            "seed" => 13,
            "cols" => 1000,
            "size_ratio" => 1.0,
            "build_time" => 0.1,
        )
        for status in (
            "NOTSET",
            "LOAD_ERROR",
            "MODEL_ERROR",
            "PRESOLVE_ERROR",
            "SOLVE_ERROR",
            "POSTSOLVE_ERROR",
            "UNKNOWN",
            "ITERATION_LIMIT",
            "SOLUTION_LIMIT",
            "INTERRUPT",
            "MEMORY_LIMIT",
            "OBJECTIVE_BOUND",
            "OBJECTIVE_TARGET",
            "STATUS_99",
        )
            summary = variant_summary([merge(base, Dict("solve_status" => status))], args)
            @test summary.errors == 1
            @test summary.flags == ["error"]
            @test isempty(summary.violations)
        end
        for status in
            ("MODEL_EMPTY", "OPTIMAL", "INFEASIBLE", "UNBOUNDED", "UNBOUNDED_OR_INFEASIBLE")
            summary = variant_summary(
                [merge(base, Dict("status" => "unknown", "solve_status" => status))], args
            )
            @test summary.errors == 0
            @test isempty(summary.flags)
        end
        timeout = variant_summary([merge(base, Dict("solve_status" => "TIME_LIMIT"))], args)
        @test timeout.errors == 0
        @test timeout.timeouts == 1
        @test timeout.flags == ["timeout"]
        @test isempty(variant_summary([base], args).flags) # solving disabled
        skipped = variant_summary([merge(base, Dict("skipped" => "nnz limit"))], args)
        @test skipped.errors == 0
        @test skipped.flags == ["skipped"]
        for record in (
            merge(base, Dict("error" => "solver failed", "solve_status" => "UNKNOWN")),
            Dict{String, Any}("target" => 1000, "error" => "build failed"),
        )
            @test variant_summary([record], args).errors == 1
        end
        for (request, status) in (
            ("feasible", "INFEASIBLE"),
            ("feasible", "UNBOUNDED_OR_INFEASIBLE"),
            ("infeasible", "OPTIMAL"),
        )
            summary = variant_summary(
                [merge(base, Dict("status" => request, "solve_status" => status))], args
            )
            @test summary.errors == 0
            @test summary.flags == ["contract"]
        end

        # Exercise JSONL ingestion and --flagged-only rendering: UNKNOWN must
        # survive the filter even when no exception or other flag was recorded.
        mktempdir() do dir
            args["report"] = joinpath(dir, "audit.jsonl")
            args["report-out"] = joinpath(dir, "report.md")
            open(args["report"], "w") do io
                println(io, JSON.json(merge(base, Dict("solve_status" => "UNKNOWN"))))
                println(
                    io,
                    JSON.json(
                        merge(
                            base,
                            Dict("variant" => "healthy/variant", "solve_status" => "INFEASIBLE"),
                        ),
                    ),
                )
            end
            report_main(args)
            report = read(args["report-out"], String)
            @test occursin("2 records, 2 variants, 1 flagged", report)
            @test occursin("`blending/robust`", report)
            @test occursin("| 1 | 0 | **error** |", report)
            @test !occursin("`healthy/variant`", report)
        end
    end
end

end # module
