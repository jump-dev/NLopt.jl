module TestMOIWrapper

using MathOptInterface
using NLopt
using Test

const MOI = MathOptInterface

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_SolverName()
    @test MOI.get(NLopt.Optimizer(), MOI.SolverName()) == "NLopt"
    return
end

function test_supports_incremental_interface()
    @test MOI.supports_incremental_interface(NLopt.Optimizer())
    return
end

function test_runtests()
    model = MOI.instantiate(NLopt.Optimizer; with_bridge_type = Float64)
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), :LD_SLSQP)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            optimal_status = MOI.LOCALLY_SOLVED,
            exclude = Any[
                MOI.ConstraintName,
                MOI.VariableName,
                MOI.SolverVersion,
                MOI.ConstraintDual,
                MOI.DualObjectiveValue,
            ]
        ),
        exclude = String[
            "test_conic_NormInfinityCone_3", # OTHER_ERROR
            "test_conic_NormOneCone",
            "test_conic_NormInfinityCone_INFEASIBLE",
            "test_conic_linear_INFEASIBLE",
        ]
    )
    return
end

end

TestMOIWrapper.runtests()
