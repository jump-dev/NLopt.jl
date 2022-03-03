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
    # MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), :LD_MMA)
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), :LD_SLSQP)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            optimal_status = MOI.LOCALLY_SOLVED,
            atol = 1e-2,
            rtol = 1e-2,
            exclude = Any[
                MOI.ConstraintName,
                MOI.VariableName,
                MOI.SolverVersion,
                MOI.ConstraintDual,
                MOI.DualObjectiveValue,
                MOI.NLPBlockDual,
                MOI.ObjectiveBound,

                MOI.VariableBasisStatus,
                MOI.ConstraintBasisStatus,
            ]
        ),
        exclude = String[
            # TODO(odow): investigate failures. A lot of these are probably just
            # suboptimal solutions.
            "test_conic_NormInfinityCone_3",
            "test_conic_NormInfinityCone_INFEASIBLE",
            "test_conic_NormOneCone",
            "test_conic_NormOneCone_INFEASIBLE",
            "test_conic_linear_INFEASIBLE",
            "test_conic_linear_INFEASIBLE_2",
            "test_infeasible_",
            "test_linear_DUAL_INFEASIBLE",
            "test_linear_INFEASIBLE",
            "test_linear_add_constraints",
            "test_linear_VectorAffineFunction_empty_row",
            "test_model_ScalarAffineFunction_ConstraintName",
            "test_model_duplicate_ScalarAffineFunction_ConstraintName",
            "test_nonlinear_invalid",
            "test_objective_ObjectiveFunction_VariableIndex",
            "test_quadratic_SecondOrderCone_basic",
            "test_quadratic_constraint_integration",
            "test_quadratic_nonconvex_constraint_basic",
            "test_quadratic_nonconvex_constraint_integration",
            "test_unbounded_",
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            "test_solve_result_index",
        ]
    )
    return
end

end

TestMOIWrapper.runtests()
