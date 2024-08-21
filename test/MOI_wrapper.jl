# Copyright (c) 2013: Steven G. Johnson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestMOIWrapper

using NLopt
using Test
import MathOptInterface as MOI

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

function test_runtests()
    model = MOI.instantiate(
        NLopt.Optimizer;
        with_bridge_type = Float64,
        with_cache_type = Float64,
    )
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), :LD_SLSQP)
    MOI.set(model, MOI.RawOptimizerAttribute("maxtime"), 10.0)
    other_failures = Any[]
    if Sys.WORD_SIZE == 32
        push!(other_failures, r"^test_constraint_qcp_duplicate_diagonal$")
    end
    MOI.Test.runtests(
        model,
        MOI.Test.Config(;
            optimal_status = MOI.LOCALLY_SOLVED,
            atol = 1e-2,
            rtol = 1e-2,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.ConstraintDual,
                MOI.DualObjectiveValue,
                MOI.ObjectiveBound,
                MOI.NLPBlockDual,
                MOI.VariableBasisStatus,
            ],
        );
        exclude = [
            # Issues related to detecting infeasibility
            r"^test_conic_NormInfinityCone_INFEASIBLE$",
            r"^test_conic_NormOneCone_INFEASIBLE$",
            r"^test_conic_linear_INFEASIBLE$",
            r"^test_conic_linear_INFEASIBLE_2$",
            r"^test_infeasible_MIN_SENSE$",
            r"^test_infeasible_MIN_SENSE_offset$",
            r"^test_linear_DUAL_INFEASIBLE$",
            r"^test_linear_DUAL_INFEASIBLE_2$",
            r"^test_linear_INFEASIBLE$",
            r"^test_linear_INFEASIBLE_2$",
            r"^test_solve_TerminationStatus_DUAL_INFEASIBLE$",
            # ArgumentError: invalid NLopt arguments: too many equality constraints
            r"^test_linear_VectorAffineFunction_empty_row$",
            # Evaluated: MathOptInterface.ALMOST_LOCALLY_SOLVED == MathOptInterface.LOCALLY_SOLVED
            r"^test_linear_add_constraints$",
            # NLopt#31
            r"^test_nonlinear_invalid$",
            # TODO(odow): wrong solutions?
            r"^test_quadratic_SecondOrderCone_basic$",
            r"^test_quadratic_constraint_integration$",
            # Perhaps an expected failure because the problem is non-convex
            r"^test_quadratic_nonconvex_constraint_basic$",
            r"^test_quadratic_nonconvex_constraint_integration$",
            # A whole bunch of issues to diagnose here
            "test_basic_VectorNonlinearFunction_",
            # INVALID_OPTION?
            r"^test_nonlinear_expression_hs109$",
            other_failures...,
        ],
    )
    return
end

function test_list_of_model_attributes_set()
    attr = MOI.ListOfModelAttributesSet()
    model = NLopt.Optimizer()
    ret = MOI.AbstractModelAttribute[]
    @test MOI.get(model, attr) == ret
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    push!(ret, MOI.ObjectiveSense())
    @test MOI.get(model, attr) == ret
    x = MOI.add_variable(model)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    push!(ret, MOI.ObjectiveFunction{MOI.VariableIndex}())
    @test MOI.get(model, attr) == ret
    return
end

function test_list_and_number_of_constraints()
    model = NLopt.Optimizer()
    x = MOI.add_variable(model)
    F1, S1 = MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}
    F2, S2 = MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64}
    @test MOI.get(model, MOI.NumberOfConstraints{F1,S1}()) == 0
    @test MOI.get(model, MOI.NumberOfConstraints{F2,S2}()) == 0
    @test MOI.get(model, MOI.ListOfConstraintIndices{F1,S1}()) == []
    @test MOI.get(model, MOI.ListOfConstraintIndices{F2,S2}()) == []
    c1 = MOI.add_constraint(model, 1.0 * x, MOI.EqualTo(2.0))
    @test MOI.get(model, MOI.NumberOfConstraints{F1,S1}()) == 1
    @test MOI.get(model, MOI.NumberOfConstraints{F2,S2}()) == 0
    @test MOI.get(model, MOI.ListOfConstraintIndices{F1,S1}()) == [c1]
    @test MOI.get(model, MOI.ListOfConstraintIndices{F2,S2}()) == []
    c2 = MOI.add_constraint(model, 1.0 * x * x, MOI.LessThan(2.0))
    @test MOI.get(model, MOI.NumberOfConstraints{F1,S1}()) == 1
    @test MOI.get(model, MOI.NumberOfConstraints{F2,S2}()) == 1
    @test MOI.get(model, MOI.ListOfConstraintIndices{F1,S1}()) == [c1]
    @test MOI.get(model, MOI.ListOfConstraintIndices{F2,S2}()) == [c2]
    @test MOI.get(model, MOI.ConstraintSet(), c1) == MOI.EqualTo(2.0)
    @test MOI.get(model, MOI.ConstraintSet(), c2) == MOI.LessThan(2.0)
    return
end

function test_raw_optimizer_attribute()
    model = NLopt.Optimizer()
    attr = MOI.RawOptimizerAttribute("algorithm")
    @test MOI.supports(model, attr)
    @test MOI.get(model, attr) == :none
    MOI.set(model, attr, :LD_MMA)
    @test MOI.get(model, attr) == :LD_MMA
    bad_attr = MOI.RawOptimizerAttribute("foobar")
    @test !MOI.supports(model, bad_attr)
    @test_throws MOI.GetAttributeNotAllowed MOI.get(model, bad_attr)
    return
end

function test_list_of_variable_attributes_set()
    model = NLopt.Optimizer()
    @test MOI.get(model, MOI.ListOfVariableAttributesSet()) ==
          MOI.AbstractVariableAttribute[]
    x = MOI.add_variables(model, 2)
    MOI.supports(model, MOI.VariablePrimalStart(), MOI.VariableIndex)
    MOI.set(model, MOI.VariablePrimalStart(), x[2], 1.0)
    @test MOI.get(model, MOI.ListOfVariableAttributesSet()) ==
          MOI.AbstractVariableAttribute[MOI.VariablePrimalStart()]
    @test MOI.get(model, MOI.VariablePrimalStart(), x[1]) === nothing
    @test MOI.get(model, MOI.VariablePrimalStart(), x[2]) === 1.0
    return
end

function test_list_of_constraint_attributes_set()
    model = NLopt.Optimizer()
    F, S = MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}
    @test MOI.get(model, MOI.ListOfConstraintAttributesSet{F,S}()) ==
          MOI.AbstractConstraintAttribute[]
    return
end

function test_get_objective_function()
    model = NLopt.Optimizer()
    x = MOI.add_variable(model)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    @test MOI.get(model, MOI.ObjectiveFunction{MOI.VariableIndex}()) == x
    F = MOI.ScalarAffineFunction{Float64}
    @test isapprox(MOI.get(model, MOI.ObjectiveFunction{F}()), 1.0 * x)
    return
end

end  # module

TestMOIWrapper.runtests()
