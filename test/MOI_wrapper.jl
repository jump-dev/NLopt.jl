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

function test_raw_optimizer_attribute_in_optimize()
    model = NLopt.Optimizer()
    x = MOI.add_variables(model, 2)
    f = (x[1] - 2.0) * (x[1] - 2.0) + (x[2] + 1.0)^2# * (x[2] + 1)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    for (k, v) in (
        "algorithm" => :LD_SLSQP,
        "stopval" => 1.0,
        "ftol_rel" => 1e-6,
        "ftol_abs" => 1e-6,
        "xtol_rel" => 1e-6,
        "xtol_abs" => 1e-6,
        "maxeval" => 100,
        "maxtime" => 60.0,
        "initial_step" => [0.1, 0.1],
        "population" => 10,
        "seed" => 1234,
        "vector_storage" => 3,
    )
        attr = MOI.RawOptimizerAttribute(k)
        MOI.set(model, attr, v)
    end
    MOI.optimize!(model)
    @test ≈(MOI.get.(model, MOI.VariablePrimal(), x), [2.0, -1.0]; atol = 1e-4)
    return
end

function test_local_optimizer_Symbol()
    model = NLopt.Optimizer()
    x = MOI.add_variables(model, 2)
    f = (x[1] - 2.0) * (x[1] - 2.0) + (x[2] + 1.0) * (x[2] + 1.0)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), :AUGLAG)
    attr = MOI.RawOptimizerAttribute("local_optimizer")
    @test MOI.get(model, attr) === nothing
    MOI.set(model, attr, :LD_SLSQP)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) isa MOI.TerminationStatusCode
    return
end

function test_local_optimizer_Opt()
    model = NLopt.Optimizer()
    x = MOI.add_variables(model, 2)
    f = (x[1] - 2.0) * (x[1] - 2.0) + (x[2] + 1.0) * (x[2] + 1.0)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), :GD_MLSL)
    attr = MOI.RawOptimizerAttribute("local_optimizer")
    @test MOI.get(model, attr) === nothing
    MOI.set(model, attr, NLopt.Opt(:LD_MMA, 2))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) isa MOI.TerminationStatusCode
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

function test_ScalarNonlinearFunction_mix_apis_nlpblock_last()
    model = NLopt.Optimizer()
    x = MOI.add_variable(model)
    f = MOI.ScalarNonlinearFunction(:log, Any[x])
    MOI.add_constraint(model, f, MOI.LessThan(1.0))
    evaluator = MOI.Test.HS071(false, false)
    bounds = MOI.NLPBoundsPair.([25.0, 40.0], [Inf, 40.0])
    block = MOI.NLPBlockData(bounds, evaluator, true)
    @test_throws(
        ErrorException("Cannot mix the new and legacy nonlinear APIs"),
        MOI.set(model, MOI.NLPBlock(), block),
    )
    return
end

function test_ScalarNonlinearFunction_mix_apis_nlpblock_first()
    model = NLopt.Optimizer()
    x = MOI.add_variable(model)
    evaluator = MOI.Test.HS071(false, false)
    bounds = MOI.NLPBoundsPair.([25.0, 40.0], [Inf, 40.0])
    block = MOI.NLPBlockData(bounds, evaluator, true)
    MOI.set(model, MOI.NLPBlock(), block)
    f = MOI.ScalarNonlinearFunction(:log, Any[x])
    @test_throws(
        ErrorException("Cannot mix the new and legacy nonlinear APIs"),
        MOI.add_constraint(model, f, MOI.LessThan(1.0)),
    )
    return
end

function test_ScalarNonlinearFunction_is_valid()
    model = NLopt.Optimizer()
    x = MOI.add_variable(model)
    F, S = MOI.ScalarNonlinearFunction, MOI.EqualTo{Float64}
    @test MOI.is_valid(model, MOI.ConstraintIndex{F,S}(1)) == false
    f = MOI.ScalarNonlinearFunction(:sin, Any[x])
    c = MOI.add_constraint(model, f, MOI.EqualTo(0.0))
    @test c isa MOI.ConstraintIndex{F,S}
    @test MOI.is_valid(model, c) == true
    return
end

function test_ScalarNonlinearFunction_ObjectiveFunctionType()
    model = NLopt.Optimizer()
    x = MOI.add_variable(model)
    f = MOI.ScalarNonlinearFunction(:log, Any[x])
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    F = MOI.ScalarNonlinearFunction
    MOI.set(model, MOI.ObjectiveFunction{F}(), f)
    @test MOI.get(model, MOI.ObjectiveFunctionType()) == F
    return
end

function test_AutomaticDifferentiationBackend()
    model = NLopt.Optimizer()
    attr = MOI.AutomaticDifferentiationBackend()
    @test MOI.supports(model, attr)
    @test MOI.get(model, attr) == MOI.Nonlinear.SparseReverseMode()
    MOI.set(model, attr, MOI.Nonlinear.ExprGraphOnly())
    @test MOI.get(model, attr) == MOI.Nonlinear.ExprGraphOnly()
    return
end

function test_ScalarNonlinearFunction_LessThan()
    model = NLopt.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), :LD_SLSQP)
    x = MOI.add_variable(model)
    # Needed for NLopt#31
    MOI.set(model, MOI.VariablePrimalStart(), x, 1.0)
    f = MOI.ScalarNonlinearFunction(:log, Any[x])
    MOI.add_constraint(model, f, MOI.LessThan(2.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    MOI.optimize!(model)
    @test isapprox(MOI.get(model, MOI.VariablePrimal(), x), exp(2); atol = 1e-4)
    return
end

function test_ScalarNonlinearFunction_GreaterThan()
    model = NLopt.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), :LD_SLSQP)
    x = MOI.add_variable(model)
    # Needed for NLopt#31
    MOI.set(model, MOI.VariablePrimalStart(), x, 1.0)
    f = MOI.ScalarNonlinearFunction(:log, Any[x])
    MOI.add_constraint(model, f, MOI.GreaterThan(2.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    MOI.optimize!(model)
    @test isapprox(MOI.get(model, MOI.VariablePrimal(), x), exp(2); atol = 1e-4)
    return
end

function test_ScalarNonlinearFunction_Interval()
    model = NLopt.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), :LD_SLSQP)
    x = MOI.add_variable(model)
    # Needed for NLopt#31
    MOI.set(model, MOI.VariablePrimalStart(), x, 1.0)
    f = MOI.ScalarNonlinearFunction(:log, Any[x])
    MOI.add_constraint(model, f, MOI.Interval(1.0, 2.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    MOI.optimize!(model)
    @test isapprox(MOI.get(model, MOI.VariablePrimal(), x), exp(2); atol = 1e-4)
    return
end

function test_ScalarNonlinearFunction_derivative_free()
    model = NLopt.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), :LN_COBYLA)
    x = MOI.add_variable(model)
    # Needed for NLopt#31
    MOI.set(model, MOI.VariablePrimalStart(), x, 1.0)
    f = MOI.ScalarNonlinearFunction(:log, Any[x])
    MOI.add_constraint(model, f, MOI.GreaterThan(2.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    MOI.optimize!(model)
    @test isapprox(MOI.get(model, MOI.VariablePrimal(), x), exp(2); atol = 1e-4)
    return
end

end  # module

TestMOIWrapper.runtests()
