using Test

using MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

using NLopt
const solver = MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LD_SLSQP)

optimizer = MOI.instantiate(solver)

const config = MOIT.TestConfig(atol=1e-2, rtol=1e-2, duals=false,
                               optimal_status=MOI.LOCALLY_SOLVED)

@testset "SolverName" begin
    @test MOI.get(optimizer, MOI.SolverName()) == "NLopt"
end

@testset "supports_default_copy_to" begin
    @test MOIU.supports_default_copy_to(optimizer, false)
    @test !MOIU.supports_default_copy_to(optimizer, true)
end

function test_nlp(solver)
    optimizer = MOI.instantiate(solver)
    MOIT.nlptest(optimizer, config)
end
@testset "Non-Linear tests" begin
    test_nlp(solver)
    test_nlp(MOI.OptimizerWithAttributes(
        NLopt.Optimizer,
        "algorithm" => :AUGLAG,
        "local_optimizer" => :LD_LBFGS,
    ))
    # NLP tests have different number of variables so we
    # cannot run through them all with the same `local_optimizer`.
    # Let's just do hs071.
    local_optimizer = Opt(:LD_LBFGS, 4)
    opt.xtol_rel = 1e-6
    MOIT.hs071_test(MOI.instantiate(MOI.OptimizerWithAttributes(
        NLopt.Optimizer,
        "algorithm" => :AUGLAG,
        "local_optimizer" => local_optimizer,
    )), config)
end

@testset "Testing getters" begin
    MOI.Test.copytest(MOI.instantiate(solver, with_bridge_type=Float64), MOIU.Model{Float64}())
end

@testset "Bounds set twice" begin
    MOI.Test.set_lower_bound_twice(optimizer, Float64)
    MOI.Test.set_upper_bound_twice(optimizer, Float64)
end

MOI.empty!(optimizer)
bridged = MOIB.full_bridge_optimizer(MOIU.CachingOptimizer(MOIU.UniversalFallback(MOIU.Model{Float64}()), optimizer), Float64)

@testset "Continuous Linear tests" begin
    MOIT.contlineartest(bridged, config, [
        # Infeasibility and unboundedness not detected by NLopt
        "linear8a", "linear8b", "linear8c", "linear12",
        # Terminates with `:ROUNDOFF_LIMITED` due to the bad scaling of input problem
        "linear9",
        # FIXME invalid NLopt arguments: too many equality constraints
        "linear15"])
end

@testset "Continuous Quadratic tests" begin
    MOIT.qptest(bridged, config)
end

MOIB.add_bridge(bridged, MOIB.Constraint.SOCtoNonConvexQuadBridge{Float64})

@testset "Continuous Conic tests" begin
    MOIT.lintest(bridged, config, [
        # Infeasibility and unboundedness not detected by NLopt
        "lin3", "lin4"
    ])
    MOIT.soctest(bridged, config, [
        # Infeasibility and unboundedness not detected by NLopt
        "soc3"
    ])
    MOIT.rsoctest(bridged, config, [
        # Infeasibility and unboundedness not detected by NLopt
        "rotatedsoc2",
        # FIXME invalid NLopt arguments: bounds 4 fail 1 <= 0 <= 1
        "rotatedsoc3",
    ])
    MOIT.geomeantest(bridged, config, [
        # FIXME Status is ok but solution is not, e.g.
        #   Expression: ≈(MOI.get(model, MOI.ObjectiveValue()), 1.0, atol = atol, rtol = rtol)
        #   Evaluated: 0.006271435980170771 ≈ 1.0 (atol=0.01, rtol=0.01)
        "geomean2v", "geomean2f"
    ])
    MOIT.norminftest(bridged, config, [
        # Infeasibility and unboundedness not detected by NLopt
        "norminf2",
    ])
    MOIT.normonetest(bridged, config, [
        # Infeasibility and unboundedness not detected by NLopt
        "normone2",
    ])
end

@testset "Unit" begin
    exclude = [
        # Integer variables not supported
        "solve_zero_one_with_bounds_1",
        "solve_zero_one_with_bounds_2",
        "solve_zero_one_with_bounds_3",
        "solve_integer_edge_cases",
        # ObjectiveBound not supported.
        "solve_objbound_edge_cases",
        # NumberOfThreads not supported
        "number_threads",
        # Infeasibility and unboundedness not detected by NLopt
        "solve_unbounded_model",
        "solve_farkas_interval_lower",
        "solve_farkas_lessthan",
        "solve_farkas_equalto_lower",
        "solve_farkas_equalto_upper",
        "solve_farkas_variable_lessthan",
        "solve_farkas_variable_lessthan_max",
        "solve_farkas_greaterthan",
        "solve_farkas_interval_upper",
        "solve_farkas_lessthan",
    ]
    MOIT.unittest(bridged, config, exclude)
end
