# Copyright (c) 2013: Steven G. Johnson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestCAPI

using NLopt
using Test

function runtests()
    for name in names(@__MODULE__; all = true)
        if !startswith("$(name)", "test_")
            continue
        end
        @testset "$(name)" begin
            getfield(@__MODULE__, name)()
        end
    end
    return
end

function test_issue_163()
    opt = Opt(:LN_COBYLA, 2)
    opt.min_objective = (x, g) -> sum(x .^ 2)
    inequality_constraint!(opt, 2, (result, x, g) -> (result .= 1 .- x))
    (minf, minx, ret) = optimize(opt, [2.0, 2.0])
    @test minx ≈ [1.0, 1.0]
    return
end

function test_issue_132()
    opt = Opt(:LN_COBYLA, 2)
    err = ErrorException(
        "Getting `initial_step` is unsupported. Use " *
        "`initial_step(opt, x)` to access the initial step at a point `x`.",
    )
    @test_throws err opt.initial_step
    return
end

function test_issue_156_CapturedException()
    f(x, g = []) = (error("test error"); x[1]^2)
    opt = Opt(:LN_SBPLX, 1)
    opt.min_objective = f
    @test_throws CapturedException optimize(opt, [0.1234])
    @test NLopt.nlopt_exception === nothing
    try
        optimize(opt, [0.1234])
    catch e
        # Check that the backtrace is being printed
        @test length(sprint(show, e)) > 100
    end
    return
end

function test_issue_156_ForcedStop()
    f(x, g = []) = (throw(NLopt.ForcedStop()); x[1]^2)
    opt = Opt(:LN_SBPLX, 1)
    opt.min_objective = f
    fmin, xmin, ret = optimize(opt, [0.1234])
    @test ret == :FORCED_STOP
    @test NLopt.nlopt_exception === nothing
    return
end

function test_issue_156_no_error()
    f(x, g = []) = (x[1]^2)
    opt = Opt(:LN_SBPLX, 1)
    opt.min_objective = f
    fmin, xmin, ret = optimize(opt, [0.1234])
    @test ret ∈ (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED)
    @test NLopt.nlopt_exception === nothing
    return
end

function test_invalid_algorithms()
    @test_throws ArgumentError("unknown algorithm BILL") Algorithm(:BILL)
    @test_throws ArgumentError("unknown algorithm BILL") Opt(:BILL, 420)
    return
end

function test_issue_133()
    function rosenbrock(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1])
            grad[2] = 200 * (x[2] - x[1]^2)
        end
        return (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    end
    function ineq01(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = 1
            grad[2] = 2
        end
        return x[1] + 2 * x[2] - 1
    end
    function ineq02(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = 2 * x[1]
            grad[2] = 1
        end
        return x[1]^2 + x[2] - 1
    end
    function ineq03(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = 2 * x[1]
            grad[2] = -1
        end
        return x[1]^2 - x[2] - 1
    end
    function eq01(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = 2
            grad[2] = 1
        end
        return 2 * x[1] + x[2] - 1
    end
    opt = Opt(:LD_SLSQP, 2)
    opt.lower_bounds = [0, -0.5]
    opt.upper_bounds = [1, 2]
    opt.xtol_rel = 1e-21
    opt.min_objective = rosenbrock
    opt.inequality_constraint = ineq01
    opt.inequality_constraint = ineq02
    opt.inequality_constraint = ineq03
    opt.equality_constraint = eq01
    (minf, minx, ret) = optimize(opt, [0.5, 0])
    println("got $minf at $minx with constraints (returned $ret)")
    @test minx[1] ≈ 0.4149 rtol = 1e-3
    @test minx[2] ≈ 0.1701 rtol = 1e-3
    remove_constraints!(opt)
    (minf, minx, ret) = optimize(opt, [0.5, 0])
    println("got $minf at $minx after removing constraints (returned $ret)")
    @test minx[1] ≈ 1 rtol = 1e-5
    @test minx[2] ≈ 1 rtol = 1e-5
    return
end

function test_tutorial()
    count = 0 # keep track of # function evaluations
    function myfunc(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = 0
            grad[2] = 0.5 / sqrt(x[2])
        end
        global count
        count::Int += 1
        println("f_$count($x)")
        return sqrt(x[2])
    end
    function myconstraint(x::Vector, grad::Vector, a, b)
        if length(grad) > 0
            grad[1] = 3a * (a * x[1] + b)^2
            grad[2] = -1
        end
        return (a * x[1] + b)^3 - x[2]
    end
    opt = Opt(:LD_MMA, 2)
    opt.lower_bounds = [-Inf, 0.0]
    opt.xtol_rel = 1e-4
    opt.min_objective = myfunc
    opt.inequality_constraint = (x, g) -> myconstraint(x, g, 2, 0)
    opt.inequality_constraint = (x, g) -> myconstraint(x, g, -1, 1)
    # test algorithm-parameter API
    opt.params["verbosity"] = 0
    opt.params["inner_maxeval"] = 10
    opt.params["dual_alg"] = NLopt.LD_MMA
    @test opt.params == Dict(
        "verbosity" => 0,
        "inner_maxeval" => 10,
        "dual_alg" => Int(NLopt.LD_MMA),
    )
    @test get(opt.params, "foobar", 3.14159) === 3.14159
    (minf, minx, ret) = optimize(opt, [1.234, 5.678])
    println("got $minf at $minx after $count iterations (returned $ret)")
    @test minx[1] ≈ 1 / 3 rtol = 1e-5
    @test minx[2] ≈ 8 / 27 rtol = 1e-5
    @test minf ≈ sqrt(8 / 27) rtol = 1e-5
    @test ret == :XTOL_REACHED
    @test opt.numevals == count
    return
end

end  # module

TestCAPI.runtests()
