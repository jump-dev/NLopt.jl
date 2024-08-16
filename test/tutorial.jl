# Copyright (c) 2013: Steven G. Johnson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using NLopt, Test

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
