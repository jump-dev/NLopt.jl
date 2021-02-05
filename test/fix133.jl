using NLopt, Test

# Objective function
function rosenbrock(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1])
        grad[2] = 200 * (x[2] - x[1]^2)
    end
    (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2
end

function ineq01(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 1
        grad[2] = 2
    end
    x[1] + 2 * x[2] - 1
end

function ineq02(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 2*x[1]
        grad[2] = 1
    end
    x[1]^2 + x[2] - 1
end

function ineq03(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 2*x[1]
        grad[2] = -1
    end
    x[1]^2 - x[2] - 1
end

function eq01(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 2
        grad[2] = 1
    end
    2 * x[1] + x[2] - 1
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

(minf,minx,ret) = optimize(opt, [0.5, 0])
println("got $minf at $minx with constraints (returned $ret)")
@test minx[1] ≈ 0.4149 rtol=1e-3
@test minx[2] ≈ 0.1701 rtol=1e-3

remove_constraints!(opt)
(minf,minx,ret) = optimize(opt, [0.5, 0])
println("got $minf at $minx after removing constraints (returned $ret)")
@test minx[1] ≈ 1 rtol=1e-5
@test minx[2] ≈ 1 rtol=1e-5