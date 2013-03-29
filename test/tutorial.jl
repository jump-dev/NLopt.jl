using NLopt
using Base.Test

count = 0 # keep track of # function evaluations

function myfunc(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5/sqrt(x[2])
    end

    global count
    count::Int += 1
    println("f_$count($x)")

    sqrt(x[2])
end

function myconstraint(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

opt = Opt(:LD_MMA, 2)
lower_bounds!(opt, [-inf(0.), 0.])
xtol_rel!(opt,1e-4)

min_objective!(opt, myfunc)
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

(minf,minx,ret) = optimize(opt, [1.234, 5.678])
println("got $minf at $minx after $count iterations (returned $ret)")

@test_approx_eq_eps minx[1] 1/3 1e-5
@test_approx_eq_eps minx[2] 8/27 1e-5
@test_approx_eq_eps minf sqrt(8/27) 1e-5
@test ret == :XTOL_REACHED
