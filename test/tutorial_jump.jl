using JuMP
using NLopt
using Base.Test

m = Model(solver=NLoptSolver(algorithm=:LD_MMA))

a1 = 2
b1 = 0
a2 = -1
b2 = 1

@defVar(m, x1)
@defVar(m, x2 >= 0)

@setNLObjective(m, Min, sqrt(x2))
@addNLConstraint(m, x2 >= (a1*x1+b1)^3)
@addNLConstraint(m, x2 >= (a2*x1+b2)^3)

setValue(x1, 1.234)
setValue(x2, 5.678)

status = solve(m)

println("got ", getObjectiveValue(m), " at ", [getValue(x1),getValue(x2)])

@test_approx_eq_eps getValue(x1) 1/3 1e-5
@test_approx_eq_eps getValue(x2) 8/27 1e-5
@test_approx_eq_eps getObjectiveValue(m) sqrt(8/27) 1e-5
@test status == :Optimal



