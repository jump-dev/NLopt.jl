using JuMP, NLopt, Test

m = Model(solver=NLoptSolver(algorithm=:LD_MMA))

a1 = 2
b1 = 0
a2 = -1
b2 = 1

@variable(m, x1)
@variable(m, x2 >= 0)

@NLobjective(m, Min, sqrt(x2))
@NLconstraint(m, x2 >= (a1*x1+b1)^3)
@NLconstraint(m, x2 >= (a2*x1+b2)^3)

setvalue(x1, 1.234)
setvalue(x2, 5.678)

status = solve(m)

println("got ", getobjectivevalue(m), " at ", [getvalue(x1),getvalue(x2)])

@test_approx_eq_eps getvalue(x1) 1/3 1e-5
@test_approx_eq_eps getvalue(x2) 8/27 1e-5
@test_approx_eq_eps getobjectivevalue(m) sqrt(8/27) 1e-5
@test status == :Optimal



