using JuMP, NLopt, Test

model = Model(NLopt.Optimizer)
set_optimizer_attribute(model, "algorithm", :LD_MMA)

a1 = 2
b1 = 0
a2 = -1
b2 = 1

@variable(model, x1)
@variable(model, x2 >= 0)

@NLobjective(model, Min, sqrt(x2))
@NLconstraint(model, x2 >= (a1*x1+b1)^3)
@NLconstraint(model, x2 >= (a2*x1+b2)^3)

fix(x1, 1.234)
fix(x2, 5.678)

optimize!(model)

println("got ", objective_value(model), " at ", [value(x1), value(x2)])

@test_approx_eq_eps value(x1) 1/3 1e-5
@test_approx_eq_eps value(x2) 8/27 1e-5
@test_approx_eq_eps objective_value(m) sqrt(8/27) 1e-5
@test termination_status(model) == MOI.OPTIMAL
