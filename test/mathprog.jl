using NLopt

include(joinpath(Pkg.dir("MathProgBase"),"test","nlp.jl"))
nlptest(NLoptSolver(algorithm=:LD_SLSQP))
# test derivative-free
rosenbrocktest(NLoptSolver(algorithm=:LN_PRAXIS))
