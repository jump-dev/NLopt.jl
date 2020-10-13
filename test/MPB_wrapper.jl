using NLopt

import MathProgBase
nlp = joinpath(dirname(dirname(pathof(MathProgBase))), "test", "nlp.jl")
include(nlp)

nlptest(NLoptSolver(algorithm=:LD_SLSQP))
# test derivative-free
rosenbrocktest(NLoptSolver(algorithm=:LN_PRAXIS))
