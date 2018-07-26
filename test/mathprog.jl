using NLopt

# test model from MathProgBase, using undocumented
# function Base.find_package for now (julia#27592):
nlp = joinpath(dirname(Base.find_package(NLopt, "MathProgBase")), "..", "test", "nlp.jl")
include(nlp)

nlptest(NLoptSolver(algorithm=:LD_SLSQP))
# test derivative-free
rosenbrocktest(NLoptSolver(algorithm=:LN_PRAXIS))
