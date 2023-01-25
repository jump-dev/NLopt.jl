# Copyright (c) 2013: Steven G. Johnson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using NLopt

import MathProgBase
nlp = joinpath(dirname(dirname(pathof(MathProgBase))), "test", "nlp.jl")
include(nlp)

nlptest(NLoptSolver(algorithm=:LD_SLSQP))
# test derivative-free
rosenbrocktest(NLoptSolver(algorithm=:LN_PRAXIS))
