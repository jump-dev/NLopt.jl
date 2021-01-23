# Copyright (c) 2013: Steven G. Johnson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

include("tutorial.jl")
include("fix133.jl")
include("MOI_wrapper.jl")
include("exceptions.jl")

using NLopt
using Test

@testset "Fix #163" begin
    opt = Opt(:LN_COBYLA, 2)
    opt.min_objective = (x, g) -> sum(x.^2)
    inequality_constraint!(opt, 2, (result, x, g) -> (result .= 1 .- x))
    (minf, minx, ret) = optimize(opt, [2.0, 2.0])
    @test minx ≈ [1.0, 1.0]
end

@testset "Fix #132" begin
    opt = Opt(:LN_COBYLA, 2)
    err = ErrorException(
        "Getting `initial_step` is unsupported. Use " *
        "`initial_step(opt, x)` to access the initial step at a point `x`.",
    )
    @test_throws err opt.initial_step
end
