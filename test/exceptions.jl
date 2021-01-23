using NLopt, Test

# Issue #156
let
    f(x, g=[]) = (error("test error"); x[1]^2)
    opt = Opt(:LN_SBPLX, 1)
    opt.min_objective = f
    @test_throws NLopt.SavedException{ErrorException} optimize(opt, [0.1234])
    @test NLopt.nlopt_exception === nothing
    try
        optimize(opt, [0.1234])
    catch e
        io = IOBuffer()
        show(io, e)
        # Check that the backtrace is being printed
        @test length(String(take!(io))) > 100
    end
end

let
    f(x, g=[]) = (throw(NLopt.ForcedStop()); x[1]^2)
    opt = Opt(:LN_SBPLX, 1)
    opt.min_objective = f
    fmin, xmin, ret = optimize(opt, [0.1234])
    @test ret == :FORCED_STOP
    @test NLopt.nlopt_exception === nothing
end

# check nlopt_exception is cleared correctly
let
    f(x, g=[]) = (x[1]^2)
    opt = Opt(:LN_SBPLX, 1)
    opt.min_objective = f
    fmin, xmin, ret = optimize(opt, [0.1234])
    @test ret ∈ (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED)
    @test NLopt.nlopt_exception === nothing
end
