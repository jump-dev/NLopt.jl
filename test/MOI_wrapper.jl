using Test

using MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

using NLopt
const solver = MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LD_SLSQP)

optimizer = MOI.instantiate(solver)

const config = MOIT.TestConfig(atol=1e-2, rtol=1e-2,
                               optimal_status=MOI.LOCALLY_SOLVED)

@testset "SolverName" begin
    @test MOI.get(optimizer, MOI.SolverName()) == "NLopt"
end

@testset "supports_default_copy_to" begin
    @test MOIU.supports_default_copy_to(optimizer, false)
    @test !MOIU.supports_default_copy_to(optimizer, true)
end

@testset "MOI NLP tests" begin
    MOIT.nlptest(optimizer, config)
end
