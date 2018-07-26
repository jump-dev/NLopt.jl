

export NLoptSolver
struct NLoptSolver <: SolverInterface.AbstractMathProgSolver
    algorithm::Symbol
    stopval::Real
    ftol_rel::Real
    ftol_abs::Real
    xtol_rel::Real
    xtol_abs
    constrtol_abs::Real
    maxeval::Integer
    maxtime::Real
    initial_step
    population::Integer
    seed
    vector_storage::Integer
end

function NLoptSolver(;algorithm::Symbol=:none, stopval::Real=NaN,
    ftol_rel::Real=1e-7, ftol_abs::Real=NaN, xtol_rel::Real=1e-7,
    xtol_abs=nothing, constrtol_abs=1e-7,
    maxeval::Integer=0, maxtime::Real=0, initial_step=nothing,
    population::Integer=0, seed=nothing, vector_storage::Integer=0)
    if algorithm == :none
        error("Must specify an algorithm for NLoptSolver")
    end
    return NLoptSolver(algorithm, stopval, ftol_rel, ftol_abs, xtol_rel,
        xtol_abs, constrtol_abs, maxeval, maxtime, initial_step, population, seed,
        vector_storage)
end

mutable struct NLoptMathProgModel <: SolverInterface.AbstractNonlinearModel
    algorithm::Symbol
    opt # can't create Opt object on construction because it needs problem dimensions
    x::Vector{Float64}
    objval::Float64
    status::Symbol
    # options...
    stopval::Real
    ftol_rel::Real
    ftol_abs::Real
    xtol_rel::Real
    xtol_abs
    constrtol_abs::Real
    maxeval::Integer
    maxtime::Real
    initial_step
    population::Integer
    seed
    vector_storage::Integer
end

SolverInterface.NonlinearModel(s::NLoptSolver) = NLoptMathProgModel(s.algorithm, nothing, Float64[], NaN, :NotSolved, s.stopval, s.ftol_rel, s.ftol_abs, s.xtol_rel, s.xtol_abs, s.constrtol_abs, s.maxeval, s.maxtime, s.initial_step, s.population, s.seed, s.vector_storage)

function SolverInterface.loadproblem!(m::NLoptMathProgModel, numVar::Integer, numConstr::Integer, x_l, x_u, g_lb, g_ub, sense::Symbol, d::SolverInterface.AbstractNLPEvaluator)

    (sense == :Min || sense == :Max) || error("Unrecognized sense $sense")
    m.opt = Opt(m.algorithm, numVar)

    # load parameters
    stopval!(m.opt, m.stopval)
    if !isnan(m.ftol_rel)
        ftol_rel!(m.opt, m.ftol_rel)
    end
    if !isnan(m.ftol_abs)
        ftol_abs!(m.opt, m.ftol_abs)
    end
    if !isnan(m.xtol_rel)
        xtol_rel!(m.opt, m.xtol_rel)
    end
    if m.xtol_abs != nothing
        xtol_abs!(m.opt, m.xtol_abs)
    end
    maxeval!(m.opt, m.maxeval)
    maxtime!(m.opt, m.maxtime)
    if m.initial_step != nothing
        initial_step!(m.opt, m.initial_step)
    end
    population!(m.opt, m.population)
    if isa(m.seed, Integer)
        NLopt.srand(m.seed)
    end
    vector_storage!(m.opt, m.vector_storage)

    lower_bounds!(m.opt, x_l)
    upper_bounds!(m.opt, x_u)

    eqidx = findall(g_lb .== g_ub) # indices of equalities
    ineqidx = findall(g_lb .!= g_ub)

    # map from eqidx/ineqidx to index in equalities/inequalities
    constrmap = zeros(Int, numConstr)
    for i in 1:length(eqidx)
        constrmap[eqidx[i]] = i
    end
    ineqcounter = 1
    for i in 1:length(ineqidx)
        k = ineqidx[i]
        constrmap[k] = ineqcounter
        if isinf(g_lb[k]) || isinf(g_ub[k])
            ineqcounter += 1
        else # constraint has bounds on both sides, keep room for it
            ineqcounter += 2
        end
    end
    numineq = ineqcounter-1
    numeq = length(eqidx)

    isderivativefree = string(m.algorithm)[2] == 'N'
    if isderivativefree
        requested_features = Symbol[]
    else
        requested_features = numConstr > 0 ? [:Grad, :Jac] : [:Grad]
    end

    SolverInterface.initialize(d, requested_features)

    function f(x::Vector, grad::Vector)
        if length(grad) > 0
            SolverInterface.eval_grad_f(d, grad, x)
        end
        return SolverInterface.eval_f(d, x)
    end
    if sense == :Min
        min_objective!(m.opt, f)
    else
        max_objective!(m.opt, f)
    end

    Jac_I,Jac_J = numConstr > 0 ? SolverInterface.jac_structure(d) : (Int[], Int[])
    Jac_val = zeros(length(Jac_I))
    g_vec = zeros(numConstr)

    # somewhat inefficient because we evaluate g and the constraints
    # once for equalities and once for inequalities

    function g_eq(result::Vector, x::Vector, jac::Matrix)
        if length(jac) > 0
            fill!(jac, 0.0)
            SolverInterface.eval_jac_g(d, Jac_val, x)
            for k in 1:length(Jac_val)
                row = Jac_I[k]
                if g_lb[row] == g_ub[row]
                    jac[Jac_J[k],constrmap[row]] += Jac_val[k]
                end
            end
        end
        SolverInterface.eval_g(d, g_vec, x)
        for (ctr,idx) in enumerate(eqidx)
            result[ctr] = g_vec[idx] - g_ub[idx]
        end
    end

    equality_constraint!(m.opt, g_eq, fill(m.constrtol_abs,numeq))

    # inequalities need to be massaged a bit
    # f(x) <= u   =>  f(x) - u <= 0
    # f(x) >= l   =>  l - f(x) <= 0

    function g_ineq(result::Vector, x::Vector, jac::Matrix)
        if length(jac) > 0
            fill!(jac, 0.0)
            SolverInterface.eval_jac_g(d, Jac_val, x)
            for k in 1:length(Jac_val)
                row = Jac_I[k]
                g_lb[row] == g_ub[row] && continue
                if isinf(g_lb[row]) # upper bound
                    jac[Jac_J[k],constrmap[row]] += Jac_val[k]
                elseif isinf(g_ub[row]) # lower bound
                    jac[Jac_J[k],constrmap[row]] -= Jac_val[k]
                else
                    # boxed
                    jac[Jac_J[k],constrmap[row]] += Jac_val[k]
                    jac[Jac_J[k],constrmap[row]+1] -= Jac_val[k]
                end
            end
        end
        SolverInterface.eval_g(d, g_vec, x)
        for row in 1:numConstr
            g_lb[row] == g_ub[row] && continue
            if isinf(g_lb[row])
                result[constrmap[row]] = g_vec[row] - g_ub[row]
            elseif isinf(g_ub[row])
                result[constrmap[row]] = g_lb[row] - g_vec[row]
            else
                result[constrmap[row]] = g_vec[row] - g_ub[row]
                result[constrmap[row]+1] = g_lb[row] - g_vec[row]
            end
        end
    end


    inequality_constraint!(m.opt, g_ineq, fill(m.constrtol_abs, numineq))
end

function SolverInterface.setwarmstart!(m::NLoptMathProgModel,x)
    m.x = copy(float(x))
end

function SolverInterface.optimize!(m::NLoptMathProgModel)
    isa(m.opt, Opt) || error("Must load problem before solving")
    (optf,optx,ret) = optimize!(m.opt, m.x)
    m.objval = optf
    m.status = ret
end

function SolverInterface.status(m::NLoptMathProgModel)
    if m.status == :SUCCESS || m.status == :FTOL_REACHED || m.status == :XTOL_REACHED
        return :Optimal
    elseif m.status == :ROUNDOFF_LIMITED
        return :Suboptimal
    elseif m.status in (:STOPVAL_REACHED,:MAXEVAL_REACHED,:MAXTIME_REACHED)
        return :UserLimit
    else
        error("Unknown status $(m.status)")
    end
end

SolverInterface.getsolution(m::NLoptMathProgModel) = m.x
SolverInterface.getobjval(m::NLoptMathProgModel) = m.objval

