

export NLoptSolver
immutable NLoptSolver <: MathProgSolverInterface.AbstractMathProgSolver
    algorithm::Symbol
    options
end

function NLoptSolver(;algorithm::Symbol=:none, kwargs...)
    if algorithm == :none
        error("Must specify an algorithm for NLoptSolver")
    end
    return NLoptSolver(algorithm, kwargs)
end

type NLoptMathProgModel <: MathProgSolverInterface.AbstractMathProgModel
    algorithm::Symbol
    options
    opt # can't create Opt object on construction because it needs problem dimensions
    x::Vector{Float64}
    objval::Float64
    status::Symbol
end

MathProgSolverInterface.model(s::NLoptSolver) = NLoptMathProgModel(s.algorithm, s.options, nothing, Float64[], NaN, :NotSolved)

function MathProgSolverInterface.loadnonlinearproblem!(m::NLoptMathProgModel, numVar::Integer, numConstr::Integer, x_l, x_u, g_lb, g_ub, sense::Symbol, d::MathProgSolverInterface.AbstractNLPEvaluator)

    (sense == :Min || sense == :Max) || error("Unrecognized sense $sense")
    m.opt = Opt(m.algorithm, numVar)

    lower_bounds!(m.opt, x_l)
    upper_bounds!(m.opt, x_u)

    eqidx = find(g_lb .== g_ub) # indices of equalities
    ineqidx = find(g_lb .!= g_ub)

    # map from eqidx/ineqidx to index in equalities/inequalities
    constrmap = Array(Int,numConstr)
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


    MathProgSolverInterface.initialize(d, [:Grad, :Jac])
    # TODO: don't ask for gradients if using derivative-free algorithm

    function f(x::Vector, grad::Vector)
        if length(grad) > 0
            MathProgSolverInterface.eval_grad_f(d, grad, x)
        end
        return MathProgSolverInterface.eval_f(d, x)
    end
    if sense == :Min
        min_objective!(m.opt, f)
    else
        max_objective!(m.opt, f)
    end

    Jac_I,Jac_J = MathProgSolverInterface.jac_structure(d)
    Jac_val = zeros(length(Jac_I))
    g_vec = zeros(numConstr)

    # somewhat inefficient because we evaluate g and the constraints
    # once for equalities and once for inequalities

    function g_eq(result::Vector, x::Vector, jac::Matrix)
        if length(jac) > 0
            fill!(jac, 0.0)
            MathProgSolverInterface.eval_jac_g(d, Jac_val, x)
            for k in 1:length(Jac_val)
                row = Jac_I[k]
                if g_lb[row] == g_ub[row]
                    jac[Jac_J[k],constrmap[row]] += Jac_val[k]
                end
            end
        end
        MathProgSolverInterface.eval_g(d, g_vec, x)
        for (ctr,idx) in enumerate(eqidx)
            result[ctr] = g_vec[idx] - g_ub[idx]
        end
    end

    # TODO: make tolerance a parameter
    equality_constraint!(m.opt, g_eq, zeros(numeq))

    # inequalities need to be massaged a bit
    # f(x) <= u   =>  f(x) - u <= 0
    # f(x) >= l   =>  l - f(x) <= 0

    function g_ineq(result::Vector, x::Vector, jac::Matrix)
        if length(jac) > 0
            fill!(jac, 0.0)
            MathProgSolverInterface.eval_jac_g(d, Jac_val, x)
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
                    jac[Jac_J[k],constrmap[row]+1] += Jac_val[k]
                end
            end
        end
        MathProgSolverInterface.eval_g(d, g_vec, x)
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


    equality_constraint!(m.opt, g_ineq, zeros(numineq))
end

function MathProgSolverInterface.setwarmstart!(m::NLoptMathProgModel,x)
    m.x = copy(float(x))
end

function MathProgSolverInterface.optimize!(m::NLoptMathProgModel)
    isa(m.opt, Opt) || error("Must load problem before solving")
    (optf,optx,ret) = optimize!(m.opt, m.x)
    m.objval = optf
    m.status = ret
end

function MathProgSolverInterface.status(m::NLoptMathProgModel)
    if m.status == :SUCCESS || m.status == :ROUNDOFF_LIMITED
        return :Optimal
    else
        error("Unknown status $(m.status)")
    end
end

MathProgSolverInterface.getsolution(m::NLoptMathProgModel) = m.x
MathProgSolverInterface.getobjval(m::NLoptMathProgModel) = m.objval

