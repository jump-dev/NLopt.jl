# Support for the legacy MOI.NLPBlock interface. Keep this file in sync
# between NLPModelsJuMP.jl and NLopt.jl.

# An objective-only evaluator can omit every constraint callback. The QP
# layer still calls them on its inner evaluator when it evaluates its rows.
struct _ObjectiveOnlyEvaluator{E} <: MOI.AbstractNLPEvaluator
    inner::E
end

function MOI.features_available(d::_ObjectiveOnlyEvaluator)
    return union(MOI.features_available(d.inner), [:Jac, :JacVec])
end

function MOI.initialize(d::_ObjectiveOnlyEvaluator, features)
    return MOI.initialize(d.inner, setdiff(features, [:Jac, :JacVec]))
end

function MOI.eval_objective(d::_ObjectiveOnlyEvaluator, x)
    return MOI.eval_objective(d.inner, x)
end

function MOI.eval_objective_gradient(d::_ObjectiveOnlyEvaluator, g, x)
    return MOI.eval_objective_gradient(d.inner, g, x)
end

function MOI.eval_constraint(::_ObjectiveOnlyEvaluator, g, x)
    return
end

function MOI.jacobian_structure(::_ObjectiveOnlyEvaluator)
    return Tuple{Int,Int}[]
end

function MOI.eval_constraint_jacobian(::_ObjectiveOnlyEvaluator, J, x)
    return
end

function MOI.eval_constraint_jacobian_product(
    ::_ObjectiveOnlyEvaluator,
    y,
    x,
    v,
)
    return
end

function MOI.eval_constraint_jacobian_transpose_product(
    ::_ObjectiveOnlyEvaluator,
    y,
    x,
    v,
)
    fill!(y, 0)
    return
end

function MOI.hessian_lagrangian_structure(d::_ObjectiveOnlyEvaluator)
    return MOI.hessian_lagrangian_structure(d.inner)
end

function MOI.eval_hessian_lagrangian(d::_ObjectiveOnlyEvaluator, H, x, σ, μ)
    return MOI.eval_hessian_lagrangian(d.inner, H, x, σ, μ)
end

function MOI.eval_hessian_lagrangian_product(
    d::_ObjectiveOnlyEvaluator,
    Hv,
    x,
    v,
    σ,
    μ,
)
    return MOI.eval_hessian_lagrangian_product(d.inner, Hv, x, v, σ, μ)
end

function _legacy_evaluator(
    model::MOI.Nonlinear.ModelWithQuad,
    data::MOI.NLPBlockData,
    backend,
)
    if isempty(data.constraint_bounds)
        if !data.has_objective
            vars = MOI.get(model, MOI.ListOfVariableIndices())
            return MOI.Nonlinear.Evaluator(model, backend, vars)
        end
        inner = _ObjectiveOnlyEvaluator(data.evaluator)
    else
        inner = data.evaluator
    end
    return MOI.Nonlinear.EvaluatorWithQuad(model, inner)
end

# EvaluatorWithQuad places the model's rows before the legacy rows.
function _constraint_bounds(model, data::Union{Nothing,MOI.NLPBlockData})
    bounds = MOI.Utilities.constraint_bounds(model)
    pairs = MOI.NLPBoundsPair.(bounds.lower, bounds.upper)
    if !isnothing(data)
        append!(pairs, data.constraint_bounds)
    end
    return pairs
end
