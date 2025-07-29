# Copyright (c) 2013: Steven G. Johnson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module NLoptMathOptInterfaceExt

import MathOptInterface as MOI
import NLopt

function __init__()
    # we need to add extension types back to the toplevel module
    @static if VERSION >= v"1.9"
        setglobal!(NLopt, :Optimizer, Optimizer)
    end
    return
end

mutable struct _ConstraintInfo{F,S}
    func::F
    set::S
end

"""
    Optimizer()

Create a new Optimizer object.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::Union{NLopt.Opt,Nothing}
    variables::MOI.Utilities.VariablesContainer{Float64}
    starting_values::Vector{Union{Nothing,Float64}}
    nlp_data::MOI.NLPBlockData
    nlp_model::Union{Nothing,MOI.Nonlinear.Model}
    ad_backend::MOI.Nonlinear.AbstractAutomaticDifferentiation
    sense::Union{Nothing,MOI.OptimizationSense}
    objective::Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
        Nothing,
    }
    linear_le_constraints::Vector{
        _ConstraintInfo{
            MOI.ScalarAffineFunction{Float64},
            MOI.LessThan{Float64},
        },
    }
    linear_eq_constraints::Vector{
        _ConstraintInfo{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}},
    }
    quadratic_le_constraints::Vector{
        _ConstraintInfo{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.LessThan{Float64},
        },
    }
    quadratic_eq_constraints::Vector{
        _ConstraintInfo{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.EqualTo{Float64},
        },
    }
    # Parameters.
    silent::Bool
    options::Dict{String,Any}
    # Solution attributes.
    objective_value::Float64
    solution::Vector{Float64}
    status::Symbol
    solve_time::Float64

    function Optimizer()
        return new(
            nothing,
            MOI.Utilities.VariablesContainer{Float64}(),
            Union{Nothing,Float64}[],
            MOI.NLPBlockData([], _EmptyNLPEvaluator(), false),
            nothing,
            MOI.Nonlinear.SparseReverseMode(),
            nothing,
            nothing,
            _ConstraintInfo{
                MOI.ScalarAffineFunction{Float64},
                MOI.LessThan{Float64},
            }[],
            _ConstraintInfo{
                MOI.ScalarAffineFunction{Float64},
                MOI.EqualTo{Float64},
            }[],
            _ConstraintInfo{
                MOI.ScalarQuadraticFunction{Float64},
                MOI.LessThan{Float64},
            }[],
            _ConstraintInfo{
                MOI.ScalarQuadraticFunction{Float64},
                MOI.EqualTo{Float64},
            }[],
            false,
            copy(_DEFAULT_OPTIONS),
            NaN,
            Float64[],
            :NOT_CALLED,
            NaN,
        )
    end
end

struct _EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

MOI.initialize(::_EmptyNLPEvaluator, ::Vector{Symbol}) = nothing

MOI.eval_constraint(::_EmptyNLPEvaluator, g, x) = nothing

MOI.eval_constraint_jacobian(::_EmptyNLPEvaluator, J, x) = nothing

function MOI.empty!(model::Optimizer)
    model.inner = nothing
    MOI.empty!(model.variables)
    empty!(model.starting_values)
    model.nlp_data = MOI.NLPBlockData([], _EmptyNLPEvaluator(), false)
    model.nlp_model = nothing
    model.sense = nothing
    model.objective = nothing
    empty!(model.linear_le_constraints)
    empty!(model.linear_eq_constraints)
    empty!(model.quadratic_le_constraints)
    empty!(model.quadratic_eq_constraints)
    model.status = :NOT_CALLED
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.variables) &&
           isempty(model.starting_values) &&
           model.nlp_data.evaluator isa _EmptyNLPEvaluator &&
           model.nlp_model === nothing &&
           model.sense == nothing &&
           isempty(model.linear_le_constraints) &&
           isempty(model.linear_eq_constraints) &&
           isempty(model.quadratic_le_constraints) &&
           isempty(model.quadratic_eq_constraints)
end

function MOI.get(model::Optimizer, ::MOI.ListOfModelAttributesSet)
    ret = MOI.AbstractModelAttribute[]
    if model.sense !== nothing
        push!(ret, MOI.ObjectiveSense())
    end
    if model.objective !== nothing
        F = MOI.get(model, MOI.ObjectiveFunctionType())
        push!(ret, MOI.ObjectiveFunction{F}())
    end
    return ret
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "NLopt"

MOI.get(::Optimizer, ::MOI.SolverVersion) = "$(NLopt.version())"

function _constraints(
    model,
    ::Type{<:MOI.ScalarAffineFunction},
    ::Type{<:MOI.LessThan},
)
    return model.linear_le_constraints
end

function _constraints(
    model,
    ::Type{<:MOI.ScalarAffineFunction},
    ::Type{<:MOI.EqualTo},
)
    return model.linear_eq_constraints
end

function _constraints(
    model,
    ::Type{<:MOI.ScalarQuadraticFunction},
    ::Type{<:MOI.LessThan},
)
    return model.quadratic_le_constraints
end

function _constraints(
    model,
    ::Type{<:MOI.ScalarQuadraticFunction},
    ::Type{<:MOI.EqualTo},
)
    return model.quadratic_eq_constraints
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{
        <:Union{
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
        },
    },
    ::Type{<:Union{MOI.LessThan{Float64},MOI.EqualTo{Float64}}},
)
    return true
end

function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S<:Union{MOI.LessThan{Float64},MOI.EqualTo{Float64}},
}
    return length(_constraints(model, F, S))
end

function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintTypesPresent)
    constraints = MOI.get(model.variables, attr)
    function _check(model, F, S)
        if !isempty(_constraints(model, F, S))
            push!(constraints, (F, S))
        end
    end
    _check(model, MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64})
    _check(model, MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64})
    _check(model, MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64})
    _check(model, MOI.ScalarQuadraticFunction{Float64}, MOI.EqualTo{Float64})
    return constraints
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S<:Union{MOI.LessThan{Float64},MOI.EqualTo{Float64}},
}
    return MOI.ConstraintIndex{F,S}.(eachindex(_constraints(model, F, S)))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S<:Union{MOI.LessThan{Float64},MOI.EqualTo{Float64}},
}
    return copy(_constraints(model, F, S)[c.value].func)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S<:Union{MOI.LessThan{Float64},MOI.EqualTo{Float64}},
}
    return _constraints(model, F, S)[c.value].set
end

# ObjectiveSense

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    model.sense = sense
    return
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveSense)
    return something(model.sense, MOI.FEASIBILITY_SENSE)
end

# MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

# MOI.TimeLimitSec

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    MOI.set(model, MOI.RawOptimizerAttribute("max_cpu_time"), Float64(value))
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    delete!(model.options, "max_cpu_time")
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return get(model.options, "max_cpu_time", nothing)
end

# MOI.RawOptimizerAttribute

const _DEFAULT_OPTIONS = Dict{String,Any}(
    "algorithm" => :none,
    "stopval" => NaN,
    "ftol_rel" => 1e-7,
    "ftol_abs" => NaN,
    "xtol_rel" => 1e-7,
    "xtol_abs" => nothing,
    "constrtol_abs" => 1e-7,
    "maxeval" => 0,
    "maxtime" => 0.0,
    "initial_step" => nothing,
    "population" => 0,
    "seed" => nothing,
    "vector_storage" => 0,
    "local_optimizer" => nothing,
)

function MOI.supports(::Optimizer, p::MOI.RawOptimizerAttribute)
    # TODO(odow): this ignores other algorithm-specific parameters?
    return haskey(_DEFAULT_OPTIONS, p.name)
end

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    model.options[p.name] = value
    return
end

function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    if !haskey(model.options, p.name)
        msg = "RawOptimizerAttribute with name $(p.name) is not set."
        throw(MOI.GetAttributeNotAllowed(p, msg))
    end
    return model.options[p.name]
end

# Variables

function MOI.get(model::Optimizer, ::MOI.ListOfVariableAttributesSet)
    ret = MOI.AbstractVariableAttribute[]
    if any(!isnothing, model.starting_values)
        push!(ret, MOI.VariablePrimalStart())
    end
    return ret
end

function MOI.add_variable(model::Optimizer)
    push!(model.starting_values, nothing)
    return MOI.add_variable(model.variables)
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{
        <:Union{
            MOI.LessThan{Float64},
            MOI.GreaterThan{Float64},
            MOI.EqualTo{Float64},
            MOI.Interval{Float64},
        },
    },
)
    return true
end

function MOI.get(
    model::Optimizer,
    attr::Union{
        MOI.NumberOfVariables,
        MOI.ListOfVariableIndices,
        MOI.NumberOfConstraints{MOI.VariableIndex},
        MOI.ListOfConstraintIndices{MOI.VariableIndex},
    },
)
    return MOI.get(model.variables, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet},
    ci::MOI.ConstraintIndex{MOI.VariableIndex},
)
    return MOI.get(model.variables, attr, ci)
end

function MOI.is_valid(
    model::Optimizer,
    index::Union{MOI.VariableIndex,MOI.ConstraintIndex{MOI.VariableIndex}},
)
    return MOI.is_valid(model.variables, index)
end

function MOI.add_constraint(
    model::Optimizer,
    vi::MOI.VariableIndex,
    set::Union{
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.EqualTo{Float64},
        MOI.Interval{Float64},
    },
)
    return MOI.add_constraint(model.variables, vi, set)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
    set::S,
) where {S}
    return MOI.set(model.variables, attr, ci, set)
end

function MOI.delete(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex},
)
    return MOI.delete(model.variables, ci)
end

# constraints

function MOI.get(::Optimizer, ::MOI.ListOfConstraintAttributesSet)
    return MOI.AbstractConstraintAttribute[]
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S<:Union{MOI.LessThan{Float64},MOI.EqualTo{Float64}},
}
    return 1 <= ci.value <= length(_constraints(model, F, S))
end

function _check_inbounds(model, f::MOI.VariableIndex)
    return MOI.throw_if_not_valid(model, f)
end

function _check_inbounds(model, f::MOI.ScalarAffineFunction{Float64})
    for term in f.terms
        MOI.throw_if_not_valid(model, term.variable)
    end
    return
end

function _check_inbounds(model, f::MOI.ScalarQuadraticFunction{Float64})
    for term in f.affine_terms
        MOI.throw_if_not_valid(model, term.variable)
    end
    for term in f.quadratic_terms
        MOI.throw_if_not_valid(model, term.variable_1)
        MOI.throw_if_not_valid(model, term.variable_2)
    end
    return
end

function MOI.add_constraint(
    model::Optimizer,
    func::F,
    set::S,
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S<:Union{MOI.LessThan{Float64},MOI.EqualTo{Float64}},
}
    _check_inbounds(model, func)
    constraints = _constraints(model, F, S)
    push!(constraints, _ConstraintInfo(func, set))
    return MOI.ConstraintIndex{F,S}(length(constraints))
end

# MOI.VariablePrimalStart

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, vi)
    model.starting_values[vi.value] = value
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
)
    MOI.throw_if_not_valid(model, vi)
    return model.starting_values[vi.value]
end

# MOI.NLPBlock

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    if model.nlp_model !== nothing
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    model.nlp_data = nlp_data
    return
end

# MOI.ObjectiveFunction

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{
        <:Union{
            MOI.VariableIndex,
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
        },
    },
)
    return true
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveFunctionType)
    if model.nlp_model !== nothing && model.nlp_model.objective !== nothing
        return MOI.ScalarNonlinearFunction
    end
    return typeof(model.objective)
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveFunction{F}) where {F}
    return convert(F, model.objective)::F
end

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveFunction{F},
    func::F,
) where {
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    _check_inbounds(model, func)
    model.objective = func
    if model.nlp_model !== nothing
        MOI.Nonlinear.set_objective(model.nlp_model, nothing)
    end
    return
end

# ScalarNonlinearFunction

function _init_nlp_model(model)
    if model.nlp_model === nothing
        if !(model.nlp_data.evaluator isa _EmptyNLPEvaluator)
            error("Cannot mix the new and legacy nonlinear APIs")
        end
        model.nlp_model = MOI.Nonlinear.Model()
    end
    return
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,S},
) where {
    S<:Union{
        MOI.EqualTo{Float64},
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.Interval{Float64},
    },
}
    if model.nlp_model === nothing
        return false
    end
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    return MOI.is_valid(model.nlp_model, index)
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarNonlinearFunction},
    ::Type{S},
) where {
    S<:Union{
        MOI.EqualTo{Float64},
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.Interval{Float64},
    },
}
    return true
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarNonlinearFunction,
    set::Union{
        MOI.EqualTo{Float64},
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.Interval{Float64},
    },
)
    _init_nlp_model(model)
    index = MOI.Nonlinear.add_constraint(model.nlp_model, f, set)
    return MOI.ConstraintIndex{typeof(f),typeof(set)}(index.value)
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
)
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
    func::MOI.ScalarNonlinearFunction,
)
    _init_nlp_model(model)
    MOI.Nonlinear.set_objective(model.nlp_model, func)
    return
end

### MOI.AutomaticDifferentiationBackend

MOI.supports(::Optimizer, ::MOI.AutomaticDifferentiationBackend) = true

function MOI.get(model::Optimizer, ::MOI.AutomaticDifferentiationBackend)
    return model.ad_backend
end

function MOI.set(
    model::Optimizer,
    ::MOI.AutomaticDifferentiationBackend,
    backend::MOI.Nonlinear.AbstractAutomaticDifferentiation,
)
    model.ad_backend = backend
    return
end

### MOI.UserDefinedFunction

MOI.supports(model::Optimizer, ::MOI.UserDefinedFunction) = true

function MOI.set(model::Optimizer, attr::MOI.UserDefinedFunction, args)
    _init_nlp_model(model)
    MOI.Nonlinear.register_operator(
        model.nlp_model,
        attr.name,
        attr.arity,
        args...,
    )
    return
end

### MOI.ListOfSupportedNonlinearOperators

function MOI.get(model::Optimizer, attr::MOI.ListOfSupportedNonlinearOperators)
    _init_nlp_model(model)
    return MOI.get(model.nlp_model, attr)
end

# optimize!

function _fill_gradient(grad, x, f::MOI.VariableIndex)
    grad[f.value] = 1.0
    return
end

function _fill_gradient(grad, x, f::MOI.ScalarAffineFunction{Float64})
    for term in f.terms
        grad[term.variable.value] += term.coefficient
    end
    return
end

function _fill_gradient(grad, x, f::MOI.ScalarQuadraticFunction{Float64})
    for term in f.affine_terms
        grad[term.variable.value] += term.coefficient
    end
    for term in f.quadratic_terms
        i, j = term.variable_1.value, term.variable_2.value
        grad[i] += term.coefficient * x[j]
        if i != j
            grad[j] += term.coefficient * x[i]
        end
    end
    return
end

function _fill_result(result::Vector, x, offset, constraints::Vector)
    for (i, constraint) in enumerate(constraints)
        lhs = MOI.Utilities.eval_variables(vi -> x[vi.value], constraint.func)
        result[offset+i] = lhs - MOI.constant(constraint.set)
    end
    return
end

function _fill_jacobian(jac, x, offset, term::MOI.ScalarAffineTerm)
    jac[term.variable.value, offset] += term.coefficient
    return
end

function _fill_jacobian(jac, x, offset, term::MOI.ScalarQuadraticTerm)
    i, j = term.variable_1.value, term.variable_2.value
    jac[i, offset] += term.coefficient * x[j]
    if i != j
        jac[j, offset] += term.coefficient * x[i]
    end
    return
end

function _fill_jacobian(jac, x, offset, f::MOI.ScalarAffineFunction)
    for term in f.terms
        _fill_jacobian(jac, x, offset, term)
    end
    return
end

function _fill_jacobian(jac, x, offset, f::MOI.ScalarQuadraticFunction)
    for term in f.affine_terms
        _fill_jacobian(jac, x, offset, term)
    end
    for q_term in f.quadratic_terms
        _fill_jacobian(jac, x, offset, q_term)
    end
    return
end

function _fill_jacobian(jac, x, offset, constraints::Vector)
    for (i, constraint) in enumerate(constraints)
        _fill_jacobian(jac, x, offset + i, constraint.func)
    end
    return
end

function objective_fn(model::Optimizer, x::Vector, grad::Vector)
    # The order of the conditions is important. NLP objectives override regular
    # objectives.
    if length(grad) > 0
        fill!(grad, 0.0)
        if model.sense == MOI.FEASIBILITY_SENSE
            # nothing
        elseif model.nlp_data.has_objective
            MOI.eval_objective_gradient(model.nlp_data.evaluator, grad, x)
        elseif model.objective !== nothing
            _fill_gradient(grad, x, model.objective)
        end
    end
    if model.sense == MOI.FEASIBILITY_SENSE
        return 0.0
    elseif model.nlp_data.has_objective
        return MOI.eval_objective(model.nlp_data.evaluator, x)
    elseif model.objective !== nothing
        return MOI.Utilities.eval_variables(vi -> x[vi.value], model.objective)
    end
    # No ObjectiveFunction is set, but ObjectiveSense is?
    return 0.0
end

function _initialize_options!(model::Optimizer)
    local_optimizer = model.options["local_optimizer"]
    if local_optimizer !== nothing
        num_variables = length(model.starting_values)
        local_optimizer = if local_optimizer isa Symbol
            NLopt.Opt(local_optimizer, num_variables)
        else
            @assert local_optimizer isa NLopt.Opt
            NLopt.Opt(local_optimizer.algorithm, num_variables)
        end
        NLopt.local_optimizer!(model.inner, local_optimizer)
    end
    NLopt.stopval!(model.inner, model.options["stopval"])
    if !isnan(model.options["ftol_rel"])
        NLopt.ftol_rel!(model.inner, model.options["ftol_rel"])
    end
    if !isnan(model.options["ftol_abs"])
        NLopt.ftol_abs!(model.inner, model.options["ftol_abs"])
    end
    if !isnan(model.options["xtol_rel"])
        NLopt.xtol_rel!(model.inner, model.options["xtol_rel"])
    end
    if model.options["xtol_abs"] != nothing
        NLopt.xtol_abs!(model.inner, model.options["xtol_abs"])
    end
    NLopt.maxeval!(model.inner, model.options["maxeval"])
    NLopt.maxtime!(model.inner, model.options["maxtime"])
    if model.options["initial_step"] != nothing
        NLopt.initial_step!(model.inner, model.options["initial_step"])
    end
    NLopt.population!(model.inner, model.options["population"])
    if model.options["seed"] isa Integer
        NLopt.srand(model.options["seed"])
    end
    NLopt.vector_storage!(model.inner, model.options["vector_storage"])
    return
end

function MOI.optimize!(model::Optimizer)
    num_variables = length(model.starting_values)
    model.inner = NLopt.Opt(model.options["algorithm"], num_variables)
    _initialize_options!(model)
    if model.nlp_model !== nothing
        vars = MOI.VariableIndex.(1:num_variables)
        model.nlp_data = MOI.NLPBlockData(
            MOI.Nonlinear.Evaluator(model.nlp_model, model.ad_backend, vars),
        )
    end
    NLopt.lower_bounds!(model.inner, model.variables.lower)
    NLopt.upper_bounds!(model.inner, model.variables.upper)
    nonlinear_equality_indices = findall(
        bound -> bound.lower == bound.upper,
        model.nlp_data.constraint_bounds,
    )
    nonlinear_inequality_indices = findall(
        bound -> bound.lower != bound.upper,
        model.nlp_data.constraint_bounds,
    )
    num_nlpblock_constraints = length(model.nlp_data.constraint_bounds)
    # map from eqidx/ineqidx to index in equalities/inequalities
    constrmap = zeros(Int, num_nlpblock_constraints)
    for (i, k) in enumerate(nonlinear_equality_indices)
        constrmap[k] = i
    end
    num_nlpblock_inequalities = 0
    for (i, k) in enumerate(nonlinear_inequality_indices)
        num_nlpblock_inequalities += 1
        constrmap[k] = num_nlpblock_inequalities
        bounds = model.nlp_data.constraint_bounds[k]
        if !isinf(bounds.lower) && !isinf(bounds.upper)
            # constraint has bounds on both sides, keep room for it
            num_nlpblock_inequalities += 1
        end
    end
    if string(model.options["algorithm"])[2] == 'N'
        # Derivative free optimizer chosen
        MOI.initialize(model.nlp_data.evaluator, Symbol[])
    elseif num_nlpblock_constraints > 0
        MOI.initialize(model.nlp_data.evaluator, [:Grad, :Jac])
    else
        MOI.initialize(model.nlp_data.evaluator, [:Grad])
    end
    if model.sense == MOI.MAX_SENSE
        NLopt.max_objective!(model.inner, (x, g) -> objective_fn(model, x, g))
    else
        NLopt.min_objective!(model.inner, (x, g) -> objective_fn(model, x, g))
    end
    Jac_IJ = Tuple{Int,Int}[]
    if num_nlpblock_constraints > 0
        append!(Jac_IJ, MOI.jacobian_structure(model.nlp_data.evaluator))
    end
    Jac_val = zeros(length(Jac_IJ))
    g_vec = zeros(num_nlpblock_constraints)
    function equality_constraint_fn(result::Vector, x::Vector, jac::Matrix)
        if length(jac) > 0
            fill!(jac, 0.0)
            MOI.eval_constraint_jacobian(model.nlp_data.evaluator, Jac_val, x)
            for ((row, col), val) in zip(Jac_IJ, Jac_val)
                bounds = model.nlp_data.constraint_bounds[row]
                if bounds.lower == bounds.upper
                    jac[col, constrmap[row]] += val
                end
            end
            offset = length(nonlinear_equality_indices)
            _fill_jacobian(jac, x, offset, model.linear_eq_constraints)
            offset += length(model.linear_eq_constraints)
            _fill_jacobian(jac, x, offset, model.quadratic_eq_constraints)
        end
        MOI.eval_constraint(model.nlp_data.evaluator, g_vec, x)
        for (i, index) in enumerate(nonlinear_equality_indices)
            bounds = model.nlp_data.constraint_bounds[index]
            result[i] = g_vec[index] - bounds.upper
        end
        offset = length(nonlinear_equality_indices)
        _fill_result(result, x, offset, model.linear_eq_constraints)
        offset += length(model.linear_eq_constraints)
        _fill_result(result, x, offset, model.quadratic_eq_constraints)
        return
    end
    num_equality_constraints =
        length(nonlinear_equality_indices) +
        length(model.linear_eq_constraints) +
        length(model.quadratic_eq_constraints)
    if num_equality_constraints > 0
        NLopt.equality_constraint!(
            model.inner,
            num_equality_constraints,
            equality_constraint_fn,
            model.options["constrtol_abs"],
        )
    end
    # inequalities need to be massaged a bit
    # f(x) <= u   =>  f(x) - u <= 0
    # f(x) >= l   =>  l - f(x) <= 0
    function inequality_constraint_fn(result::Vector, x::Vector, jac::Matrix)
        if length(jac) > 0
            fill!(jac, 0.0)
            MOI.eval_constraint_jacobian(model.nlp_data.evaluator, Jac_val, x)
            for ((row, col), val) in zip(Jac_IJ, Jac_val)
                bounds = model.nlp_data.constraint_bounds[row]
                if bounds.lower == bounds.upper
                    continue  # This is an equality constraint
                elseif isinf(bounds.lower)  # upper bound
                    jac[col, constrmap[row]] += val
                elseif isinf(bounds.upper)  # lower bound
                    jac[col, constrmap[row]] -= val
                else  # boxed
                    jac[col, constrmap[row]] += val
                    jac[col, constrmap[row]+1] -= val
                end
            end
            offset = num_nlpblock_inequalities
            _fill_jacobian(jac, x, offset, model.linear_le_constraints)
            offset += length(model.linear_le_constraints)
            _fill_jacobian(jac, x, offset, model.quadratic_le_constraints)
        end
        # Fill in the result. The first entries are from NLPBlock, and the value
        # of g(x) is placed in g_vec.
        MOI.eval_constraint(model.nlp_data.evaluator, g_vec, x)
        for row in 1:num_nlpblock_constraints
            index = constrmap[row]
            bounds = model.nlp_data.constraint_bounds[row]
            if bounds.lower == bounds.upper
                continue  # This is an equality constraint
            elseif isinf(bounds.lower)  # g(x) <= u --> g(x) - u <= 0
                result[index] = g_vec[row] - bounds.upper
            elseif isinf(bounds.upper)  # g(x) >= l --> l - g(x) <= 0
                result[index] = bounds.lower - g_vec[row]
            else  # l <= g(x) <= u
                result[index] = g_vec[row] - bounds.upper
                result[index+1] = bounds.lower - g_vec[row]
            end
        end
        offset = num_nlpblock_inequalities
        _fill_result(result, x, offset, model.linear_le_constraints)
        offset += length(model.linear_le_constraints)
        _fill_result(result, x, offset, model.quadratic_le_constraints)
        return
    end
    num_inequality_constraints =
        num_nlpblock_inequalities +
        length(model.linear_le_constraints) +
        length(model.quadratic_le_constraints)
    if num_inequality_constraints > 0
        NLopt.inequality_constraint!(
            model.inner,
            num_inequality_constraints,
            inequality_constraint_fn,
            model.options["constrtol_abs"],
        )
    end
    # Set MOI.VariablePrimalStart, clamping to bound nearest 0 if not given.
    model.solution = something.(
        model.starting_values,
        clamp.(0.0, model.variables.lower, model.variables.upper),
    )
    start_time = time()
    model.objective_value, _, model.status =
        NLopt.optimize!(model.inner, model.solution)
    model.solve_time = time() - start_time
    return
end

const _STATUS_MAP = Dict(
    :NOT_CALLED => (MOI.OPTIMIZE_NOT_CALLED, MOI.NO_SOLUTION),
    # The order here matches the nlopt_result enum
    :FAILURE => (MOI.OTHER_ERROR, MOI.UNKNOWN_RESULT_STATUS),
    :INVALID_ARGS => (MOI.INVALID_OPTION, MOI.UNKNOWN_RESULT_STATUS),
    :OUT_OF_MEMORY => (MOI.MEMORY_LIMIT, MOI.UNKNOWN_RESULT_STATUS),
    :ROUNDOFF_LIMITED =>
        (MOI.ALMOST_LOCALLY_SOLVED, MOI.NEARLY_FEASIBLE_POINT),
    :FORCED_STOP => (MOI.OTHER_ERROR, MOI.UNKNOWN_RESULT_STATUS),
    :SUCCESS => (MOI.LOCALLY_SOLVED, MOI.FEASIBLE_POINT),
    :STOPVAL_REACHED => (MOI.OBJECTIVE_LIMIT, MOI.UNKNOWN_RESULT_STATUS),
    :FTOL_REACHED => (MOI.LOCALLY_SOLVED, MOI.FEASIBLE_POINT),
    :XTOL_REACHED => (MOI.LOCALLY_SOLVED, MOI.FEASIBLE_POINT),
    :MAXEVAL_REACHED => (MOI.ITERATION_LIMIT, MOI.UNKNOWN_RESULT_STATUS),
    :MAXTIME_REACHED => (MOI.TIME_LIMIT, MOI.UNKNOWN_RESULT_STATUS),
)

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    return _STATUS_MAP[model.status][1]
end

MOI.get(model::Optimizer, ::MOI.RawStatusString) = string(model.status)

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.status == :NOT_CALLED ? 0 : 1
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    return _STATUS_MAP[model.status][2]
end

MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.objective_value
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    return model.solution[vi.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return MOI.Utilities.get_fallback(model, attr, ci)
end

end # module
