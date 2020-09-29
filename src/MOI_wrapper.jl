import MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

mutable struct VariableInfo
    lower_bound::Float64  # May be -Inf even if has_lower_bound == true
    has_lower_bound::Bool # Implies lower_bound == Inf
    upper_bound::Float64  # May be Inf even if has_upper_bound == true
    has_upper_bound::Bool # Implies upper_bound == Inf
    is_fixed::Bool        # Implies lower_bound == upper_bound and !has_lower_bound and !has_upper_bound.
    start::Union{Nothing, Float64}
end

VariableInfo() = VariableInfo(-Inf, false, Inf, false, false, nothing)

mutable struct ConstraintInfo{F, S}
    func::F
    set::S
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::Union{Opt,Nothing}

    # Problem data.
    variable_info::Vector{VariableInfo}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{MOI.SingleVariable,MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64},Nothing}
    linear_le_constraints::Vector{ConstraintInfo{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}}
    linear_eq_constraints::Vector{ConstraintInfo{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}}
    quadratic_le_constraints::Vector{ConstraintInfo{MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64}}}
    quadratic_eq_constraints::Vector{ConstraintInfo{MOI.ScalarQuadraticFunction{Float64}, MOI.EqualTo{Float64}}}

    # Parameters.
    silent::Bool
    options::Dict{String, Any}

    # Solution attributes.
    objective_value::Float64
    solution::Vector{Float64}
    status::Symbol
    solve_time::Float64
end

struct EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end
MOI.features_available(::EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::EmptyNLPEvaluator, features) = nothing
MOI.eval_objective(::EmptyNLPEvaluator, x) = NaN
function MOI.eval_constraint(::EmptyNLPEvaluator, g, x)
    @assert length(g) == 0
    return
end
function MOI.eval_objective_gradient(::EmptyNLPEvaluator, g, x)
    fill!(g, 0.0)
    return
end
MOI.jacobian_structure(::EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.hessian_lagrangian_structure(::EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
function MOI.eval_constraint_jacobian(::EmptyNLPEvaluator, J, x)
    return
end
function MOI.eval_hessian_lagrangian(::EmptyNLPEvaluator, H, x, σ, μ)
    @assert length(H) == 0
    return
end


empty_nlp_data() = MOI.NLPBlockData([], EmptyNLPEvaluator(), false)

const DEFAULT_OPTIONS = Dict{String, Any}(
    "algorithm" => :none,
    "stopval" => NaN,
    "ftol_rel" => 1e-7,
    "ftol_abs" => NaN,
    "xtol_rel" => 1e-7,
    "xtol_abs" => nothing,
    "constrtol_abs" => 1e-7,
    "maxeval" => 0,
    "maxtime" => 0,
    "initial_step" => nothing,
    "population" => 0,
    "seed" => nothing,
    "vector_storage" => 0,
)

function Optimizer()
    return Optimizer(nothing, VariableInfo[], empty_nlp_data(), MOI.FEASIBILITY_SENSE,
                     nothing, [], [], [], [],
                     false, copy(DEFAULT_OPTIONS), NaN, Float64[], :NOT_CALLED, NaN)
end

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

function MOI.supports(::Optimizer,
                      ::MOI.ObjectiveFunction{MOI.SingleVariable})
    return true
end

function MOI.supports(::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}})
    return true
end

function MOI.supports(::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}})
    return true
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

MOI.supports(::Optimizer, ::MOI.Silent) = true

MOI.supports(::Optimizer, p::MOI.RawParameter) = p.name in DEFAULT_OPTIONS

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart,
                      ::Type{MOI.VariableIndex})
    return true
end

MOI.supports_constraint(::Optimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.EqualTo{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true

MOIU.supports_default_copy_to(::Optimizer, copy_names::Bool) = !copy_names

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike; copy_names = false)
    return MOIU.default_copy_to(model, src, copy_names)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "NLopt"

MOI.get(model::Optimizer, ::MOI.ObjectiveFunctionType) = typeof(model.objective)

MOI.get(model::Optimizer, ::MOI.NumberOfVariables) = length(model.variable_info)

MOI.get(model::Optimizer, ::MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}) = length(model.linear_le_constraints)
MOI.get(model::Optimizer, ::MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}) = length(model.linear_eq_constraints)
MOI.get(model::Optimizer, ::MOI.NumberOfConstraints{MOI.SingleVariable, MOI.LessThan{Float64}}) = count(e -> e.has_upper_bound, model.variable_info)
MOI.get(model::Optimizer, ::MOI.NumberOfConstraints{MOI.SingleVariable, MOI.EqualTo{Float64}}) = count(e -> e.is_fixed, model.variable_info)
MOI.get(model::Optimizer, ::MOI.NumberOfConstraints{MOI.SingleVariable, MOI.GreaterThan{Float64}}) = count(e -> e.has_lower_bound, model.variable_info)

function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return [MOI.VariableIndex(i) for i in 1:length(model.variable_info)]
end


function MOI.get(model::Optimizer, ::MOI.ListOfConstraints)
    constraints = Set{Tuple{DataType, DataType}}()
    for info in model.variable_info
        if info.has_lower_bound
            push!(constraints, (MOI.SingleVariable, MOI.LessThan{Float64}))
        end
        if info.has_upper_bound
            push!(constraints, (MOI.SingleVariable, MOI.GreaterThan{Float64}))
        end
        if info.is_fixed
            push!(constraints, (MOI.SingleVariable, MOI.EqualTo{Float64}))
        end
    end

    # handling model constraints separately
    if !isempty(model.linear_le_constraints)
        push!(constraints, (MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}))
    end
    if !isempty(model.linear_eq_constraints)
        push!(constraints, (MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}))
    end
    if !isempty(model.quadratic_le_constraints)
        push!(constraints, (MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64}))
    end
    if !isempty(model.quadratic_eq_constraints)
        push!(constraints, (MOI.ScalarQuadraticFunction{Float64}, MOI.EqualTo{Float64}))
    end

    return collect(constraints)
end


function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
)
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}.(eachindex(model.linear_le_constraints))
end


function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}
)
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}.(eachindex(model.linear_eq_constraints))
end


function MOI.get(model::Optimizer, ::MOI.ListOfConstraintIndices{MOI.SingleVariable, MOI.LessThan{Float64}})
    dict = Dict(model.variable_info[i] => i for i in 1:length(model.variable_info))
    filter!(info -> info.first.has_upper_bound, dict)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}.(values(dict))
end

function MOI.get(model::Optimizer, ::MOI.ListOfConstraintIndices{MOI.SingleVariable, MOI.EqualTo{Float64}})
    dict = Dict(model.variable_info[i] => i for i in 1:length(model.variable_info))
    filter!(info -> info.first.is_fixed, dict)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}.(values(dict))
end

function MOI.get(model::Optimizer, ::MOI.ListOfConstraintIndices{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    dict = Dict(model.variable_info[i] => i for i in 1:length(model.variable_info))
    filter!(info -> info.first.has_lower_bound, dict)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}.(values(dict))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
)
    return model.linear_le_constraints[c.value].func
end


function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}
)
    return model.linear_eq_constraints[c.value].func
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}
)
    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}
)
    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}
)
    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
)
    return model.linear_le_constraints[c.value].set
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}
)
    return model.linear_eq_constraints[c.value].set
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}
)
    return MOI.LessThan{Float64}(model.variable_info[c.value].upper_bound)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}
)
    return MOI.EqualTo{Float64}(model.variable_info[c.value].lower_bound)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}
)
    return MOI.GreaterThan{Float64}(model.variable_info[c.value].lower_bound)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ObjectiveFunction
)
    return model.objective
end


function MOI.set(model::Optimizer, ::MOI.ObjectiveSense,
                 sense::MOI.OptimizationSense)
    model.sense = sense
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

const TIME_LIMIT = "max_cpu_time"
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    MOI.set(model, MOI.RawParameter(TIME_LIMIT), Float64(value))
end
function MOI.set(model::Optimizer, attr::MOI.TimeLimitSec, ::Nothing)
    delete!(model.options, TIME_LIMIT)
end
function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return get(model.options, TIME_LIMIT, nothing)
end


function MOI.set(model::Optimizer, p::MOI.RawParameter, value)
    model.options[p.name] = value
    return
end

function MOI.get(model::Optimizer, p::MOI.RawParameter)
    if haskey(model.options, p.name)
        return model.options[p.name]
    end
    error("RawParameter with name $(p.name) is not set.")
end

MOI.get(model::Optimizer, ::MOI.SolveTime) = model.solve_time

function MOI.empty!(model::Optimizer)
    model.inner = nothing
    empty!(model.variable_info)
    model.nlp_data = empty_nlp_data()
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective = nothing
    empty!(model.linear_le_constraints)
    empty!(model.linear_eq_constraints)
    empty!(model.quadratic_le_constraints)
    empty!(model.quadratic_eq_constraints)
end

function MOI.is_empty(model::Optimizer)
    return isempty(model.variable_info) &&
           model.nlp_data.evaluator isa EmptyNLPEvaluator &&
           model.sense == MOI.FEASIBILITY_SENSE &&
           isempty(model.linear_le_constraints) &&
           isempty(model.linear_eq_constraints) &&
           isempty(model.quadratic_le_constraints) &&
           isempty(model.quadratic_eq_constraints)
end

function MOI.add_variable(model::Optimizer)
    push!(model.variable_info, VariableInfo())
    return MOI.VariableIndex(length(model.variable_info))
end
function MOI.add_variables(model::Optimizer, n::Int)
    return [MOI.add_variable(model) for i in 1:n]
end
MOI.is_valid(model::Optimizer, vi::MOI.VariableIndex) = vi.value in eachindex(model.variable_info)
function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    return MOI.is_valid(model, vi) && has_upper_bound(model, vi)
end
function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    return MOI.is_valid(model, vi) && has_lower_bound(model, vi)
end
function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}})
    vi = MOI.VariableIndex(ci.value)
    return MOI.is_valid(model, vi) && is_fixed(model, vi)
end

function check_inbounds(model::Optimizer, var::MOI.SingleVariable)
    return MOI.throw_if_not_valid(model, var.variable)
end

function check_inbounds(model::Optimizer, aff::MOI.ScalarAffineFunction)
    for term in aff.terms
        MOI.throw_if_not_valid(model, term.variable_index)
    end
end

function check_inbounds(model::Optimizer, quad::MOI.ScalarQuadraticFunction)
    for term in quad.affine_terms
        MOI.throw_if_not_valid(model, term.variable_index)
    end
    for term in quad.quadratic_terms
        MOI.throw_if_not_valid(model, term.variable_index_1)
        MOI.throw_if_not_valid(model, term.variable_index_2)
    end
end

function has_upper_bound(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].has_upper_bound
end

function has_lower_bound(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].has_lower_bound
end

function is_fixed(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].is_fixed
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, lt::MOI.LessThan{Float64})
    vi = v.variable
    MOI.throw_if_not_valid(model, vi)
    if isnan(lt.upper)
        error("Invalid upper bound value $(lt.upper).")
    end
    if has_upper_bound(model, vi)
        throw(MOI.UpperBoundAlreadySet{typeof(lt), typeof(lt)}(vi))
    end
    if is_fixed(model, vi)
        throw(MOI.UpperBoundAlreadySet{MOI.EqualTo{Float64}, typeof(lt)}(vi))
    end
    model.variable_info[vi.value].upper_bound = lt.upper
    model.variable_info[vi.value].has_upper_bound = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}(vi.value)
end

function MOI.set(model::Optimizer, ::MOI.ConstraintSet,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}},
                 set::MOI.LessThan{Float64})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].upper_bound = set.upper
    return
end

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].upper_bound = Inf
    model.variable_info[ci.value].has_upper_bound = false
    return
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, gt::MOI.GreaterThan{Float64})
    vi = v.variable
    MOI.throw_if_not_valid(model, vi)
    if isnan(gt.lower)
        error("Invalid lower bound value $(gt.lower).")
    end
    if has_lower_bound(model, vi)
        throw(MOI.LowerBoundAlreadySet{typeof(gt), typeof(gt)}(vi))
    end
    if is_fixed(model, vi)
        throw(MOI.LowerBoundAlreadySet{MOI.EqualTo{Float64}, typeof(gt)}(vi))
    end
    model.variable_info[vi.value].lower_bound = gt.lower
    model.variable_info[vi.value].has_lower_bound = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}(vi.value)
end

function MOI.set(model::Optimizer, ::MOI.ConstraintSet,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}},
                 set::MOI.GreaterThan{Float64})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = set.lower
    return
end

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = -Inf
    model.variable_info[ci.value].has_lower_bound = false
    return
end

function _fill_jacobian_terms(jac::Matrix, x, row, terms::Vector{MOI.ScalarAffineTerm{Float64}})
    for term in terms
        jac[row, term.variable_index.value] += term.coefficient
    end
end
function _fill_jacobian_terms(jac::Matrix, x, row, terms::Vector{MOI.ScalarQuadraticTerm{Float64}})
    for term in terms
        jac[row, term.variable_index_1.value] += term.coefficient * x[term.variable_index_2.value]
        if term.variable_index_1 != term.variable_index_2
            jac[row, term.variable_index_2.value] += term.coefficient * x[term.variable_index_1.value]
        end
    end
end
function _fill_jacobian(jac::Matrix, x, offset, constraints::Vector{<:ConstraintInfo{MOI.ScalarAffineFunction{Float64}}})
    for (k, con) in enumerate(constraints)
        _fill_jacobian_terms(jac, x, offset + k, con.func.terms)
    end
end
function _fill_jacobian(jac::Matrix, x, offset, constraints::Vector{<:ConstraintInfo{MOI.ScalarQuadraticFunction{Float64}}})
    for (k, con) in enumerate(constraints)
        _fill_jacobian_terms(jac, x, offset + k, con.func.affine_terms)
        _fill_jacobian_terms(jac, x, offset + k, con.func.quadratic_terms)
    end
end

function _fill_result(result::Vector, x, offset, constraints::Vector)
    for (k, con) in enumerate(constraints)
        result[offset + k] = MOI.Utilities.eval_variables(vi -> x[vi.value], con.func) - MOI.constant(con.set)
    end
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, eq::MOI.EqualTo{Float64})
    vi = v.variable
    MOI.throw_if_not_valid(model, vi)
    if isnan(eq.value)
        error("Invalid fixed value $(gt.lower).")
    end
    if has_lower_bound(model, vi)
        throw(MOI.LowerBoundAlreadySet{MOI.GreaterThan{Float64}, typeof(eq)}(vi))
    end
    if has_upper_bound(model, vi)
        throw(MOI.UpperBoundAlreadySet{MOI.LessThan{Float64}, typeof(eq)}(vi))
    end
    if is_fixed(model, vi)
        throw(MOI.LowerBoundAlreadySet{typeof(eq), typeof(eq)}(vi))
    end
    model.variable_info[vi.value].lower_bound = eq.value
    model.variable_info[vi.value].upper_bound = eq.value
    model.variable_info[vi.value].is_fixed = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}(vi.value)
end

function MOI.set(model::Optimizer, ::MOI.ConstraintSet,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}},
                 set::MOI.EqualTo{Float64})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = set.value
    model.variable_info[ci.value].upper_bound = set.value
    return
end

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = -Inf
    model.variable_info[ci.value].upper_bound = Inf
    model.variable_info[ci.value].is_fixed = false
    return
end

macro define_add_constraint(function_type, set_type, prefix)
    array_name = Symbol(string(prefix) * "_constraints")
    quote
        function MOI.add_constraint(model::Optimizer, func::$function_type, set::$set_type)
            check_inbounds(model, func)
            push!(model.$(array_name), ConstraintInfo(func, set))
            return MOI.ConstraintIndex{$function_type, $set_type}(length(model.$(array_name)))
        end
    end
end

@define_add_constraint(MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64},
                       linear_le)
@define_add_constraint(MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64},
                       linear_eq)
@define_add_constraint(MOI.ScalarQuadraticFunction{Float64},
                       MOI.LessThan{Float64}, quadratic_le)
@define_add_constraint(MOI.ScalarQuadraticFunction{Float64},
                       MOI.EqualTo{Float64}, quadratic_eq)

function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart,
                 vi::MOI.VariableIndex, value::Union{Real, Nothing})
    MOI.throw_if_not_valid(model, vi)
    model.variable_info[vi.value].start = value
    return
end

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    model.nlp_data = nlp_data
    return
end

function MOI.set(model::Optimizer, ::MOI.ObjectiveFunction,
                 func::Union{MOI.SingleVariable, MOI.ScalarAffineFunction,
                             MOI.ScalarQuadraticFunction})
    check_inbounds(model, func)
    model.objective = func
    return
end

function eval_objective(model::Optimizer, x)
    # The order of the conditions is important. NLP objectives override regular
    # objectives.
    if model.nlp_data.has_objective
        return MOI.eval_objective(model.nlp_data.evaluator, x)
    elseif model.objective !== nothing
        return eval_function(model.objective, x)
    else
        # No objective function set. This could happen with FEASIBILITY_SENSE.
        return 0.0
    end
end

function fill_gradient!(grad, x, var::MOI.SingleVariable)
    fill!(grad, 0.0)
    grad[var.variable.value] = 1.0
end

function fill_gradient!(grad, x, aff::MOI.ScalarAffineFunction{Float64})
    fill!(grad, 0.0)
    for term in aff.terms
        grad[term.variable_index.value] += term.coefficient
    end
end

function fill_gradient!(grad, x, quad::MOI.ScalarQuadraticFunction{Float64})
    fill!(grad, 0.0)
    for term in quad.affine_terms
        grad[term.variable_index.value] += term.coefficient
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_index_1
        col_idx = term.variable_index_2
        coefficient = term.coefficient
        if row_idx == col_idx
            grad[row_idx.value] += coefficient*x[row_idx.value]
        else
            grad[row_idx.value] += coefficient*x[col_idx.value]
            grad[col_idx.value] += coefficient*x[row_idx.value]
        end
    end
end

function eval_objective_gradient(model::Optimizer, grad, x)
    if model.nlp_data.has_objective
        MOI.eval_objective_gradient(model.nlp_data.evaluator, grad, x)
    elseif model.objective !== nothing
        fill_gradient!(grad, x, model.objective)
    else
        fill!(grad, 0.0)
    end
    return
end

# Refers to local variables in eval_constraint() below.
macro eval_function(array_name)
    escrow = esc(:row)
    quote
        for info in $(esc(array_name))
            $(esc(:g))[$escrow] = eval_function(info.func, $(esc(:x)))
            $escrow += 1
        end
    end
end

function eval_constraint(model::Optimizer, g, x)
    row = 1
    @eval_function model.linear_le_constraints
    @eval_function model.linear_eq_constraints
    @eval_function model.quadratic_le_constraints
    @eval_function model.quadratic_eq_constraints
    nlp_g = view(g, row:length(g))
    MOI.eval_constraint(model.nlp_data.evaluator, nlp_g, x)
    return
end

function fill_constraint_jacobian!(values, start_offset, x, aff::MOI.ScalarAffineFunction)
    num_coefficients = length(aff.terms)
    for i in 1:num_coefficients
        values[start_offset+i] = aff.terms[i].coefficient
    end
    return num_coefficients
end

function fill_constraint_jacobian!(values, start_offset, x, quad::MOI.ScalarQuadraticFunction)
    num_affine_coefficients = length(quad.affine_terms)
    for i in 1:num_affine_coefficients
        values[start_offset+i] = quad.affine_terms[i].coefficient
    end
    num_quadratic_coefficients = 0
    for term in quad.quadratic_terms
        row_idx = term.variable_index_1
        col_idx = term.variable_index_2
        coefficient = term.coefficient
        if row_idx == col_idx
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+1] = coefficient*x[col_idx.value]
            num_quadratic_coefficients += 1
        else
            # Note that the order matches the Jacobian sparsity pattern.
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+1] = coefficient*x[col_idx.value]
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+2] = coefficient*x[row_idx.value]
            num_quadratic_coefficients += 2
        end
    end
    return num_affine_coefficients + num_quadratic_coefficients
end

# Refers to local variables in eval_constraint_jacobian() below.
macro fill_constraint_jacobian(array_name)
    esc_offset = esc(:offset)
    quote
        for info in $(esc(array_name))
            $esc_offset += fill_constraint_jacobian!($(esc(:values)),
                                                     $esc_offset, $(esc(:x)),
                                                     info.func)
        end
    end
end

function eval_constraint_jacobian(model::Optimizer, values, x)
    offset = 0
    @fill_constraint_jacobian model.linear_le_constraints
    @fill_constraint_jacobian model.linear_eq_constraints
    @fill_constraint_jacobian model.quadratic_le_constraints
    @fill_constraint_jacobian model.quadratic_eq_constraints

    nlp_values = view(values, 1+offset:length(values))
    MOI.eval_constraint_jacobian(model.nlp_data.evaluator, nlp_values, x)
    return
end

function fill_hessian_lagrangian!(values, start_offset, scale_factor,
                                  ::Union{MOI.SingleVariable,
                                          MOI.ScalarAffineFunction,Nothing})
    return 0
end

function fill_hessian_lagrangian!(values, start_offset, scale_factor,
                                  quad::MOI.ScalarQuadraticFunction)
    for i in 1:length(quad.quadratic_terms)
        values[start_offset + i] = scale_factor*quad.quadratic_terms[i].coefficient
    end
    return length(quad.quadratic_terms)
end

function eval_hessian_lagrangian(model::Optimizer, values, x, obj_factor, lambda)
    offset = 0
    if !model.nlp_data.has_objective
        offset += fill_hessian_lagrangian!(values, 0, obj_factor,
                                          model.objective)
    end
    for (i, info) in enumerate(model.quadratic_le_constraints)
        offset += fill_hessian_lagrangian!(values, offset, lambda[i+quadratic_le_offset(model)], info.func)
    end
    for (i, info) in enumerate(model.quadratic_eq_constraints)
        offset += fill_hessian_lagrangian!(values, offset, lambda[i+quadratic_eq_offset(model)], info.func)
    end
    nlp_values = view(values, 1 + offset : length(values))
    nlp_lambda = view(lambda, 1 + nlp_constraint_offset(model) : length(lambda))
    MOI.eval_hessian_lagrangian(model.nlp_data.evaluator, nlp_values, x, obj_factor, nlp_lambda)
end

function constraint_bounds(model::Optimizer)
    constraint_lb = Float64[]
    constraint_ub = Float64[]
    for info in model.linear_le_constraints
        push!(constraint_lb, -Inf)
        push!(constraint_ub, info.set.upper)
    end
    for info in model.linear_eq_constraints
        push!(constraint_lb, info.set.value)
        push!(constraint_ub, info.set.value)
    end
    for info in model.quadratic_le_constraints
        push!(constraint_lb, -Inf)
        push!(constraint_ub, info.set.upper)
    end
    for info in model.quadratic_eq_constraints
        push!(constraint_lb, info.set.value)
        push!(constraint_ub, info.set.value)
    end
    for bound in model.nlp_data.constraint_bounds
        push!(constraint_lb, bound.lower)
        push!(constraint_ub, bound.upper)
    end
    return constraint_lb, constraint_ub
end

function MOI.optimize!(model::Optimizer)
    # TODO: Reuse model.inner for incremental solves if possible.
    num_variables = length(model.variable_info)
    model.inner = Opt(model.options["algorithm"], num_variables)

    # load parameters
    stopval!(model.inner, model.options["stopval"])
    if !isnan(model.options["ftol_rel"])
        ftol_rel!(model.inner, model.options["ftol_rel"])
    end
    if !isnan(model.options["ftol_abs"])
        ftol_abs!(model.inner, model.options["ftol_abs"])
    end
    if !isnan(model.options["xtol_rel"])
        xtol_rel!(model.inner, model.options["xtol_rel"])
    end
    if model.options["xtol_abs"] != nothing
        xtol_abs!(model.inner, model.options["xtol_abs"])
    end
    maxeval!(model.inner, model.options["maxeval"])
    maxtime!(model.inner, model.options["maxtime"])
    if model.options["initial_step"] != nothing
        initial_step!(model.inner, model.options["initial_step"])
    end
    population!(model.inner, model.options["population"])
    if isa(model.options["seed"], Integer)
        NLopt.srand(model.options["seed"])
    end
    vector_storage!(model.inner, model.options["vector_storage"])

    lower_bounds!(model.inner, [model.variable_info[i].lower_bound for i in 1:num_variables])
    upper_bounds!(model.inner, [model.variable_info[i].upper_bound for i in 1:num_variables])

    nleqidx = findall(bound -> bound.lower == bound.upper, model.nlp_data.constraint_bounds) # indices of equalities
    nlineqidx = findall(bound -> bound.lower != bound.upper, model.nlp_data.constraint_bounds)

    num_nl_constraints = length(model.nlp_data.constraint_bounds)

    # map from eqidx/ineqidx to index in equalities/inequalities
    constrmap = zeros(Int, num_nl_constraints)
    for i in eachindex(nleqidx)
        constrmap[nleqidx[i]] = i
    end
    ineqcounter = 1
    for i in eachindex(nlineqidx)
        k = nlineqidx[i]
        constrmap[k] = ineqcounter
        bounds = model.nlp_data.constraint_bounds[k]
        if isinf(bounds.lower) || isinf(bounds.upper)
            ineqcounter += 1
        else # constraint has bounds on both sides, keep room for it
            ineqcounter += 2
        end
    end
    num_nl_ineq = ineqcounter - 1
    num_nl_eq = length(nleqidx)

    isderivativefree = string(model.options["algorithm"])[2] == 'N'
    if isderivativefree
        requested_features = Symbol[]
    else
        requested_features = num_nl_constraints > 0 ? [:Grad, :Jac] : [:Grad]
    end

    MOI.initialize(model.nlp_data.evaluator, requested_features)

    if model.sense == MOI.FEASIBILITY_SENSE
        # If we don't give any objective to NLopt, it throws the error:
        # invalid NLopt arguments: NULL args to nlopt_optimize
        function z(x::Vector, grad::Vector)
            fill!(grad, 0.0)
            return 0.0
        end
        min_objective!(model.inner, z)
    else
        function f(x::Vector, grad::Vector)
            if length(grad) > 0
                eval_objective_gradient(model, grad, x)
            end
            return eval_objective(model, x)
        end
        if model.sense == MOI.MIN_SENSE
            min_objective!(model.inner, f)
        else
            max_objective!(model.inner, f)
        end
    end

    Jac_IJ = num_nl_constraints > 0 ? MOI.jacobian_structure(model.nlp_data.evaluator) : (Int[], Int[])
    Jac_val = zeros(length(Jac_IJ))
    g_vec = zeros(num_nl_constraints)
    MOI.eval_constraint_jacobian(model.nlp_data.evaluator, Jac_val, zeros(num_variables))

    if num_nl_eq > 0
        function g_eq(result::Vector, x::Vector, jac::Matrix)
            if length(jac) > 0
                fill!(jac, 0.0)
                MOI.eval_constraint_jacobian(model.nlp_data.evaluator, Jac_val, x)
                for k in 1:length(Jac_val)
                    row, col = Jac_IJ[k]
                    bounds = model.nlp_data.constraint_bounds[row]
                    if bounds.lower == bounds.upper
                        jac[col, constrmap[row]] += Jac_val[k]
                    end
                end
                _fill_jacobian(jac, x, num_nl_eq, model.linear_eq_constraints)
                _fill_jacobian(jac, x, num_nl_eq + length(model.linear_eq_constraints), model.quadratic_eq_constraints)
            end
            MOI.eval_constraint(model.nlp_data.evaluator, g_vec, x)
            for (ctr, idx) in enumerate(nleqidx)
                bounds = model.nlp_data.constraint_bounds[idx]
                result[ctr] = g_vec[idx] - bounds.upper
            end
            _fill_result(result, x, num_nl_eq, model.linear_eq_constraints)
            _fill_result(result, x, num_nl_eq + length(model.linear_eq_constraints), model.quadratic_eq_constraints)
        end

        equality_constraint!(model.inner, g_eq, fill(model.options["constrtol_abs"], num_nl_eq + length(model.linear_eq_constraints) + length(model.quadratic_eq_constraints)))
    end

    # inequalities need to be massaged a bit
    # f(x) <= u   =>  f(x) - u <= 0
    # f(x) >= l   =>  l - f(x) <= 0

    if num_nl_ineq > 0
        function g_ineq(result::Vector, x::Vector, jac::Matrix)
            if length(jac) > 0
                fill!(jac, 0.0)
                MOI.eval_constraint_jacobian(model.nlp_data.evaluator, Jac_val, x)
                for k in 1:length(Jac_val)
                    row, col = Jac_IJ[k]
                    bounds = model.nlp_data.constraint_bounds[row]
                    bounds.lower == bounds.upper && continue
                    if isinf(bounds.lower) # upper bound
                        jac[col, constrmap[row]] += Jac_val[k]
                    elseif isinf(bounds.upper) # lower bound
                        jac[col, constrmap[row]] -= Jac_val[k]
                    else
                        # boxed
                        jac[col, constrmap[row]] += Jac_val[k]
                        jac[col, constrmap[row] + 1] -= Jac_val[k]
                    end
                end
                _fill_jacobian(jac, x, num_nl_ineq, model.linear_le_constraints)
                _fill_jacobian(jac, x, num_nl_ineq + length(model.linear_le_constraints), model.quadratic_le_constraints)
            end
            MOI.eval_constraint(model.nlp_data.evaluator, g_vec, x)
            for row in 1:num_nl_constraints
                bounds = model.nlp_data.constraint_bounds[row]
                bounds.lower == bounds.upper && continue
                if isinf(bounds.lower)
                    result[constrmap[row]] = g_vec[row] - bounds.upper
                elseif isinf(bounds.upper)
                    result[constrmap[row]] = bounds.lower - g_vec[row]
                else
                    result[constrmap[row]] = g_vec[row] - bounds.upper
                    result[constrmap[row] + 1] = bounds.lower - g_vec[row]
                end
            end
            _fill_result(result, x, num_nl_ineq, model.linear_le_constraints)
            _fill_result(result, x, num_nl_ineq + length(model.linear_le_constraints), model.quadratic_le_constraints)
        end

        inequality_constraint!(model.inner, g_ineq, fill(model.options["constrtol_abs"], num_nl_ineq + length(model.linear_le_constraints) + length(model.quadratic_le_constraints)))
    end

    # If nothing is provided, the default starting value is 0.0.
    model.solution = zeros(num_variables)
    for (i, v) in enumerate(model.variable_info)
        if v.start !== nothing
            model.solution[i] = v.start
        else
            if v.has_lower_bound && v.has_upper_bound
                if 0.0 <= v.lower_bound
                    model.solution[i] = v.lower_bound
                elseif v.upper_bound <= 0.0
                    model.solution[i] = v.upper_bound
                end
            elseif v.has_lower_bound
                model.solution[i] = max(0.0, v.lower_bound)
            else
                model.solution[i] = min(0.0, v.upper_bound)
            end
        end
    end

    start_time = time()

    model.objective_value, _, model.status = optimize!(model.inner, model.solution)

    model.solve_time = time() - start_time
    return
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.status == :NOT_CALLED
        return MOI.OPTIMIZE_NOT_CALLED
    elseif model.status == :SUCCESS || model.status == :FTOL_REACHED || model.status == :XTOL_REACHED
        return MOI.LOCALLY_SOLVED
    elseif model.status == :ROUNDOFF_LIMITED
        return MOI.ALMOST_LOCALLY_SOLVED
    elseif model.status == :MAXEVAL_REACHED
        return MOI.ITERATION_LIMIT
    elseif model.status == :MAXTIME_REACHED
        return MOI.TIME_LIMIT
    elseif model.status == :STOPVAL_REACHED || model.status == :FORCED_STOP
        return MOI.OTHER_LIMIT
    else
        error("Unknown status $(model.status)")
    end
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    return string(model.status)
end

# Ipopt always has an iterate available.
function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.status == :NOT_CALLED ? 0 : 1
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.N <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    if model.status == :SUCCESS || model.status == :FTOL_REACHED || model.status == :XTOL_REACHED
        return MOI.FEASIBLE_POINT
    elseif model.status == :ROUNDOFF_LIMITED
        return MOI.NEARLY_FEASIBLE_POINT
    elseif model.status in (:STOPVAL_REACHED, :MAXEVAL_REACHED, :MAXTIME_REACHED, :FORCED_STOP)
        return MOI.UNKNOWN_RESULT_STATUS
    else
        error("Unknown status $(model.status)")
    end
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.objective_value
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    return model.solution[vi.value]
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,
                                         <:Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}}})
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.solution[ci.value]
end
