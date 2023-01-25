import MathOptInterface

const MOI = MathOptInterface

mutable struct _ConstraintInfo{F,S}
    func::F
    set::S
end

"""
    Optimizer()

Create a new Optimizer object.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::Union{Opt,Nothing}
    # Problem data.
    variables::MOI.Utilities.VariablesContainer{Float64}
    starting_values::Vector{Union{Nothing,Float64}}
    nlp_data::MOI.NLPBlockData
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
    constraint_primal_linear_le::Vector{Float64}
    constraint_primal_linear_eq::Vector{Float64}
    constraint_primal_quadratic_le::Vector{Float64}
    constraint_primal_quadratic_eq::Vector{Float64}
    status::Symbol
    solve_time::Float64

    function Optimizer()
        return new(
            nothing,
            MOI.Utilities.VariablesContainer{Float64}(),
            Union{Nothing,Float64}[],
            empty_nlp_data(),
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
            copy(DEFAULT_OPTIONS),
            NaN,
            Float64[],
            Float64[],
            Float64[],
            Float64[],
            Float64[],
            :NOT_CALLED,
            NaN,
        )
    end
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

function MOI.eval_constraint_jacobian(::EmptyNLPEvaluator, J, x)
    return
end

empty_nlp_data() = MOI.NLPBlockData([], EmptyNLPEvaluator(), false)

# empty! and is_empty

function MOI.empty!(model::Optimizer)
    model.inner = nothing
    MOI.empty!(model.variables)
    empty!(model.starting_values)
    model.nlp_data = empty_nlp_data()
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
           model.nlp_data.evaluator isa EmptyNLPEvaluator &&
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

const _F_TYPES = Union{
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
}

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
    ::Type{<:_F_TYPES},
    ::Type{<:Union{MOI.LessThan{Float64},MOI.EqualTo{Float64}}},
)
    return true
end

function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{F,S},
) where {F<:_F_TYPES,S}
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
) where {F<:_F_TYPES,S}
    return MOI.ConstraintIndex{F,S}.(eachindex(_constraints(model, F, S)))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{F,S},
) where {F<:_F_TYPES,S}
    return _constraints(model, F, S)[c.value].func
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,S},
) where {F<:_F_TYPES,S}
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

function MOI.set(model::Optimizer, ::MOI.Silent, value)
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

const DEFAULT_OPTIONS = Dict{String,Any}(
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
    "local_optimizer" => nothing,
)

function MOI.supports(::Optimizer, p::MOI.RawOptimizerAttribute)
    return p.name in DEFAULT_OPTIONS
end

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    model.options[p.name] = value
    return
end

function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    if !haskey(model.options, p.name)
        error("RawOptimizerAttribute with name $(p.name) is not set.")
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

const _BOUNDS = Union{
    MOI.LessThan{Float64},
    MOI.GreaterThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{<:_BOUNDS},
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
    set::_BOUNDS,
)
    return MOI.add_constraint(model.variables, vi, set)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex},
    set,
)
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
) where {F,S}
    return ci.value in eachindex(_constraints(model, F, S))
end

function _check_inbounds(model::Optimizer, var::MOI.VariableIndex)
    return MOI.throw_if_not_valid(model, var)
end

function _check_inbounds(
    model::Optimizer,
    aff::MOI.ScalarAffineFunction{Float64},
)
    for term in aff.terms
        MOI.throw_if_not_valid(model, term.variable)
    end
    return
end

function _check_inbounds(
    model::Optimizer,
    quad::MOI.ScalarQuadraticFunction{Float64},
)
    for term in quad.affine_terms
        MOI.throw_if_not_valid(model, term.variable)
    end
    for term in quad.quadratic_terms
        MOI.throw_if_not_valid(model, term.variable_1)
        MOI.throw_if_not_valid(model, term.variable_2)
    end
    return
end

function _fill_jacobian_terms(
    jac::Matrix,
    x,
    row,
    terms::Vector{MOI.ScalarAffineTerm{Float64}},
)
    for term in terms
        jac[term.variable.value, row] += term.coefficient
    end
    return
end

function _fill_jacobian_terms(
    jac::Matrix,
    x,
    row,
    terms::Vector{MOI.ScalarQuadraticTerm{Float64}},
)
    for term in terms
        jac[term.variable_1.value, row] +=
            term.coefficient * x[term.variable_2.value]
        if term.variable_1 != term.variable_2
            jac[term.variable_2.value, row] +=
                term.coefficient * x[term.variable_1.value]
        end
    end
    return
end

function _fill_jacobian(
    jac::Matrix,
    x,
    offset,
    constraints::Vector{<:_ConstraintInfo{MOI.ScalarAffineFunction{Float64}}},
)
    for (k, con) in enumerate(constraints)
        _fill_jacobian_terms(jac, x, offset + k, con.func.terms)
    end
    return
end

function _fill_jacobian(
    jac::Matrix,
    x,
    offset,
    constraints::Vector{
        <:_ConstraintInfo{MOI.ScalarQuadraticFunction{Float64}},
    },
)
    for (k, con) in enumerate(constraints)
        _fill_jacobian_terms(jac, x, offset + k, con.func.affine_terms)
        _fill_jacobian_terms(jac, x, offset + k, con.func.quadratic_terms)
    end
    return
end

function _fill_result(result::Vector, x, offset, constraints::Vector)
    for (k, con) in enumerate(constraints)
        result[offset+k] =
            MOI.Utilities.eval_variables(vi -> x[vi.value], con.func) -
            MOI.constant(con.set)
    end
    return
end

function _fill_primal(x, constraints::Vector)
    result = zeros(length(constraints))
    for (k, con) in enumerate(constraints)
        result[k] = MOI.Utilities.eval_variables(vi -> x[vi.value], con.func)
    end
    return result
end

function MOI.add_constraint(
    model::Optimizer,
    func::_F_TYPES,
    set::Union{MOI.LessThan{Float64},MOI.EqualTo{Float64}},
)
    _check_inbounds(model, func)
    constraints = _constraints(model, typeof(func), typeof(set))
    push!(constraints, _ConstraintInfo(func, set))
    return MOI.ConstraintIndex{typeof(func),typeof(set)}(length(constraints))
end

function starting_value(optimizer::Optimizer, i)
    if optimizer.starting_values[i] !== nothing
        return optimizer.starting_values[i]
    else
        v = optimizer.variables
        return min(max(0.0, v.lower[i]), v.upper[i])
    end
end

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

MOI.get(model::Optimizer, ::MOI.ObjectiveFunctionType) = typeof(model.objective)

function MOI.get(model::Optimizer, ::MOI.ObjectiveFunction{F}) where {F}
    return convert(F, model.objective)::F
end

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveFunction,
    func::Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
)
    _check_inbounds(model, func)
    model.objective = func
    return
end

function eval_objective(model::Optimizer, x)
    # The order of the conditions is important. NLP objectives override regular
    # objectives.
    if model.nlp_data.has_objective
        return MOI.eval_objective(model.nlp_data.evaluator, x)
    elseif model.objective !== nothing
        return MOI.Utilities.eval_variables(vi -> x[vi.value], model.objective)
    else
        # No objective function set. This could happen with FEASIBILITY_SENSE.
        return 0.0
    end
end

function fill_gradient!(grad, x, var::MOI.VariableIndex)
    fill!(grad, 0.0)
    grad[var.value] = 1.0
    return retur
end

function fill_gradient!(grad, x, aff::MOI.ScalarAffineFunction{Float64})
    fill!(grad, 0.0)
    for term in aff.terms
        grad[term.variable.value] += term.coefficient
    end
    return
end

function fill_gradient!(grad, x, quad::MOI.ScalarQuadraticFunction{Float64})
    fill!(grad, 0.0)
    for term in quad.affine_terms
        grad[term.variable.value] += term.coefficient
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_1
        col_idx = term.variable_2
        coefficient = term.coefficient
        if row_idx == col_idx
            grad[row_idx.value] += coefficient * x[row_idx.value]
        else
            grad[row_idx.value] += coefficient * x[col_idx.value]
            grad[col_idx.value] += coefficient * x[row_idx.value]
        end
    end
    return
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

function eval_constraint(model::Optimizer, g, x)
    row = 1
    for info in model.linear_le_constraints
        g[row] = eval_function(info.func, x)
        row += 1
    end
    for info in model.linear_eq_constraints
        g[row] = eval_function(info.func, x)
        row += 1
    end
    for info in model.quadratic_le_constraints
        g[row] = eval_function(info.func, x)
        row += 1
    end
    for info in model.quadratic_eq_constraints
        g[row] = eval_function(info.func, x)
        row += 1
    end
    nlp_g = view(g, row:length(g))
    MOI.eval_constraint(model.nlp_data.evaluator, nlp_g, x)
    return
end

function fill_constraint_jacobian!(
    values,
    start_offset,
    x,
    aff::MOI.ScalarAffineFunction{Float64},
)
    num_coefficients = length(aff.terms)
    for i in 1:num_coefficients
        values[start_offset+i] = aff.terms[i].coefficient
    end
    return num_coefficients
end

function fill_constraint_jacobian!(
    values,
    start_offset,
    x,
    quad::MOI.ScalarQuadraticFunction{Float64},
)
    num_affine_coefficients = length(quad.affine_terms)
    for i in 1:num_affine_coefficients
        values[start_offset+i] = quad.affine_terms[i].coefficient
    end
    num_quadratic_coefficients = 0
    for term in quad.quadratic_terms
        row_idx = term.variable_1
        col_idx = term.variable_2
        coefficient = term.coefficient
        if row_idx == col_idx
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+1] =
                coefficient * x[col_idx.value]
            num_quadratic_coefficients += 1
        else
            # Note that the order matches the Jacobian sparsity pattern.
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+1] =
                coefficient * x[col_idx.value]
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+2] =
                coefficient * x[row_idx.value]
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
            $esc_offset += fill_constraint_jacobian!(
                $(esc(:values)),
                $esc_offset,
                $(esc(:x)),
                info.func,
            )
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
    num_variables = length(model.starting_values)
    model.inner = Opt(model.options["algorithm"], num_variables)
    local_optimizer = model.options["local_optimizer"]
    if local_optimizer !== nothing
        if local_optimizer isa Symbol
            local_optimizer = Opt(local_optimizer, num_variables)
        else
            local_optimizer = Opt(local_optimizer.algorithm, num_variables)
        end
        local_optimizer!(model.inner, local_optimizer)
    end
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
    lower_bounds!(model.inner, model.variables.lower)
    upper_bounds!(model.inner, model.variables.upper)
    nleqidx = findall(
        bound -> bound.lower == bound.upper,
        model.nlp_data.constraint_bounds,
    ) # indices of equalities
    nlineqidx = findall(
        bound -> bound.lower != bound.upper,
        model.nlp_data.constraint_bounds,
    )
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
    Jac_IJ =
        num_nl_constraints > 0 ?
        MOI.jacobian_structure(model.nlp_data.evaluator) : Tuple{Int,Int}[]
    Jac_val = zeros(length(Jac_IJ))
    g_vec = zeros(num_nl_constraints)
    num_eq =
        num_nl_eq +
        length(model.linear_eq_constraints) +
        length(model.quadratic_eq_constraints)
    if num_eq > 0
        function g_eq(result::Vector, x::Vector, jac::Matrix)
            if length(jac) > 0
                fill!(jac, 0.0)
                MOI.eval_constraint_jacobian(
                    model.nlp_data.evaluator,
                    Jac_val,
                    x,
                )
                for k in 1:length(Jac_val)
                    row, col = Jac_IJ[k]
                    bounds = model.nlp_data.constraint_bounds[row]
                    if bounds.lower == bounds.upper
                        jac[col, constrmap[row]] += Jac_val[k]
                    end
                end
                _fill_jacobian(jac, x, num_nl_eq, model.linear_eq_constraints)
                _fill_jacobian(
                    jac,
                    x,
                    num_nl_eq + length(model.linear_eq_constraints),
                    model.quadratic_eq_constraints,
                )
            end
            MOI.eval_constraint(model.nlp_data.evaluator, g_vec, x)
            for (ctr, idx) in enumerate(nleqidx)
                bounds = model.nlp_data.constraint_bounds[idx]
                result[ctr] = g_vec[idx] - bounds.upper
            end
            _fill_result(result, x, num_nl_eq, model.linear_eq_constraints)
            return _fill_result(
                result,
                x,
                num_nl_eq + length(model.linear_eq_constraints),
                model.quadratic_eq_constraints,
            )
        end
        g_eq(zeros(num_eq), zeros(num_variables), zeros(num_variables, num_eq))
        equality_constraint!(
            model.inner,
            g_eq,
            fill(model.options["constrtol_abs"], num_eq),
        )
    end
    # inequalities need to be massaged a bit
    # f(x) <= u   =>  f(x) - u <= 0
    # f(x) >= l   =>  l - f(x) <= 0
    num_ineq =
        num_nl_ineq +
        length(model.linear_le_constraints) +
        length(model.quadratic_le_constraints)
    if num_ineq > 0
        function g_ineq(result::Vector, x::Vector, jac::Matrix)
            if length(jac) > 0
                fill!(jac, 0.0)
                MOI.eval_constraint_jacobian(
                    model.nlp_data.evaluator,
                    Jac_val,
                    x,
                )
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
                        jac[col, constrmap[row]+1] -= Jac_val[k]
                    end
                end
                _fill_jacobian(jac, x, num_nl_ineq, model.linear_le_constraints)
                _fill_jacobian(
                    jac,
                    x,
                    num_nl_ineq + length(model.linear_le_constraints),
                    model.quadratic_le_constraints,
                )
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
                    result[constrmap[row]+1] = bounds.lower - g_vec[row]
                end
            end
            _fill_result(result, x, num_nl_ineq, model.linear_le_constraints)
            return _fill_result(
                result,
                x,
                num_nl_ineq + length(model.linear_le_constraints),
                model.quadratic_le_constraints,
            )
        end
        g_ineq(
            zeros(num_ineq),
            zeros(num_variables),
            zeros(num_variables, num_ineq),
        )
        inequality_constraint!(
            model.inner,
            g_ineq,
            fill(model.options["constrtol_abs"], num_ineq),
        )
    end
    # If nothing is provided, the default starting value is 0.0.
    model.solution = zeros(num_variables)
    for i in eachindex(model.starting_values)
        model.solution[i] = starting_value(model, i)
    end
    start_time = time()
    model.objective_value, _, model.status =
        optimize!(model.inner, model.solution)
    model.constraint_primal_linear_le =
        _fill_primal(model.solution, model.linear_le_constraints)
    model.constraint_primal_linear_eq =
        _fill_primal(model.solution, model.linear_eq_constraints)
    model.constraint_primal_quadratic_le =
        _fill_primal(model.solution, model.quadratic_le_constraints)
    model.constraint_primal_quadratic_eq =
        _fill_primal(model.solution, model.quadratic_eq_constraints)
    model.solve_time = time() - start_time
    return
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.status == :NOT_CALLED
        return MOI.OPTIMIZE_NOT_CALLED
    elseif model.status in (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED)
        return MOI.LOCALLY_SOLVED
    elseif model.status == :ROUNDOFF_LIMITED
        return MOI.ALMOST_LOCALLY_SOLVED
    elseif model.status == :MAXEVAL_REACHED
        return MOI.ITERATION_LIMIT
    elseif model.status == :MAXTIME_REACHED
        return MOI.TIME_LIMIT
    elseif model.status == :STOPVAL_REACHED
        return MOI.OTHER_LIMIT
    elseif model.status == :FORCED_STOP
        # May be due to an error in the callbacks `f`, `g_eq` and `g_ineq`
        # defined in `MOI.optimize!`
        return MOI.OTHER_ERROR
    elseif model.status == :OUT_OF_MEMORY
        return MOI.MEMORY_LIMIT
    elseif model.status == :INVALID_ARGS
        return MOI.INVALID_OPTION
    elseif model.status == :FAILURE
        return MOI.OTHER_ERROR
    end
    return MOI.OTHER_ERROR
end

MOI.get(model::Optimizer, ::MOI.RawStatusString) = string(model.status)

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.status == :NOT_CALLED ? 0 : 1
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    elseif model.status in (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED)
        return MOI.FEASIBLE_POINT
    elseif model.status == :ROUNDOFF_LIMITED
        return MOI.NEARLY_FEASIBLE_POINT
    end
    @assert model.status in (
        :STOPVAL_REACHED,
        :MAXEVAL_REACHED,
        :MAXTIME_REACHED,
        :FORCED_STOP,
        :OUT_OF_MEMORY,
        :INVALID_ARGS,
        :FAILURE,
    )
    return MOI.UNKNOWN_RESULT_STATUS
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
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_BOUNDS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.solution[ci.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    },
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.constraint_primal_linear_le[ci.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.EqualTo{Float64},
    },
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.constraint_primal_linear_eq[ci.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{
        MOI.ScalarQuadraticFunction{Float64},
        MOI.LessThan{Float64},
    },
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.constraint_primal_quadratic_le[ci.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{
        MOI.ScalarQuadraticFunction{Float64},
        MOI.EqualTo{Float64},
    },
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.constraint_primal_quadratic_eq[ci.value]
end
