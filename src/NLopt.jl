# Copyright (c) 2013: Steven G. Johnson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module NLopt

using CEnum: @cenum
using NLopt_jll: libnlopt

# NLopt@2.9.0 removed the LD_LBFGS_NOCEDAL enum.
# See https://github.com/stevengj/nlopt/issues/584 for details.
function _is_version_newer_than_2_9()
    major, minor, bugfix = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    @ccall libnlopt.nlopt_version(
        major::Ptr{Cint},
        minor::Ptr{Cint},
        bugfix::Ptr{Cint},
    )::Cvoid
    return (major[] > 2) || (major[] == 2 && minor[] >= 9)
end

include("libnlopt.jl")

############################################################################
# Mirrors of NLopt's C enum constants:

@static if _is_version_newer_than_2_9()
    @enum Algorithm::Cint begin
        GN_DIRECT = 0
        GN_DIRECT_L
        GN_DIRECT_L_RAND
        GN_DIRECT_NOSCAL
        GN_DIRECT_L_NOSCAL
        GN_DIRECT_L_RAND_NOSCAL
        GN_ORIG_DIRECT
        GN_ORIG_DIRECT_L
        GD_STOGO
        GD_STOGO_RAND
        # LD_LBFGS_NOCEDAL
        LD_LBFGS
        LN_PRAXIS
        LD_VAR1
        LD_VAR2
        LD_TNEWTON
        LD_TNEWTON_RESTART
        LD_TNEWTON_PRECOND
        LD_TNEWTON_PRECOND_RESTART
        GN_CRS2_LM
        GN_MLSL
        GD_MLSL
        GN_MLSL_LDS
        GD_MLSL_LDS
        LD_MMA
        LN_COBYLA
        LN_NEWUOA
        LN_NEWUOA_BOUND
        LN_NELDERMEAD
        LN_SBPLX
        LN_AUGLAG
        LD_AUGLAG
        LN_AUGLAG_EQ
        LD_AUGLAG_EQ
        LN_BOBYQA
        GN_ISRES
        AUGLAG
        AUGLAG_EQ
        G_MLSL
        G_MLSL_LDS
        LD_SLSQP
        LD_CCSAQ
        GN_ESCH
        GN_AGS
    end
else
    @enum Algorithm::Cint begin
        GN_DIRECT = 0
        GN_DIRECT_L
        GN_DIRECT_L_RAND
        GN_DIRECT_NOSCAL
        GN_DIRECT_L_NOSCAL
        GN_DIRECT_L_RAND_NOSCAL
        GN_ORIG_DIRECT
        GN_ORIG_DIRECT_L
        GD_STOGO
        GD_STOGO_RAND
        LD_LBFGS_NOCEDAL
        LD_LBFGS
        LN_PRAXIS
        LD_VAR1
        LD_VAR2
        LD_TNEWTON
        LD_TNEWTON_RESTART
        LD_TNEWTON_PRECOND
        LD_TNEWTON_PRECOND_RESTART
        GN_CRS2_LM
        GN_MLSL
        GD_MLSL
        GN_MLSL_LDS
        GD_MLSL_LDS
        LD_MMA
        LN_COBYLA
        LN_NEWUOA
        LN_NEWUOA_BOUND
        LN_NELDERMEAD
        LN_SBPLX
        LN_AUGLAG
        LD_AUGLAG
        LN_AUGLAG_EQ
        LD_AUGLAG_EQ
        LN_BOBYQA
        GN_ISRES
        AUGLAG
        AUGLAG_EQ
        G_MLSL
        G_MLSL_LDS
        LD_SLSQP
        LD_CCSAQ
        GN_ESCH
        GN_AGS
    end
end

Base.convert(::Type{nlopt_algorithm}, a::Algorithm) = nlopt_algorithm(Int(a))
Base.convert(::Type{Algorithm}, r::nlopt_algorithm) = Algorithm(Int(r))

function Algorithm(name::Symbol)::Algorithm
    algorithm = nlopt_algorithm_from_string("$name")
    if UInt32(algorithm) == 0xffffffff
        throw(ArgumentError("unknown algorithm: $name"))
    end
    return algorithm
end

# enum nlopt_result
@enum Result::Cint begin
    FORCED_STOP = -5
    ROUNDOFF_LIMITED = -4
    OUT_OF_MEMORY = -3
    INVALID_ARGS = -2
    FAILURE = -1
    SUCCESS = 1
    STOPVAL_REACHED = 2
    FTOL_REACHED = 3
    XTOL_REACHED = 4
    MAXEVAL_REACHED = 5
    MAXTIME_REACHED = 6
end

Base.convert(::Type{nlopt_result}, r::Result) = nlopt_result(Int(r))
Base.convert(::Type{Result}, r::nlopt_result) = Result(Int(r))

# so that result < 0 checks continue to work
Base.isless(x::Integer, r::Result) = isless(x, Cint(r))
Base.isless(r::Result, x::Integer) = isless(Cint(r), x)
# so that == :Foo checks continue to work
Base.:(==)(s::Symbol, r::Result) = s == Symbol(r)
Base.:(==)(r::Result, s::Symbol) = s == r

############################################################################
# wrapper around nlopt_opt type

# pass both f and o to the callback so that we can handle exceptions
mutable struct Callback_Data
    f::Function
    o::Any # should be Opt, but see Julia issue #269
end

function Base.unsafe_convert(::Type{Ptr{Cvoid}}, c::Callback_Data)
    return pointer_from_objref(c)
end

mutable struct Opt
    opt::Ptr{Cvoid}

    # need to store callback data for objective and constraints in
    # Opt so that they aren't garbage-collected.  cb[1] is the objective.
    cb::Vector{Callback_Data}

    exception::Any

    function Opt(p::Ptr{Cvoid})
        @assert p != C_NULL
        opt = new(p, Array{Callback_Data}(undef, 1), nothing)
        finalizer(destroy, opt)
        return opt
    end
end

function Opt(algorithm::Algorithm, n::Integer)
    if n < 0
        throw(ArgumentError("invalid dimension $n < 0"))
    end
    p = nlopt_create(algorithm, n)
    return Opt(p)
end

function Opt(algorithm::Union{Integer,Symbol}, n::Integer)
    return Opt(Algorithm(algorithm), n)
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, o::Opt) = getfield(o, :opt)

destroy(o::Opt) = nlopt_destroy(o)

Base.ndims(o::Opt)::Int = nlopt_get_dimension(o)

algorithm(o::Opt)::Algorithm = nlopt_get_algorithm(o)

Base.show(io::IO, o::Opt) = print(io, "Opt($(algorithm(o)), $(ndims(o)))")

############################################################################
# copying is a little tricky because we have to tell NLopt to use new
# Callback_Data.

function munge_callback(p::Ptr{Cvoid}, p_user_data::Ptr{Cvoid})
    old_to_new_pointer_map =
        unsafe_pointer_to_objref(p_user_data)::Dict{Ptr{Cvoid},Ptr{Cvoid}}
    return old_to_new_pointer_map[p]
end

function Base.copy(opt::Opt)
    p = nlopt_copy(opt)
    if p == C_NULL
        error("Error in nlopt_copy")
    end
    new_opt = Opt(p)
    opt_callbacks = getfield(opt, :cb)
    new_callbacks = Vector{Callback_Data}(undef, length(opt_callbacks))
    setfield!(new_opt, :cb, new_callbacks)
    old_to_new_pointer_map = Dict{Ptr{Cvoid},Ptr{Cvoid}}(C_NULL => C_NULL)
    for i in 1:length(opt_callbacks)
        if isassigned(opt_callbacks, i)
            new_callbacks[i] = Callback_Data(opt_callbacks[i].f, new_opt)
            old_to_new_pointer_map[pointer_from_objref(opt_callbacks[i])] =
                pointer_from_objref(new_callbacks[i])
        end
    end
    # nlopt_munge_data is a routine that allows us to convert all pointers to
    # existing Callback_Data objects into pointers for the corresponding object
    # in new_callbacks.
    c_fn = @cfunction(munge_callback, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}))
    GC.@preserve old_to_new_pointer_map begin
        p_old_to_new_pointer_map = pointer_from_objref(old_to_new_pointer_map)
        nlopt_munge_data(new_opt, c_fn, p_old_to_new_pointer_map)
    end
    return new_opt
end

############################################################################
# converting error results into exceptions

struct ForcedStop <: Exception end

function errmsg(o::Opt)
    msg = nlopt_get_errmsg(o)
    return msg == C_NULL ? nothing : unsafe_string(msg)
end

function _errmsg(o::Opt)
    s = errmsg(o)
    return s === nothing || isempty(s) ? "" : ": " * s
end

# check result and throw an exception if necessary
chk(o::Opt, result::nlopt_result) = chk(o, convert(Result, result))

function chk(o::Opt, result::Result)
    if result >= 0
        return
    elseif result == ROUNDOFF_LIMITED
        return
    elseif result == INVALID_ARGS
        throw(ArgumentError("invalid NLopt arguments" * _errmsg(o)))
    elseif result == OUT_OF_MEMORY
        throw(OutOfMemoryError())
    else
        error("nlopt failure $result", _errmsg(o))
    end
end

############################################################################
# getting and setting scalar and vector parameters

stopval(o::Opt) = nlopt_get_stopval(o)
stopval!(o::Opt, val::Real) = chk(o, nlopt_set_stopval(o, val))

ftol_rel(o::Opt) = nlopt_get_ftol_rel(o)
ftol_rel!(o::Opt, val::Real) = chk(o, nlopt_set_ftol_rel(o, val))

ftol_abs(o::Opt) = nlopt_get_ftol_abs(o)
ftol_abs!(o::Opt, val::Real) = chk(o, nlopt_set_ftol_abs(o, val))

xtol_rel(o::Opt) = nlopt_get_xtol_rel(o)
xtol_rel!(o::Opt, val::Real) = chk(o, nlopt_set_xtol_rel(o, val))

maxeval(o::Opt) = nlopt_get_maxeval(o)
maxeval!(o::Opt, val::Integer) = chk(o, nlopt_set_maxeval(o, val))

maxtime(o::Opt) = nlopt_get_maxtime(o)
maxtime!(o::Opt, val::Real) = chk(o, nlopt_set_maxtime(o, val))

force_stop(o::Opt) = nlopt_get_force_stop(o)
force_stop!(o::Opt, val::Integer) = chk(o, nlopt_set_force_stop(o, val))
force_stop!(o::Opt) = force_stop!(o, 1)

population(o::Opt) = nlopt_get_population(o)
population!(o::Opt, val::Integer) = chk(o, nlopt_set_population(o, val))

vector_storage(o::Opt) = nlopt_get_vector_storage(o)
vector_storage!(o::Opt, val::Integer) = chk(o, nlopt_set_vector_storage(o, val))

############################################################################
# Optimizer parameters

function lower_bounds(
    o::Opt,
    v::Vector{Cdouble} = Array{Cdouble}(undef, ndims(o)),
)
    if length(v) != ndims(o)
        throw(BoundsError())
    end
    chk(o, nlopt_get_lower_bounds(o, v))
    return v
end

function lower_bounds!(o::Opt, v::Vector{Cdouble})
    if length(v) != ndims(o)
        throw(BoundsError())
    end
    return chk(o, nlopt_set_lower_bounds(o, v))
end

function lower_bounds!(o::Opt, v::AbstractVector{<:Real})
    return lower_bounds!(o, Array{Cdouble}(v))
end

lower_bounds!(o::Opt, val::Real) = chk(o, nlopt_set_lower_bounds1(o, val))

function upper_bounds(
    o::Opt,
    v::Vector{Cdouble} = Array{Cdouble}(undef, ndims(o)),
)
    if length(v) != ndims(o)
        throw(BoundsError())
    end
    chk(o, nlopt_get_upper_bounds(o, v))
    return v
end

function upper_bounds!(o::Opt, v::Vector{Cdouble})
    if length(v) != ndims(o)
        throw(BoundsError())
    end
    return chk(o, nlopt_set_upper_bounds(o, v))
end

function upper_bounds!(o::Opt, v::AbstractVector{<:Real})
    return upper_bounds!(o, Array{Cdouble}(v))
end

upper_bounds!(o::Opt, val::Real) = chk(o, nlopt_set_upper_bounds1(o, val))

function xtol_abs(o::Opt, v::Vector{Cdouble} = Array{Cdouble}(undef, ndims(o)))
    if length(v) != ndims(o)
        throw(BoundsError())
    end
    chk(o, nlopt_get_xtol_abs(o, v))
    return v
end

function xtol_abs!(o::Opt, v::Vector{Cdouble})
    if length(v) != ndims(o)
        throw(BoundsError())
    end
    return chk(o, nlopt_set_xtol_abs(o, v))
end

function xtol_abs!(o::Opt, v::AbstractVector{<:Real})
    return xtol_abs!(o, Array{Cdouble}(v))
end

xtol_abs!(o::Opt, val::Real) = chk(o, nlopt_set_xtol_abs1(o, val))

function local_optimizer!(o::Opt, lo::Opt)
    return chk(o, nlopt_set_local_optimizer(o, lo))
end

function default_initial_step!(o::Opt, x::Vector{Cdouble})
    if length(x) != ndims(o)
        throw(BoundsError())
    end
    return chk(o, nlopt_set_default_initial_step(o, x))
end

function default_initial_step!(o::Opt, x::AbstractVector{<:Real})
    return default_initial_step!(o, Array{Cdouble}(x))
end

function initial_step!(o::Opt, dx::Vector{Cdouble})
    if length(dx) != ndims(o)
        throw(BoundsError())
    end
    return chk(o, nlopt_set_initial_step(o, dx))
end

function initial_step!(o::Opt, dx::AbstractVector{<:Real})
    return initial_step!(o, Array{Cdouble}(dx))
end

function initial_step!(o::Opt, dx::Real)
    return chk(o, nlopt_set_initial_step1(o, dx))
end

function initial_step(o::Opt, x::Vector{Cdouble}, dx::Vector{Cdouble})
    if length(x) != ndims(o) || length(dx) != ndims(o)
        throw(BoundsError())
    end
    chk(o, nlopt_get_initial_step(o, x, dx))
    return dx
end

function initial_step(o::Opt, x::AbstractVector{<:Real})
    return initial_step(o, Array{Cdouble}(x), Array{Cdouble}(undef, ndims(o)))
end

############################################################################

function algorithm_name(a::Algorithm)
    p = nlopt_algorithm_name(a)
    # pointer cannot be C_NULL because we are using only valid Enums
    @assert p !== C_NULL
    return unsafe_string(p)
end

algorithm_name(a::Union{Integer,Symbol}) = algorithm_name(Algorithm(a))

algorithm_name(o::Opt) = algorithm_name(algorithm(o))

function Base.show(io::IO, ::MIME"text/plain", a::Algorithm)
    show(io, a)
    return print(io, ": ", algorithm_name(a))
end

numevals(o::Opt) = nlopt_get_numevals(o)

############################################################################

function version()
    major, minor, patch = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    nlopt_version(major, minor, patch)
    return VersionNumber(major[], minor[], patch[])
end

const NLOPT_VERSION = version()

############################################################################

srand(seed::Integer) = nlopt_srand(seed)

srand_time() = nlopt_srand_time()

############################################################################
# Objective function:

function nlopt_callback_wrapper(
    n::Cuint,
    p_x::Ptr{Cdouble},
    p_grad::Ptr{Cdouble},
    d_::Ptr{Cvoid},
)::Cdouble
    d = unsafe_pointer_to_objref(d_)::Callback_Data
    x = unsafe_wrap(Array, p_x, (n,))
    grad = unsafe_wrap(Array, p_grad, (n,))
    try
        return d.f(x, p_grad == C_NULL ? Cdouble[] : grad)
    catch e
        _catch_forced_stop(d.o, e)
    end
    return NaN
end

function min_objective!(o::Opt, f::Function)
    cb = Callback_Data(f, o)
    getfield(o, :cb)[1] = cb
    c_fn = @cfunction(
        nlopt_callback_wrapper,
        Cdouble,
        (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid})
    )
    return chk(o, nlopt_set_min_objective(o, c_fn, cb))
end

function max_objective!(o::Opt, f::Function)
    cb = Callback_Data(f, o)
    getfield(o, :cb)[1] = cb
    c_fn = @cfunction(
        nlopt_callback_wrapper,
        Cdouble,
        (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid})
    )
    return chk(o, nlopt_set_max_objective(o, c_fn, cb))
end

############################################################################
# Nonlinear constraints:

function inequality_constraint!(o::Opt, f::Function, tol::Real = 0.0)
    cb = Callback_Data(f, o)
    push!(getfield(o, :cb), cb)
    c_fn = @cfunction(
        nlopt_callback_wrapper,
        Cdouble,
        (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid})
    )
    return chk(o, nlopt_add_inequality_constraint(o, c_fn, cb, tol))
end

function equality_constraint!(o::Opt, f::Function, tol::Real = 0.0)
    cb = Callback_Data(f, o)
    push!(getfield(o, :cb), cb)
    c_fn = @cfunction(
        nlopt_callback_wrapper,
        Cdouble,
        (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid})
    )
    return chk(o, nlopt_add_equality_constraint(o, c_fn, cb, tol))
end

function remove_constraints!(o::Opt)
    resize!(getfield(o, :cb), 1)
    chk(o, nlopt_remove_inequality_constraints(o))
    chk(o, nlopt_remove_equality_constraints(o))
    return
end

############################################################################
# Vector-valued constraints

function nlopt_vcallback_wrapper(
    m::Cuint,
    p_res::Ptr{Cdouble},
    n::Cuint,
    p_x::Ptr{Cdouble},
    p_grad::Ptr{Cdouble},
    d_::Ptr{Cvoid},
)
    d = unsafe_pointer_to_objref(d_)::Callback_Data
    res = unsafe_wrap(Array, p_res, (m,))
    x = unsafe_wrap(Array, p_x, (n,))
    grad =
        p_grad == C_NULL ? zeros(Cdouble, 0, 0) :
        unsafe_wrap(Array, p_grad, (n, m))
    try
        d.f(res, x, grad)
    catch e
        _catch_forced_stop(d.o, e)
    end
    return
end

function _catch_forced_stop(o::Opt, e)
    if e isa ForcedStop
        setfield!(o, :exception, e)
    elseif e isa InterruptException
        setfield!(o, :exception, ForcedStop())
    else
        setfield!(o, :exception, CapturedException(e, catch_backtrace()))
    end
    force_stop!(o)
    return
end

function inequality_constraint!(o::Opt, f::Function, tol::Vector{Cdouble})
    cb = Callback_Data(f, o)
    push!(getfield(o, :cb), cb)
    c_fn = @cfunction(
        nlopt_vcallback_wrapper,
        Cvoid,
        (Cuint, Ptr{Cdouble}, Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}),
    )
    ret = nlopt_add_inequality_mconstraint(o, length(tol), c_fn, cb, tol)
    return chk(o, ret)
end

function inequality_constraint!(
    o::Opt,
    f::Function,
    tol::AbstractVector{<:Real},
)
    return inequality_constraint!(o, f, Array{Float64}(tol))
end

function inequality_constraint!(
    o::Opt,
    m::Integer,
    f::Function,
    tol::Real = 0.0,
)
    return inequality_constraint!(o, f, fill(Cdouble(tol), m))
end

function equality_constraint!(o::Opt, f::Function, tol::Vector{Cdouble})
    cb = Callback_Data(f, o)
    push!(getfield(o, :cb), cb)
    c_fn = @cfunction(
        nlopt_vcallback_wrapper,
        Cvoid,
        (Cuint, Ptr{Cdouble}, Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}),
    )
    return chk(o, nlopt_add_equality_mconstraint(o, length(tol), c_fn, cb, tol))
end

function equality_constraint!(o::Opt, f::Function, tol::AbstractVector{<:Real})
    return equality_constraint!(o, f, Array{Float64}(tol))
end

function equality_constraint!(o::Opt, m::Integer, f::Function, tol::Real = 0.0)
    return equality_constraint!(o, f, fill(Cdouble(tol), m))
end

############################################################################
# Dict-like API for generic algorithm properties

"""
    OptParams <: AbstractDict{String, Float64}

Dictionary-like structure for accessing algorithm-specific parameters for
an NLopt optimization object `opt`, returned by `opt.params`.

Use this object to both set and view these string-keyed numeric parameters.
"""
struct OptParams <: AbstractDict{String,Float64}
    o::Opt
end

Base.length(p::OptParams)::Int = nlopt_num_params(p.o)

Base.haskey(p::OptParams, s::AbstractString)::Bool = nlopt_has_param(p.o, s)

function Base.get(p::OptParams, s::AbstractString, default::Float64)
    return nlopt_get_param(p.o, s, default)
end

function Base.get(p::OptParams, s::AbstractString, default)
    if !haskey(p, s)
        return default
    end
    return nlopt_get_param(p.o, s, NaN)
end

function Base.setindex!(p::OptParams, v::Real, s::AbstractString)
    ret = nlopt_set_param(p.o, s, v)
    return chk(p.o, ret)
end

function Base.setindex!(p::OptParams, v::Algorithm, s::AbstractString)
    return setindex!(p, Int(v), s)
end

function Base.iterate(p::OptParams, state = 0)
    if state >= length(p)
        return nothing
    end
    name_ptr = nlopt_nth_param(p.o, state)
    @assert name_ptr != C_NULL
    name = unsafe_string(name_ptr)
    return name => p[name], state + 1
end

############################################################################
# property-based getters setters opt.foo for Julia 0.7
# … at some point we will deprecate the old interface.

function Base.getproperty(o::Opt, p::Symbol)
    if p === :lower_bounds
        return lower_bounds(o)
    elseif p === :upper_bounds
        return upper_bounds(o)
    elseif p === :stopval
        return stopval(o)
    elseif p === :ftol_rel
        return ftol_rel(o)
    elseif p === :ftol_abs
        return ftol_abs(o)
    elseif p === :xtol_rel
        return xtol_rel(o)
    elseif p === :xtol_abs
        return xtol_abs(o)
    elseif p === :maxeval
        return maxeval(o)
    elseif p === :maxtime
        return maxtime(o)
    elseif p === :force_stop
        return force_stop(o)
    elseif p === :population
        return population(o)
    elseif p === :vector_storage
        return vector_storage(o)
    elseif p === :initial_step
        error(
            "Getting `initial_step` is unsupported. Use " *
            "`initial_step(opt, x)` to access the initial step at a point `x`.",
        )
    elseif p === :algorithm
        return algorithm(o)
    elseif p === :numevals
        return numevals(o)
    elseif p === :errmsg
        return errmsg(o)
    elseif p === :params
        return OptParams(o)
    else
        error("type Opt has no readable property $p")
    end
end

function Base.setproperty!(o::Opt, p::Symbol, x)
    if p === :lower_bounds
        lower_bounds!(o, x)
    elseif p === :upper_bounds
        upper_bounds!(o, x)
    elseif p === :stopval
        stopval!(o, x)
    elseif p === :ftol_rel
        ftol_rel!(o, x)
    elseif p === :ftol_abs
        ftol_abs!(o, x)
    elseif p === :xtol_rel
        xtol_rel!(o, x)
    elseif p === :xtol_abs
        xtol_abs!(o, x)
    elseif p === :maxeval
        maxeval!(o, x)
    elseif p === :maxtime
        maxtime!(o, x)
    elseif p === :force_stop
        force_stop!(o, x)
    elseif p === :population
        population!(o, x)
    elseif p === :vector_storage
        vector_storage!(o, x)
    elseif p === :local_optimizer
        local_optimizer!(o, x)
    elseif p === :default_initial_step
        default_initial_step!(o, x)
    elseif p === :initial_step
        initial_step!(o, x)
    elseif p === :min_objective
        min_objective!(o, x)
    elseif p === :max_objective
        max_objective!(o, x)
    elseif p === :inequality_constraint
        inequality_constraint!(o, x)
    elseif p === :equality_constraint
        equality_constraint!(o, x)
    else
        error("type Opt has no writable property $p")
    end
    return x
end

function Base.propertynames(o::Opt)
    return (
        :lower_bounds,
        :upper_bounds,
        :stopval,
        :ftol_rel,
        :ftol_abs,
        :xtol_rel,
        :xtol_abs,
        :maxeval,
        :maxtime,
        :force_stop,
        :population,
        :vector_storage,
        :initial_step,
        :algorithm,
        :local_optimizer,
        :default_initial_step,
        :initial_step,
        :min_objective,
        :max_objective,
        :inequality_constraint,
        :equality_constraint,
        :numevals,
        :errmsg,
        :params,
    )
end

############################################################################
# Perform the optimization:

function optimize!(o::Opt, x::Vector{Cdouble})
    if length(x) != ndims(o)
        throw(BoundsError())
    end
    opt_f = Ref{Cdouble}(NaN)
    ret::Result = nlopt_optimize(o, x, opt_f)
    # We do not need to check the value of `ret`, except if it is a FORCED_STOP
    # with a Julia-related exception from a callback
    if ret == FORCED_STOP
        exception = getfield(o, :exception)
        setfield!(o, :exception, nothing)
        if exception !== nothing && !(exception isa ForcedStop)
            throw(exception)
        end
    end
    return opt_f[], x, Symbol(ret)
end

function optimize(o::Opt, x::AbstractVector{<:Real})
    return optimize!(o, copyto!(Array{Cdouble}(undef, length(x)), x))
end

export Opt,
    NLOPT_VERSION,
    algorithm,
    algorithm_name,
    ForcedStop,
    lower_bounds!,
    lower_bounds,
    upper_bounds!,
    upper_bounds,
    stopval!,
    stopval,
    ftol_rel!,
    ftol_rel,
    ftol_abs!,
    ftol_abs,
    xtol_rel!,
    xtol_rel,
    xtol_abs!,
    xtol_abs,
    maxeval!,
    maxeval,
    maxtime!,
    maxtime,
    force_stop!,
    force_stop,
    population!,
    population,
    vector_storage!,
    vector_storage,
    initial_step!,
    initial_step,
    default_initial_step!,
    local_optimizer!,
    min_objective!,
    max_objective!,
    equality_constraint!,
    inequality_constraint!,
    remove_constraints!,
    optimize!,
    optimize,
    Algorithm,
    Result

@static if !isdefined(Base, :get_extension)
    include("../ext/NLoptMathOptInterfaceExt.jl")
    using .NLoptMathOptInterfaceExt
    const Optimizer = NLoptMathOptInterfaceExt.Optimizer
else
    # declare this upfront so that the MathOptInterface extension can assign it
    # without creating a new global
    global Optimizer
end

end # module
