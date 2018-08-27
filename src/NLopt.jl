VERSION < v"0.7.0-beta2.199" && __precompile__()

module NLopt

export Opt, NLOPT_VERSION, algorithm, algorithm_name, ForcedStop,
       lower_bounds!, lower_bounds, upper_bounds!, upper_bounds, stopval!, stopval, ftol_rel!, ftol_rel, ftol_abs!, ftol_abs, xtol_rel!, xtol_rel, xtol_abs!, xtol_abs, maxeval!, maxeval, maxtime!, maxtime, force_stop!, force_stop, force_stop!, population!, population, vector_storage!, vector_storage, initial_step!, initial_step, default_initial_step!, local_optimizer!,
       min_objective!, max_objective!, equality_constraint!, inequality_constraint!, remove_constraints!,
       optimize!, optimize, Algorithm, Result

import MathProgBase.SolverInterface
import MathProgBase.SolverInterface.optimize!

import Libdl
const depsjl_path = joinpath(dirname(@__FILE__), "..", "deps", "deps.jl")
if !isfile(depsjl_path)
    error("NLopt not installed properly; run Pkg.build(\"NLopt\"), restart Julia, and try again.")
end
include(depsjl_path)

function __init__()
    check_deps()
    v = version()
    v >= v"2.5" || error("NLopt $v < 2.5 is too old")
end

############################################################################
# Mirrors of NLopt's C enum constants:

@enum Algorithm::Cint begin
    GN_DIRECT=0
    GN_DIRECT_L=1
    GN_DIRECT_L_RAND=2
    GN_DIRECT_NOSCAL=3
    GN_DIRECT_L_NOSCAL=4
    GN_DIRECT_L_RAND_NOSCAL=5
    GN_ORIG_DIRECT=6
    GN_ORIG_DIRECT_L=7
    GD_STOGO=8
    GD_STOGO_RAND=9
    LD_LBFGS_NOCEDAL=10
    LD_LBFGS=11
    LN_PRAXIS=12
    LD_VAR1=13
    LD_VAR2=14
    LD_TNEWTON=15
    LD_TNEWTON_RESTART=16
    LD_TNEWTON_PRECOND=17
    LD_TNEWTON_PRECOND_RESTART=18
    GN_CRS2_LM=19
    GN_MLSL=20
    GD_MLSL=21
    GN_MLSL_LDS=22
    GD_MLSL_LDS=23
    LD_MMA=24
    LN_COBYLA=25
    LN_NEWUOA=26
    LN_NEWUOA_BOUND=27
    LN_NELDERMEAD=28
    LN_SBPLX=29
    LN_AUGLAG=30
    LD_AUGLAG=31
    LN_AUGLAG_EQ=32
    LD_AUGLAG_EQ=33
    LN_BOBYQA=34
    GN_ISRES=35
    AUGLAG=36
    AUGLAG_EQ=37
    G_MLSL=38
    G_MLSL_LDS=39
    LD_SLSQP=40
    LD_CCSAQ=41
    GN_ESCH=42
    GN_AGS=43
end

const sym2alg = Dict(Symbol(i)=>i for i in instances(Algorithm))

# enum nlopt_result
@enum Result::Cint begin
    FORCED_STOP=-5
    ROUNDOFF_LIMITED=-4
    OUT_OF_MEMORY=-3
    INVALID_ARGS=-2
    FAILURE=-1
    SUCCESS=1
    STOPVAL_REACHED=2
    FTOL_REACHED=3
    XTOL_REACHED=4
    MAXEVAL_REACHED=5
    MAXTIME_REACHED=6
end

# so that result < 0 checks continue to work
Base.isless(x::Integer, r::Result) = isless(x, Cint(r))
Base.isless(r::Result, x::Integer) = isless(Cint(r), x)
# so that == :Foo checks continue to work
Base.:(==)(s::Symbol, r::Result) = s == Symbol(r)
Base.:(==)(r::Result, s::Symbol) = s == r

############################################################################
# wrapper around nlopt_opt type

const _Opt = Ptr{Cvoid} # nlopt_opt

# pass both f and o to the callback so that we can handle exceptions
struct Callback_Data
    f::Function
    o::Any # should be Opt, but see Julia issue #269
end

mutable struct Opt
    opt::_Opt

    # need to store callback data for objective and constraints in
    # Opt so that they aren't garbage-collected.  cb[1] is the objective.
    cb::Vector{Callback_Data}

    function Opt(p::_Opt)
        opt = new(p, Array{Callback_Data}(undef,1))
        finalizer(destroy,opt)
        opt
    end
    function Opt(algorithm::Algorithm, n::Integer)
        n >= 0 || throw(ArgumentError("invalid dimension $n < 0"))
        p = ccall((:nlopt_create,libnlopt), _Opt, (Algorithm, Cuint),
                  algorithm, n)
        if p == C_NULL
            error("Error in nlopt_create")
        end
        Opt(p)
    end
    Opt(alg::Integer, n::Integer) = Opt(Algorithm(alg), n)
    Opt(algorithm::Symbol, n::Integer) = Opt(try sym2alg[algorithm]
                                             catch
                         throw(ArgumentError("unknown algorithm $algorithm"))
                                             end, n)
end

Base.unsafe_convert(::Type{_Opt}, o::Opt) = getfield(o, :opt) # for passing to ccall

destroy(o::Opt) = ccall((:nlopt_destroy,libnlopt), Cvoid, (_Opt,), o)

Base.ndims(o::Opt) = Int(ccall((:nlopt_get_dimension,libnlopt), Cuint, (_Opt,), o))
algorithm(o::Opt) = ccall((:nlopt_get_algorithm,libnlopt), Algorithm, (_Opt,), o)

Base.show(io::IO, o::Opt) = print(io, "Opt($(algorithm(o)), $(ndims(o)))")

############################################################################
# copying is a little tricky because we have to tell NLopt to use
# new Callback_Data.

# callback wrapper for nlopt_munge_data in NLopt 2.4
function munge_callback(p::Ptr{Cvoid}, f_::Ptr{Cvoid})
    f = unsafe_pointer_to_objref(f_)::Function
    f(p)::Ptr{Cvoid}
end

function Base.copy(o::Opt)
    p = ccall((:nlopt_copy,libnlopt), _Opt, (_Opt,), o)
    if p == C_NULL
        error("Error in nlopt_copy")
    end
    n = Opt(p)

    cb = getfield(o.cb)
    n.cb = similar(cb)
    for i = 1:length(cb)
        try
            n.cb[i] = Callback_Data(cb[i].f, n)
        catch e
            # if objective has not been set, cb[1] will throw
            # an UndefRefError, which is okay.
            if i != 1 || !isa(e, UndefRefError)
                rethrow(e) # some not-okay exception
            end
        end
    end

    try
        # n.o, for each callback, stores a pointer to an element of o.cb,
        # and we need to convert this into a pointer to the corresponding
        # element of n.cb.  nlopt_munge_data allows us to call a function
        # to transform each stored pointer in n.o, and we use the cbi
        # dictionary to convert pointers to indices into o.cb, whence
        # we obtain the corresponding element of n.cb.
        cbi = Dict{Ptr{Cvoid},Int}()
        for i in 1:length(cb)
            try
                cbi[pointer_from_objref(cb[i])] = i
            catch
            end
        end
        munge_callback_ptr = @cfunction(munge_callback, Ptr{Cvoid},
                                        (Ptr{Cvoid}, Ptr{Cvoid}))
        ccall((:nlopt_munge_data,libnlopt), Cvoid, (_Opt, Ptr{Cvoid}, Any),
              n, munge_callback_ptr,
              p::Ptr{Cvoid} -> p==C_NULL ? C_NULL :
                              pointer_from_objref(n.cb[cbi[p]]))
    catch e0
        # nlopt_munge_data not available, punt unless there is
        # no callback data
        try
            cb[1]
        catch e
            if length(cb) == 1 && isa(e, UndefRefError)
                return n
            end
        end
        error("copy(o::Opt) not supported for NLopt version < 2.4")
    end

    return n
end

############################################################################
# converting error results into exceptions

struct ForcedStop <: Exception end

# cache current exception for forced stop
nlopt_exception = nothing

function errmsg(o::Opt)
    msg = ccall((:nlopt_get_errmsg,libnlopt), Ptr{UInt8}, (_Opt,), o)
    return msg == C_NULL ? nothing : unsafe_string(msg)
end

function _errmsg(o::Opt)
    s = errmsg(o)
    return s === nothing || isempty(s) ? "" : ": "*s
end

# check result and throw an exception if necessary
function chk(o::Opt, result::Result)
    if result < 0 && result != ROUNDOFF_LIMITED
        if result == INVALID_ARGS
            throw(ArgumentError("invalid NLopt arguments"*_errmsg(o)))
        elseif result == OUT_OF_MEMORY
            throw(OutOfMemoryError())
        elseif result == FORCED_STOP
            global nlopt_exception
            e = nlopt_exception
            if e !== nothing && !isa(e, ForcedStop)
                nlopt_exception = nothing
                rethrow(e)
            end
        else
            error("nlopt failure $result", _errmsg(o))
        end
    end
    return nothing
end

############################################################################
# getting and setting scalar and vector parameters

# make a quoted symbol expression out of the arguments
qsym(args...) = Expr(:quote, Symbol(string(args...)))

# scalar parameters p of type T
macro GETSET(T, p)
    Tg = T == :Cdouble ? :Real : (T == :Cint || T == :Cuint ? :Integer : :Any)
    ps = Symbol(string(p, "!"))
    quote
        $(esc(p))(o::Opt) = ccall(($(qsym("nlopt_get_", p)),libnlopt),
                                  $T, (_Opt,), o)
        $(esc(ps))(o::Opt, val::$Tg) =
          chk(o, ccall(($(qsym("nlopt_set_", p)),libnlopt),
                     Result, (_Opt, $T), o, val))
    end
end

# Vector{Cdouble} parameters p
macro GETSET_VEC(p)
    ps = Symbol(string(p, "!"))
    quote
        function $(esc(p))(o::Opt, v::Vector{Cdouble})
            if length(v) != ndims(o)
                throw(BoundsError())
            end
            chk(o, ccall(($(qsym("nlopt_get_", p)),libnlopt),
                      Result, (_Opt, Ptr{Cdouble}), o, v))
            v
        end
        $(esc(p))(o::Opt) = $(esc(p))(o, Array{Cdouble}(undef, ndims(o)))
        function $(esc(ps))(o::Opt, v::Vector{Cdouble})
            if length(v) != ndims(o)
                throw(BoundsError())
            end
            chk(o, ccall(($(qsym("nlopt_set_", p)),libnlopt),
                      Result, (_Opt, Ptr{Cdouble}), o, v))
        end
        $(esc(ps))(o::Opt, v::AbstractVector{<:Real}) =
          $(esc(ps))(o, Array{Cdouble}(v))
        $(esc(ps))(o::Opt, val::Real) =
          chk(o, ccall(($(qsym("nlopt_set_", p, "1")),libnlopt),
                     Result, (_Opt, Cdouble), o, val))
    end
end

############################################################################
# Optimizer parameters

@GETSET_VEC lower_bounds
@GETSET_VEC upper_bounds
@GETSET Cdouble stopval
@GETSET Cdouble ftol_rel
@GETSET Cdouble ftol_abs
@GETSET Cdouble xtol_rel
@GETSET_VEC xtol_abs
@GETSET Cint maxeval
@GETSET Cdouble maxtime
@GETSET Cint force_stop
@GETSET Cuint population
@GETSET Cuint vector_storage

force_stop!(o::Opt) = force_stop!(o, 1)

local_optimizer!(o::Opt, lo::Opt) =
  chk(o, ccall((:nlopt_set_local_optimizer,libnlopt),
             Result, (_Opt, _Opt), o, lo))

# the initial-stepsize stuff is a bit different than GETSET_VEC,
# since the heuristics depend on the position x.

function default_initial_step!(o::Opt, x::Vector{Cdouble})
    if length(x) != ndims(o)
        throw(BoundsError())
    end
    chk(o, ccall((:nlopt_set_default_initial_step,libnlopt),
               Result, (_Opt, Ptr{Cdouble}), o, x))
end
default_initial_step!(o::Opt, x::AbstractVector{<:Real}) =
  default_initial_step!(o, Array{Cdouble}(x))

function initial_step!(o::Opt, dx::Vector{Cdouble})
    if length(dx) != ndims(o)
        throw(BoundsError())
    end
    chk(o, ccall((:nlopt_set_initial_step,libnlopt),
               Result, (_Opt, Ptr{Cdouble}), o, dx))
end
initial_step!(o::Opt, dx::AbstractVector{<:Real}) =
  initial_step!(o, Array{Cdouble}(dx))
initial_step!(o::Opt, dx::Real) =
  chk(o, ccall((:nlopt_set_initial_step1,libnlopt),
             Result, (_Opt, Cdouble), o, dx))

function initial_step(o::Opt, x::Vector{Cdouble}, dx::Vector{Cdouble})
    if length(x) != ndims(o) || length(dx) != ndims(o)
        throw(BoundsError())
    end
    chk(o, ccall((:nlopt_get_initial_step,libnlopt),
               Result, (_Opt, Ptr{Cdouble}, Ptr{Cdouble}), o, x, dx))
    dx
end
initial_step(o::Opt, x::AbstractVector{<:Real}) =
    initial_step(o, Array{Cdouble}(x),
                 Array{Cdouble}(undef, ndims(o)))

############################################################################

function algorithm_name(a::Algorithm)
    s = ccall((:nlopt_algorithm_name,libnlopt), Ptr{UInt8}, (Algorithm,), a)
    if s == C_NULL
        throw(ArgumentError("invalid algorithm $a"))
    end
    return unsafe_string(s)
end

algorithm_name(a::Integer) = algorithm_name(Algorithm(a))
algorithm_name(a::Symbol) = algorithm_name(try sym2alg[a]
                                           catch
                             throw(ArgumentError("unknown algorithm $a"))
                                           end)
algorithm_name(o::Opt) = algorithm_name(algorithm(o))

function Base.show(io::IO, ::MIME"text/plain", a::Algorithm)
    show(io, a)
    print(io, ": ", algorithm_name(a))
end

numevals(o::Opt) = ccall((:nlopt_get_numevals,libnlopt), Cint, (_Opt,), o)

############################################################################

function version()
    major = Ref{Cint}()
    minor = Ref{Cint}()
    patch = Ref{Cint}()
    ccall((:nlopt_version,libnlopt), Cvoid, (Ref{Cint},Ref{Cint},Ref{Cint}),
          major, minor, patch)
    return VersionNumber(major[], minor[], patch[])
end

const NLOPT_VERSION = version()

############################################################################

srand(seed::Integer) = ccall((:nlopt_srand,libnlopt),
                             Cvoid, (Culong,), seed)
srand_time() = ccall((:nlopt_srand_time,libnlopt), Cvoid, ())

############################################################################
# Objective function:

const empty_grad = Cdouble[] # for passing when grad == C_NULL

function nlopt_callback_wrapper(n::Cuint, x::Ptr{Cdouble},
                                grad::Ptr{Cdouble}, d_::Ptr{Cvoid})
    d = unsafe_pointer_to_objref(d_)::Callback_Data
    try
        res = convert(Cdouble,
                      d.f(unsafe_wrap(Array, x, (convert(Int, n),)),
                          grad == C_NULL ? empty_grad
                          : unsafe_wrap(Array, grad, (convert(Int, n),))))
        return res::Cdouble
    catch e
        global nlopt_exception
        nlopt_exception = e
        force_stop!(d.o::Opt)
        return 0.0 # ignored by nlopt
    end
end

for m in (:min, :max)
    mf = Symbol(string(m,"_objective!"))
    @eval function $mf(o::Opt, f::Function)
        getfield(o, :cb)[1] = Callback_Data(f, o)
        nlopt_callback_wrapper_ptr = @cfunction(nlopt_callback_wrapper,
            Cdouble, (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
        chk(o, ccall(($(qsym("nlopt_set_", m, "_objective")),libnlopt),
                   Result, (_Opt, Ptr{Cvoid}, Any),
                   o, nlopt_callback_wrapper_ptr,
                   getfield(o, :cb)[1]))
    end
end

############################################################################
# Nonlinear constraints:

for c in (:inequality, :equality)
    cf = Symbol(string(c, "_constraint!"))
    @eval function $cf(o::Opt, f::Function, tol::Real=0.0)
        push!(getfield(o, :cb), Callback_Data(f, o))
        nlopt_callback_wrapper_ptr = @cfunction(nlopt_callback_wrapper,
            Cdouble, (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
        chk(o, ccall(($(qsym("nlopt_add_", c, "_constraint")),libnlopt),
                   Result, (_Opt, Ptr{Cvoid}, Any, Cdouble),
                   o, nlopt_callback_wrapper_ptr,
                   getfield(o, :cb)[end], tol))
    end
end

function remove_constraints!(o::Opt)
    resize!(getfield(o, :cb), 1)
    chkn(ccall((:nlopt_remove_inequality_constraints,libnlopt),
               Result, (_Opt,), o))
    chkn(ccall((:nlopt_remove_equality_constraints,libnlopt),
               Result, (_Opt,), o))
end

############################################################################
# Vector-valued constraints


const empty_jac = Array{Cdouble}(undef,0,0) # for passing when grad == C_NULL

function nlopt_vcallback_wrapper(m::Cuint, res::Ptr{Cdouble},
                                 n::Cuint, x::Ptr{Cdouble},
                                 grad::Ptr{Cdouble}, d_::Ptr{Cvoid})
    d = unsafe_pointer_to_objref(d_)::Callback_Data
    try
        d.f(unsafe_wrap(Array, res, (convert(Int, m),)),
            unsafe_wrap(Array, x, (convert(Int, n),)),
            grad == C_NULL ? empty_jac
            : unsafe_wrap(Array, grad, (convert(Int, n),convert(Int, m))))
    catch e
        global nlopt_exception
        nlopt_exception = e
        force_stop!(d.o::Opt)
    end
    nothing
end

for c in (:inequality, :equality)
    cf = Symbol(string(c, "_constraint!"))
    @eval begin
        function $cf(o::Opt, f::Function, tol::Vector{Cdouble})
            push!(getfield(o, :cb), Callback_Data(f, o))
            nlopt_vcallback_wrapper_ptr = @cfunction(nlopt_vcallback_wrapper, Cvoid,
                  (Cuint, Ptr{Cdouble}, Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
            chk(o, ccall(($(qsym("nlopt_add_", c, "_mconstraint")),
                        libnlopt),
                       Result, (_Opt, Cuint, Ptr{Cvoid}, Any, Ptr{Cdouble}),
                       o, length(tol), nlopt_vcallback_wrapper_ptr,
                       getfield(o, :cb)[end], tol))
        end
        $cf(o::Opt, f::Function, tol::AbstractVector{<:Real}) =
           $cf(o, f, Array{Float64}(tol))
        $cf(o::Opt, m::Integer, f::Function, tol::Real=0.0) =
           $cf(o, f, fill!(Cdouble(tol), m))
    end
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
        return initial_step(o)
    elseif p === :algorithm
        return algorithm(o)
    elseif p === :numevals
        return numevals(o)
    elseif p === :errmsg
        return errmsg(o)
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

Base.propertynames(o::Opt) =
   (:lower_bounds, :upper_bounds, :stopval, :ftol_rel, :ftol_abs, :xtol_rel, :xtol_abs, :maxeval, :maxtime, :force_stop, :population, :vector_storage, :initial_step, :algorithm, :local_optimizer, :default_initial_step, :initial_step, :min_objective, :max_objective, :inequality_constraint, :equality_constraint, :numevals, :errmsg)

############################################################################
# Perform the optimization:

function optimize!(o::Opt, x::Vector{Cdouble})
    if length(x) != ndims(o)
        throw(BoundsError())
    end
    opt_f = Array{Cdouble}(undef,1)
    ret = ccall((:nlopt_optimize,libnlopt), Result, (_Opt, Ptr{Cdouble},
                                                     Ptr{Cdouble}),
                o, x, opt_f)
    ret == INVALID_ARGS && chk(o, ret)
    return (opt_f[1], x, Symbol(ret))
end

optimize(o::Opt, x::AbstractVector{<:Real}) =
  optimize!(o, copyto!(Array{Cdouble}(undef,length(x)), x))

############################################################################

include("NLoptSolverInterface.jl")


end # module
