module NLopt

export Opt, NLOPT_VERSION, algorithm, algorithm_name, ForcedStop,
       lower_bounds!, lower_bounds, upper_bounds!, upper_bounds, stopval!, stopval, ftol_rel!, ftol_rel, ftol_abs!, ftol_abs, xtol_rel!, xtol_rel, xtol_abs!, xtol_abs, maxeval!, maxeval, maxtime!, maxtime, force_stop!, force_stop, force_stop!, population!, population, vector_storage!, vector_storage, initial_step!, initial_step, default_initial_step!, local_optimizer!,
       min_objective!, max_objective!, equality_constraint!, inequality_constraint!, remove_constraints!,
       optimize!, optimize

import Base.ndims, Base.copy, Base.convert, Base.show

############################################################################
# Mirrors of NLopt's C enum constants:

# Problem: the NLopt API uses various enum types for both arguments and
# return values, but sizeof(enum) is compiler- and architecture dependent.
# By default, gcc stores enums as int (or unsigned int) unless -fshort-enums
# is used in the compiler options, but there are some architectures (ARM?)
# where -fshort-enums is the default.  [The MSVC and Intel compilers also
# apparently default to sizeof(enum)==sizeof(int).]  So, if Julia is
# ever ported to ARM, this may need to be fixed (since all of these enums
# will be packed into a single byte on -fshort-enums architectures).
typealias Cenum Cint
typealias cenum int32

# enum nlopt_algorithm
const GN_DIRECT = cenum(0)
const GN_DIRECT_L = cenum(1)
const GN_DIRECT_L_RAND = cenum(2)
const GN_DIRECT_NOSCAL = cenum(3)
const GN_DIRECT_L_NOSCAL = cenum(4)
const GN_DIRECT_L_RAND_NOSCAL = cenum(5)
const GN_ORIG_DIRECT = cenum(6)
const GN_ORIG_DIRECT_L = cenum(7)
const GD_STOGO = cenum(8)
const GD_STOGO_RAND = cenum(9)
const LD_LBFGS_NOCEDAL = cenum(10)
const LD_LBFGS = cenum(11)
const LN_PRAXIS = cenum(12)
const LD_VAR1 = cenum(13)
const LD_VAR2 = cenum(14)
const LD_TNEWTON = cenum(15)
const LD_TNEWTON_RESTART = cenum(16)
const LD_TNEWTON_PRECOND = cenum(17)
const LD_TNEWTON_PRECOND_RESTART = cenum(18)
const GN_CRS2_LM = cenum(19)
const GN_MLSL = cenum(20)
const GD_MLSL = cenum(21)
const GN_MLSL_LDS = cenum(22)
const GD_MLSL_LDS = cenum(23)
const LD_MMA = cenum(24)
const LN_COBYLA = cenum(25)
const LN_NEWUOA = cenum(26)
const LN_NEWUOA_BOUND = cenum(27)
const LN_NELDERMEAD = cenum(28)
const LN_SBPLX = cenum(29)
const LN_AUGLAG = cenum(30)
const LD_AUGLAG = cenum(31)
const LN_AUGLAG_EQ = cenum(32)
const LD_AUGLAG_EQ = cenum(33)
const LN_BOBYQA = cenum(34)
const GN_ISRES = cenum(35)
const AUGLAG = cenum(36)
const AUGLAG_EQ = cenum(37)
const G_MLSL = cenum(38)
const G_MLSL_LDS = cenum(39)
const LD_SLSQP = cenum(40)
const LD_CCSAQ = cenum(41)
const LD_ESCH = cenum(42)
const NUM_ALGORITHMS = 43

const alg2int = (Symbol=>Cenum)[ :GN_DIRECT=>GN_DIRECT, :GN_DIRECT_L=>GN_DIRECT_L, :GN_DIRECT_L_RAND=>GN_DIRECT_L_RAND, :GN_DIRECT_NOSCAL=>GN_DIRECT_NOSCAL, :GN_DIRECT_L_NOSCAL=>GN_DIRECT_L_NOSCAL, :GN_DIRECT_L_RAND_NOSCAL=>GN_DIRECT_L_RAND_NOSCAL, :GN_ORIG_DIRECT=>GN_ORIG_DIRECT, :GN_ORIG_DIRECT_L=>GN_ORIG_DIRECT_L, :GD_STOGO=>GD_STOGO, :GD_STOGO_RAND=>GD_STOGO_RAND, :LD_LBFGS_NOCEDAL=>LD_LBFGS_NOCEDAL, :LD_LBFGS=>LD_LBFGS, :LN_PRAXIS=>LN_PRAXIS, :LD_VAR1=>LD_VAR1, :LD_VAR2=>LD_VAR2, :LD_TNEWTON=>LD_TNEWTON, :LD_TNEWTON_RESTART=>LD_TNEWTON_RESTART, :LD_TNEWTON_PRECOND=>LD_TNEWTON_PRECOND, :LD_TNEWTON_PRECOND_RESTART=>LD_TNEWTON_PRECOND_RESTART, :GN_CRS2_LM=>GN_CRS2_LM, :GN_MLSL=>GN_MLSL, :GD_MLSL=>GD_MLSL, :GN_MLSL_LDS=>GN_MLSL_LDS, :GD_MLSL_LDS=>GD_MLSL_LDS, :LD_MMA=>LD_MMA, :LN_COBYLA=>LN_COBYLA, :LN_NEWUOA=>LN_NEWUOA, :LN_NEWUOA_BOUND=>LN_NEWUOA_BOUND, :LN_NELDERMEAD=>LN_NELDERMEAD, :LN_SBPLX=>LN_SBPLX, :LN_AUGLAG=>LN_AUGLAG, :LD_AUGLAG=>LD_AUGLAG, :LN_AUGLAG_EQ=>LN_AUGLAG_EQ, :LD_AUGLAG_EQ=>LD_AUGLAG_EQ, :LN_BOBYQA=>LN_BOBYQA, :GN_ISRES=>GN_ISRES, :AUGLAG=>AUGLAG, :AUGLAG_EQ=>AUGLAG_EQ, :G_MLSL=>G_MLSL, :G_MLSL_LDS=>G_MLSL_LDS, :LD_SLSQP=>LD_SLSQP, :LD_CCSAQ=>LD_CCSAQ, :LD_ESCH=>LD_ESCH ]
const int2alg = (Cenum=>Symbol)[ alg2int[k]=>k for k in keys(alg2int) ]

# enum nlopt_result
const FAILURE = cenum(-1)
const INVALID_ARGS = cenum(-2)
const OUT_OF_MEMORY = cenum(-3)
const ROUNDOFF_LIMITED = cenum(-4)
const FORCED_STOP = cenum(-5)
const SUCCESS = cenum(1)
const STOPVAL_REACHED = cenum(2)
const FTOL_REACHED = cenum(3)
const XTOL_REACHED = cenum(4)
const MAXEVAL_REACHED = cenum(5)
const MAXTIME_REACHED = cenum(6)

const res2sym = (Cenum=>Symbol)[ FAILURE=>:FAILURE, INVALID_ARGS=>:INVALID_ARGS, OUT_OF_MEMORY=>:OUT_OF_MEMORY, ROUNDOFF_LIMITED=>:ROUNDOFF_LIMITED, FORCED_STOP=>:FORCED_STOP, SUCCESS=>:SUCCESS, STOPVAL_REACHED=>:STOPVAL_REACHED, FTOL_REACHED=>:FTOL_REACHED, XTOL_REACHED=>:XTOL_REACHED, MAXEVAL_REACHED=>:MAXEVAL_REACHED, MAXTIME_REACHED=>:MAXTIME_REACHED ]

############################################################################
# wrapper around nlopt_opt type

typealias _Opt Ptr{Void} # nlopt_opt

# pass both f and o to the callback so that we can handle exceptions
type Callback_Data
    f::Function
    o::Any # should be Opt, but see Julia issue #269
end

type Opt
    opt::_Opt

    # need to store callback data for objective and constraints in
    # Opt so that they aren't garbage-collected.  cb[1] is the objective.
    cb::Vector{Callback_Data}
    
    function Opt(p::_Opt)
        opt = new(p, Array(Callback_Data,1))
        finalizer(opt, destroy)
        opt
    end        
    function Opt(algorithm::Integer, n::Integer)
        if algorithm < 0 || algorithm > NUM_ALGORITHMS
            throw(ArgumentError("invalid algorithm $algorithm"))
        elseif n < 0
            throw(ArgumentError("invalid dimension $n < 0"))
        end
        p = ccall((:nlopt_create,:libnlopt), _Opt, (Cenum, Cuint),
                  algorithm, n)
        if p == C_NULL
            error("Error in nlopt_create")
        end
        Opt(p)
    end
    Opt(algorithm::Symbol, n::Integer) = Opt(try alg2int[algorithm]
                                             catch
                         throw(ArgumentError("unknown algorithm $algorithm"))
                                             end, n)
end

convert(::Type{_Opt}, o::Opt) = o.opt # for passing to ccall

destroy(o::Opt) = ccall((:nlopt_destroy,:libnlopt), Void, (_Opt,), o)

function copy(o::Opt)
    p = ccall((:nlopt_copy,:libnlopt), _Opt, (_Opt,), o)
    if p == C_NULL
        error("Error in nlopt_copy")
    end
    oc = Opt(p)
    oc.cb = similar(o.cb)
    for i = 1:length(o.cb)
        oc.cb[i] = Callback_Data(o.cb[i].f, oc)
    end
    return oc
end

ndims(o::Opt) = int(ccall((:nlopt_get_dimension,:libnlopt), Cuint, (_Opt,), o))
algorithm(o::Opt) = int2alg[ccall((:nlopt_get_algorithm,:libnlopt),
                                  Cenum, (_Opt,), o)]

show(io::IO, o::Opt) = print(io, "Opt(:$(algorithm(o)), $(ndims(o)))")


############################################################################
# converting error results into exceptions

type ForcedStop <: Exception end

# cache current exception for forced stop
nlopt_exception = nothing

# check result and throw an exception if necessary
function chk(result::Integer)
    if result < 0 && result != ROUNDOFF_LIMITED
        if result == INVALID_ARGS
            throw(ArgumentError("invalid NLopt arguments"))
        elseif result == OUT_OF_MEMORY
            throw(MemoryError())
        elseif result == FORCED_STOP
            global nlopt_exception
            e = nlopt_exception
            if e != nothing && !isa(e, ForcedStop)
                nlopt_exception = nothing
                rethrow(e)
            end
        else
            error("nlopt failure")
        end
    end
    result
end

chks(result::Integer) = res2sym[chk(result)]
chkn(result::Integer) = begin chk(result); nothing; end

############################################################################
# getting and setting scalar and vector parameters

# make a quoted symbol expression out of the arguments
qsym(args...) = Expr(:quote, symbol(string(args...)))

# scalar parameters p of type T
macro GETSET(T, p)
    Tg = T == :Cdouble ? :Real : (T == :Cint || T == :Cuint ? :Integer : :Any)
    ps = symbol(string(p, "!"))
    quote 
        $(esc(p))(o::Opt) = ccall(($(qsym("nlopt_get_", p)),:libnlopt),
                                  $T, (_Opt,), o)
        $(esc(ps))(o::Opt, val::$Tg) =
          chkn(ccall(($(qsym("nlopt_set_", p)),:libnlopt),
                     Cenum, (_Opt, $T), o, val))
    end
end

# Vector{Cdouble} parameters p
macro GETSET_VEC(p)
    ps = symbol(string(p, "!"))
    quote
        function $(esc(p))(o::Opt, v::Vector{Cdouble})
            if length(v) != ndims(o)
                throw(BoundsError())
            end
            chk(ccall(($(qsym("nlopt_get_", p)),:libnlopt),
                      Cenum, (_Opt, Ptr{Cdouble}), o, v))
            v
        end
        $(esc(p))(o::Opt) = $(esc(p))(o, Array(Cdouble, ndims(o)))
        function $(esc(ps))(o::Opt, v::Vector{Cdouble})
            if length(v) != ndims(o)
                throw(BoundsError())
            end
            chkn(ccall(($(qsym("nlopt_set_", p)),:libnlopt),
                      Cenum, (_Opt, Ptr{Cdouble}), o, v))
        end
        $(esc(ps)){T<:Real}(o::Opt, v::AbstractVector{T}) =
          $(esc(ps))(o, copy!(Array(Cdouble,length(v)), v))
        $(esc(ps))(o::Opt, val::Real) =
          chkn(ccall(($(qsym("nlopt_set_", p, "1")),:libnlopt),
                     Cenum, (_Opt, Cdouble), o, val))
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
  chkn(ccall((:nlopt_set_local_optimizer,:libnlopt),
             Cenum, (_Opt, _Opt), o, lo))

# the initial-stepsize stuff is a bit different than GETSET_VEC,
# since the heuristics depend on the position x.

function default_initial_step!(o::Opt, x::Vector{Cdouble})
    if length(x) != ndims(o)
        throw(BoundsError())
    end
    chkn(ccall((:nlopt_set_default_initial_step,:libnlopt),
               Cenum, (_Opt, Ptr{Cdouble}), o, x))
end
default_initial_step!{T<:Real}(o::Opt, x::AbstractVector{T}) = 
  default_initial_step!(o, copy!(Array(Cdouble,length(x)), x))

function initial_step!(o::Opt, dx::Vector{Cdouble})
    if length(dx) != ndims(o)
        throw(BoundsError())
    end
    chkn(ccall((:nlopt_set_initial_step,:libnlopt),
               Cenum, (_Opt, Ptr{Cdouble}), o, dx))
end
initial_step!{T<:Real}(o::Opt, dx::AbstractVector{T}) =
  initial_step!(o, copy!(Array(Cdouble,length(dx)), dx))
initial_step!(o::Opt, dx::Real) = 
  chkn(ccall((:nlopt_set_initial_step1,:libnlopt),
             Cenum, (_Opt, Cdouble), o, dx))

function initial_step(o::Opt, x::Vector{Cdouble}, dx::Vector{Cdouble})
    if length(x) != ndims(o) || length(dx) != ndims(o)
        throw(BoundsError())
    end
    chkn(ccall((:nlopt_get_initial_step,:libnlopt),
               Cenum, (_Opt, Ptr{Cdouble}, Ptr{Cdouble}), o, x, dx))
    dx
end
initial_step{T<:Real}(o::Opt, x::AbstractVector{T}) =
    initial_step(o, copy!(Array(Cdouble,length(x)), x), 
                 Array(Cdouble, ndims(o)))

############################################################################

function algorithm_name(a::Integer)
    s = ccall((:nlopt_algorithm_name,:libnlopt), Ptr{Uint8}, (Cenum,), a)
    if s == C_NULL
        throw(ArgumentError("invalid algorithm $a"))
    end
    return bytestring(s)
end

algorithm_name(a::Symbol) = algorithm_name(try alg2int[a]
                                           catch
                             throw(ArgumentError("unknown algorithm $a"))
                                           end)
algorithm_name(o::Opt) = algorithm_name(algorithm(o))

############################################################################

function version()
    v = Array(Cint, 3)
    pv = uint(pointer(v))
    ccall((:nlopt_version,:libnlopt), Void, (Uint,Uint,Uint),
          pv, pv + sizeof(Cint), pv + 2*sizeof(Cint))
    VersionNumber(int(v[1]),int(v[2]),int(v[3]))
end

const NLOPT_VERSION = version()

############################################################################

srand(seed::Integer) = ccall((:nlopt_srand,:libnlopt),
                             Void, (Culong,), seed)
srand_time() = ccall((:nlopt_srand_time,:libnlopt), Void, ())

############################################################################
# Objective function:

empty_grad = Cdouble[] # for passing when grad == C_NULL

function nlopt_callback_wrapper(n::Cuint, x::Ptr{Cdouble},
                                grad::Ptr{Cdouble}, d_::Ptr{Void})
    d = unsafe_pointer_to_objref(d_)::Callback_Data
    try
        res = convert(Cdouble,
                      d.f(pointer_to_array(x, (int(n),)), 
                          grad == C_NULL ? empty_grad::Vector{Cdouble}
                          : pointer_to_array(grad, (int(n),))))
        return res::Cdouble
    catch e
        global nlopt_exception
        nlopt_exception = e
        println("in callback catch")
        force_stop!(d.o::Opt)
        return 0.0 # ignored by nlopt
    end
end

for m in (:min, :max)
    mf = symbol(string(m,"_objective!"))
    @eval function $mf(o::Opt, f::Function)
        o.cb[1] = Callback_Data(f, o)
        chkn(ccall(($(qsym("nlopt_set_", m, "_objective")),:libnlopt),
                   Cenum, (_Opt, Ptr{Void}, Any),
                   o, cfunction(nlopt_callback_wrapper,
                                Cdouble, (Cuint, Ptr{Cdouble}, Ptr{Cdouble},
                                          Ptr{Void})), 
                   o.cb[1]))
    end
end

############################################################################
# Nonlinear constraints:

for c in (:inequality, :equality)
    cf = symbol(string(c, "_constraint!"))
    @eval function $cf(o::Opt, f::Function, tol::Real)
        push!(o.cb, Callback_Data(f, o))
        chkn(ccall(($(qsym("nlopt_add_", c, "_constraint")),:libnlopt),
                   Cenum, (_Opt, Ptr{Void}, Any, Cdouble),
                   o, cfunction(nlopt_callback_wrapper,
                                Cdouble, (Cuint, Ptr{Cdouble}, Ptr{Cdouble},
                                          Ptr{Void})),
                   o.cb[end], tol))
    end
    @eval $cf(o::Opt, f::Function) = $cf(o, f, 0.0)
end

function remove_constraints!(o::Opt)
    resize!(o.cb, 1)
    chkn(ccall((:nlopt_remove_inequality_constraints,:libnlopt),
               Cenum, (_Opt,), o))
    chkn(ccall((:nlopt_remove_equality_constraints,:libnlopt),
               Cenum, (_Opt,), o))
end

############################################################################
# Vector-valued constraints

function nlopt_vcallback_wrapper(m::Cuint, res::Ptr{Cdouble},
                                 n::Cuint, x::Ptr{Cdouble},
                                 grad::Ptr{Cdouble}, d_::Ptr{Void})
    try
        d = unsafe_pointer_to_objref(d_)::Callback_Data
        d.f(pointer_to_array(res, (int(m),)),
            pointer_to_array(x, (int(n),)),
            grad == C_NULL ? empty_grad::Vector{Cdouble}
            : pointer_to_array(grad, (int(n),int(m))))
    catch e
        global nlopt_exception
        nlopt_exception = e
        force_stop!(d.o::Opt)
    end
    nothing
end

for c in (:inequality, :equality)
    cf = symbol(string(c, "_constraint!"))
    @eval begin
        function $cf(o::Opt, f::Function, tol::Vector{Cdouble})
            push!(o.cb, Callback_Data(f, o))
            chkn(ccall(($(qsym("nlopt_add_", c, "_mconstraint")),
                        :libnlopt),
                       Cenum, (_Opt, Cuint, Ptr{Void}, Any, Ptr{Cdouble}),
                       o, length(tol),
                       cfunction(nlopt_vcallback_wrapper,
                                 Void, (Cuint, Ptr{Cdouble},
                                        Cuint, Ptr{Cdouble}, Ptr{Cdouble},
                                        Ptr{Void})),
                       o.cb[end], tol))
        end
        $cf{T<:Real}(o::Opt, f::Function, tol::AbstractVector{T}) =
           $cf(o, f, copy!(Array(Float64,length(tol)), tol))
        $cf(o::Opt, m::Integer, f::Function, tol::Real) =
           $cf(o, f, fill!(Array(Cdouble, m), tol))
        $cf(o::Opt, m::Integer, f::Function)=
           $cf(o, m, f, 0.0)
    end
end

############################################################################
# Perform the optimization:

function optimize!(o::Opt, x::Vector{Cdouble})
    if length(x) != ndims(o)
        throw(BoundsError())
    end
    opt_f = Array(Cdouble, 1)
    ret = ccall((:nlopt_optimize,:libnlopt), Cenum, (_Opt, Ptr{Cdouble},
                                                     Ptr{Cdouble}),
                o, x, opt_f)
    return (opt_f[1], x, chks(ret))
end

optimize{T<:Real}(o::Opt, x::AbstractVector{T}) =
  optimize!(o, copy!(Array(Cdouble,length(x)), x))

############################################################################

end # module
