# NLopt.jl

[![Build Status](https://github.com/jump-dev/NLopt.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jump-dev/NLopt.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/jump-dev/NLopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/NLopt.jl)

[NLopt.jl](https://github.com/jump-dev/NLopt.jl) is a wrapper for the
[NLopt](https://nlopt.readthedocs.io/en/latest/) library for nonlinear
optimization.

NLopt provides a common interface for many different optimization algorithms,
including:

* Both global and local optimization
* Algorithms using function values only (derivative-free) and also algorithms
  exploiting user-supplied gradients.
* Algorithms for unconstrained optimization, bound-constrained optimization,
  and general nonlinear inequality/equality constraints.

## License

`NLopt.jl` is licensed under the [MIT License](https://github.com/jump-dev/NLopt.jl/blob/master/LICENSE.md).

The underlying solver, [stevengj/nlopt](https://github.com/stevengj/nlopt), is
licensed under the [LGPL v3.0 license](https://github.com/stevengj/nlopt/blob/master/COPYING).

## Installation

Install `NLopt.jl` using the Julia package manager:
```julia
import Pkg
Pkg.add("NLopt")
```

In addition to installing the `NLopt.jl` package, this will also download and
install the NLopt binaries. You do not need to install NLopt separately.

## Tutorial

The following example code solves the nonlinearly constrained minimization
problem from the [NLopt Tutorial](https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/).

```julia
using NLopt
function my_objective_fn(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5 / sqrt(x[2])
    end
    return sqrt(x[2])
end
function my_constraint_fn(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3 * a * (a * x[1] + b)^2
        grad[2] = -1
    end
    return (a * x[1] + b)^3 - x[2]
end
opt = NLopt.Opt(:LD_MMA, 2)
NLopt.lower_bounds!(opt, [-Inf, 0.0])
NLopt.xtol_rel!(opt, 1e-4)
NLopt.min_objective!(opt, my_objective_fn)
NLopt.inequality_constraint!(opt, (x, g) -> my_constraint_fn(x, g, 2, 0), 1e-8)
NLopt.inequality_constraint!(opt, (x, g) -> my_constraint_fn(x, g, -1, 1), 1e-8)
min_f, min_x, ret = NLopt.optimize(opt, [1.234, 5.678])
num_evals = NLopt.numevals(opt)
println(
    """
    objective value       : $min_f
    solution              : $min_x
    solution status       : $ret
    # function evaluation : $num_evals
    """
)
```

The output is:

```
objective value       : 0.5443310477213124
solution              : [0.3333333342139688, 0.29629628951338166]
solution status       : XTOL_REACHED
# function evaluation : 11
```

### Algorithm-specific parameters

Some algorithms have algorithm-specific parameters that can be set via
`NLopt.nlopt_set_param`. For example:

```julia
julia> import NLopt

julia> opt = NLopt.Opt(:LD_MMA, 2)
Opt(LD_MMA, 2)

julia> NLopt.nlopt_set_param(opt, "inner_maxeval", 5)
NLOPT_SUCCESS::nlopt_result = 1
```

Consult the [NLopt documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms)
for the list of parameters supported by each algorithm.

### Trace iterations

A common feature request is for a callback that can used to trace the solution
over the iterations of the optimizer.

There is no native support for this in NLopt. Instead, add the callback to your
objective function.

```julia
julia> using NLopt

julia> begin
           trace = Any[]
           function my_objective_fn(x::Vector, grad::Vector)
               if length(grad) > 0
                   grad[1] = 0
                   grad[2] = 0.5 / sqrt(x[2])
               end
               value = sqrt(x[2])
               push!(trace, copy(x) => value)
               return value
           end
           function my_constraint_fn(x::Vector, grad::Vector, a, b)
               if length(grad) > 0
                   grad[1] = 3 * a * (a * x[1] + b)^2
                   grad[2] = -1
               end
               return (a * x[1] + b)^3 - x[2]
           end
           opt = NLopt.Opt(:LD_MMA, 2)
           NLopt.lower_bounds!(opt, [-Inf, 0.0])
           NLopt.xtol_rel!(opt, 1e-4)
           NLopt.min_objective!(opt, my_objective_fn)
           NLopt.inequality_constraint!(opt, (x, g) -> my_constraint_fn(x, g, 2, 0), 1e-8)
           NLopt.inequality_constraint!(opt, (x, g) -> my_constraint_fn(x, g, -1, 1), 1e-8)
           min_f, min_x, ret = NLopt.optimize(opt, [1.234, 5.678])
       end
(0.5443310477213124, [0.3333333342139688, 0.29629628951338166], :XTOL_REACHED)

julia> trace
11-element Vector{Any}:
                            [1.234, 5.678] => 2.382855429941145
   [0.8787394664016357, 5.551370325142423] => 2.3561346152421816
   [0.8262160034228196, 5.043903787432386] => 2.245863706334912
  [0.4739440370386794, 4.0767726724255375] => 2.0191019470114773
    [0.35389779634506047, 3.0308503583016] => 1.7409337604577608
 [0.33387310647853335, 1.9717933962872487] => 1.4042056104029954
  [0.3333337209575201, 1.0450874902862517] => 1.0222952070152005
 [0.33333357431034494, 0.4695027039311135] => 0.6852026736164369
  [0.3333332772332185, 0.3057923933552822] => 0.5529849847466767
 [0.33333339455750244, 0.2963215980646768] => 0.5443542946139737
 [0.3333333342139688, 0.29629628951338166] => 0.5443310477213124
```

## Use with JuMP

NLopt implements the [MathOptInterface interface](https://jump.dev/MathOptInterface.jl/stable/reference/nonlinear/)
for nonlinear optimization, which means that it can be used interchangeably with
other optimization packages from modeling packages like
[JuMP](https://github.com/jump-dev/JuMP.jl). Note that NLopt does not exploit
sparsity of Jacobians.

You can use NLopt with JuMP as follows:
```julia
using JuMP, NLopt
model = Model(NLopt.Optimizer)
set_attribute(model, "algorithm", :LD_MMA)
set_attribute(model, "xtol_rel", 1e-4)
set_attribute(model, "constrtol_abs", 1e-8)
@variable(model, x[1:2])
set_lower_bound(x[2], 0.0)
set_start_value.(x, [1.234, 5.678])
@NLobjective(model, Min, sqrt(x[2]))
@NLconstraint(model, (2 * x[1] + 0)^3 - x[2] <= 0)
@NLconstraint(model, (-1 * x[1] + 1)^3 - x[2] <= 0)
optimize!(model)
min_f, min_x, ret = objective_value(model), value.(x), raw_status(model)
println(
    """
    objective value       : $min_f
    solution              : $min_x
    solution status       : $ret
    """
)
```

The output is:

```
objective value       : 0.5443310477213124
solution              : [0.3333333342139688, 0.29629628951338166]
solution status       : XTOL_REACHED
```


The `algorithm` attribute is required. The value must be one of the supported
[NLopt algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/).

Other parameters include `stopval`, `ftol_rel`, `ftol_abs`, `xtol_rel`,
`xtol_abs`, `constrtol_abs`, `maxeval`, `maxtime`, `initial_step`, `population`,
`seed`, and `vector_storage`.

The ``algorithm`` parameter is required, and all others are optional. The
meaning and acceptable values of all parameters, except `constrtol_abs`, match
the descriptions below from the specialized NLopt API.

The `constrtol_abs` parameter is an absolute feasibility tolerance applied to
all constraints.

## Automatic differentiation

Some algorithms in NLopt require derivatives, which you must manually provide
in the `if length(grad) > 0` branch of your objective and constraint functions.

To stay simple and lightweight, NLopt does not provide ways to automatically
compute derivatives. If you do not have analytic expressions for the derivatives,
use a package such as [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
to compute automatic derivatives.

Here is an example of how to wrap a function `f(x::Vector)` using ForwardDiff so
that it is compatible with NLopt:
```julia
using NLopt
import ForwardDiff
function autodiff(f::Function)
    function nlopt_fn(x::Vector, grad::Vector)
        if length(grad) > 0
            # Use ForwardDiff to compute the gradient. Replace with your
            # favorite Julia automatic differentiation package.
            ForwardDiff.gradient!(grad, f, x)
        end
        return f(x)
    end
end
# These functions do not implement `grad`:
my_objective_fn(x::Vector) = sqrt(x[2]);
my_constraint_fn(x::Vector, a, b) = (a * x[1] + b)^3 - x[2];
opt = NLopt.Opt(:LD_MMA, 2)
NLopt.lower_bounds!(opt, [-Inf, 0.0])
NLopt.xtol_rel!(opt, 1e-4)
# But we wrap them in autodiff before passing to NLopt:
NLopt.min_objective!(opt, autodiff(my_objective_fn))
NLopt.inequality_constraint!(opt, autodiff(x -> my_constraint_fn(x, 2, 0)), 1e-8)
NLopt.inequality_constraint!(opt, autodiff(x -> my_constraint_fn(x, -1, 1)), 1e-8)
min_f, min_x, ret = NLopt.optimize(opt, [1.234, 5.678])
# (0.5443310477213124, [0.3333333342139688, 0.29629628951338166], :XTOL_REACHED)
```

## Reference

The main purpose of this section is to document the syntax and unique features
of the Julia interface. For more detail on the underlying features, please refer
to the C documentation in the [NLopt Reference](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/).

### Using the Julia API

To use NLopt in Julia, your Julia program should include the line:
```julia
using NLopt
```
which imports the NLopt module and its symbols.  Alternatively, you can use
`import NLopt` if you want to keep all the NLopt symbols in their own namespace.
You would then prefix all functions below with `NLopt.`, for example `NLopt.Opt` and so
on.

### The `Opt` type

The NLopt API revolves around an object of type `Opt`.

The object should normally be created via the constructor:
```julia
opt = Opt(algorithm::Symbol, n::Int)
```
given an algorithm (see [NLopt Algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)
for possible values) and the dimensionality of the problem (`n`, the number of
optimization parameters).

Whereas in C the algorithms are specified by `nlopt_algorithm` constants of the
form like `NLOPT_LD_MMA`, the Julia `algorithm` values are symbols of the form
`:LD_MMA` with the `NLOPT_` prefix replaced by `:` to create a Julia symbol.

There is also a `copy(opt::Opt)` function to make a copy of a given object
(equivalent to `nlopt_copy` in the C API).

If there is an error in these functions, an exception is thrown.

The algorithm and dimension parameters of the object are immutable (cannot be
changed without constructing a new object). Query them using:
```julia
ndims(opt::Opt)
algorithm(opt::Opt)
```

Get a string description of the algorithm via:
```julia
algorithm_name(opt::Opt)
```

### Objective function

The objective function is specified by calling one of:
```julia
min_objective!(opt::Opt, f::Function)
max_objective!(opt::Opt, f::Function)
```
depending on whether one wishes to minimize or maximize the objective function
`f`, respectively.

The function `f` must be of the form:
```julia
function f(x::Vector{Float64}, grad::Vector{Float64})
    if length(grad) > 0
        ...set grad to gradient, in-place...
    end
    return ...value of f(x)...
end
```

The return value must be the value of the function at the point `x`, where `x`
is a `Vector{Float64}` array of length `n` of the optimization parameters.

In addition, if the argument `grad` is not empty (that is, `length(grad) > 0`),
then `grad` is a `Vector{Float64}` array of length `n` which should (upon
return) be set to the gradient of the function with respect to the optimization
parameters at `x`.

Not all of the optimization algorithms (below) use the gradient information: for
algorithms listed as "derivative-free," the `grad` argument will always be empty
and need never be computed. For algorithms that do use gradient information,
`grad` may still be empty for some calls.

Note that `grad` must be modified *in-place* by your function `f`. Generally,
this means using indexing operations `grad[...] = ...` to overwrite the contents
of `grad`.  For example `grad = 2x` will *not* work, because it points `grad` to
a new array `2x` rather than overwriting the old contents; instead, use an
explicit loop or use `grad[:] = 2x`.

### Bound constraints

Add bound constraints with:
```julia
lower_bounds!(opt::Opt, lb::Union{AbstractVector,Real})
upper_bounds!(opt::Opt, ub::Union{AbstractVector,Real})
```
where `lb` and `ub` are real arrays of length `n` (the same as the dimension
passed to the `Opt` constructor).

For convenience, you can instead use a single scalar for `lb` or `ub` in order
to set the lower/upper bounds for all optimization parameters to a single
constant.

To retrieve the values of the lower or upper bounds, use:
```julia
lower_bounds(opt::Opt)
upper_bounds(opt::Opt)
```
both of which return `Vector{Float64}` arrays.

To specify an unbounded dimension, you can use `Inf` or `-Inf`.

### Nonlinear constraints

Specify nonlinear inequality and equality constraints by the functions:
```julia
inequality_constraint!(opt::Opt, f::Function, tol::Real = 0.0)
equality_constraint!(opt::Opt, f::Function, tol::Real = 0.0)
```
where the arguments `f` have the same form as the objective function above.

The optional `tol` arguments specify a tolerance (which defaults to zero) that
is used to judge feasibility for the purposes of stopping the optimization.

Each call to these function *adds* a new constraint to the set of constraints,
rather than replacing the constraints.

Remove all of the inequality and equality constraints from a given problem with:
```julia
remove_constraints!(opt::Opt)
```

### Vector-valued constraints

Specify vector-valued nonlinear inequality and equality constraints by the
functions:
```julia
inequality_constraint!(opt::Opt, f::Function, tol::AbstractVector)
equality_constraint!(opt::Opt, f::Function, tol::AbstractVector)
```
where `tol` is an array of the tolerances in each constraint dimension; the
dimensionality `m` of the constraint is determined by `length(tol)`.

The constraint function `f` must be of the form:
```julia
function f(result::Vector{Float64}, x::Vector{Float64}, grad::Matrix{Float64})
    if length(grad) > 0
        ...set grad to gradient, in-place...
    end
    result[1] = ...value of c1(x)...
    result[2] = ...value of c2(x)...
    return
```
where  `result` is a `Vector{Float64}` array whose length equals the
dimensionality `m` of the constraint (same as the length of `tol` above), which
upon return, should be set *in-place* to the constraint results at the point `x`.
Any return value of the function is ignored.

In addition, if the argument `grad` is not empty (that is, `length(grad) > 0`),
then `grad` is a matrix of size `n`&times;`m` which should (upon return) be
set in-place (see above) to the gradient of the function with respect to the
optimization parameters at `x`. That is, `grad[j,i]` should upon return contain
the partial derivative &part;f<sub>`i`</sub>/&part;x<sub>`j`</sub>.

A full example is:
```julia
julia> using NLopt

julia> function my_objective_fn(x::Vector, grad::Vector)
           if length(grad) > 0
               grad[1] = 0
               grad[2] = 0.5 / sqrt(x[2])
           end
           return sqrt(x[2])
       end
my_objective_fn (generic function with 1 method)

julia> function my_constraint_fn(x::Vector, grad::Vector, a, b)
           if length(grad) > 0
               grad[1] = 3 * a * (a * x[1] + b)^2
               grad[2] = -1
           end
           return (a * x[1] + b)^3 - x[2]
       end
my_constraint_fn (generic function with 1 method)

julia> function constraints(result::Vector, x::Vector, grad::Matrix)
           ∇g1 = [0.0, 0.0]
           ∇g2 = [0.0, 0.0]
           result[1] = my_constraint_fn(x, ∇g1, 2, 0)
           result[2] = my_constraint_fn(x, ∇g2, -1, 1)
           if length(grad) > 0
               # Note the `.=`. You must modify `grad` in-place
               grad .= vcat(∇g1', ∇g1')
           end
           return
       end
constraints (generic function with 2 methods)

julia> opt = NLopt.Opt(:LD_MMA, 2)
Opt(LD_MMA, 2)

julia> NLopt.lower_bounds!(opt, [-Inf, 0.0])

julia> NLopt.xtol_rel!(opt, 1e-6)

julia> NLopt.min_objective!(opt, my_objective_fn)

julia> NLopt.inequality_constraint!(opt, constraints, fill(1e-8, 2))

julia> min_f, min_x, ret = NLopt.optimize(opt, [1.234, 5.678])
(0.5443310692851157, [0.33333332182948433, 0.29629631298907744], :XTOL_REACHED)

julia> num_evals = NLopt.numevals(opt)
45

julia> println(
           """
           objective value       : $min_f
           solution              : $min_x
           solution status       : $ret
           # function evaluation : $num_evals
           """
       )
objective value       : 0.5443310692851157
solution              : [0.33333332182948433, 0.29629631298907744]
solution status       : XTOL_REACHED
# function evaluation : 45
```

Not all of the optimization algorithms (below) use the gradient information: for
algorithms listed as "derivative-free," the `grad` argument will always be empty
and need never be computed. For algorithms that do use gradient information,
`grad` may still be empty for some calls.

You can add multiple vector-valued constraints and/or scalar constraints in the
same problem.

### Stopping criteria

As explained in the [C API Reference](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/)
and the [Introduction](https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/),
you have multiple options for different stopping criteria that you can specify.
(Unspecified stopping criteria are disabled; that is, they have innocuous
defaults.)

For each stopping criteria, there are two functions that you can use to get and
set the value of the stopping criterion.

```julia
stopval(opt::Opt)          # return the current value of `stopval`
stopval!(opt::Opt, value)  # set stopval to `value`
```
Stop when an objective value of at least `stopval` is found. (Defaults to `-Inf`.)

```julia
ftol_rel(opt::Opt)
ftol_rel!(opt::Opt, value)
```
Relative tolerance on function value. (Defaults to `0`.)

```julia
ftol_abs(opt::Opt)
ftol_abs!(opt::Opt, value)
```
Absolute tolerance on function value. (Defaults to `0`.)

```julia
xtol_rel(opt::Opt)
xtol_rel!(opt::Opt, value)
```
Relative tolerances on the optimization parameters. (Defaults to `0`.)

```julia
xtol_abs(opt::Opt)
xtol_abs!(opt::Opt, value)
```
Absolute tolerances on the optimization parameters. (Defaults to `0`.)

In the case of `xtol_abs`, you can either set it to a scalar (to use the same
tolerance for all inputs) or a vector of length `n` (the dimension specified in
the `Opt` constructor) to use a different tolerance for each parameter.

```julia
maxeval(opt::Opt)
maxeval!(opt::Opt, value)
```
Stop when the number of function evaluations exceeds `mev`. (0 or negative for
no limit, which is the default.)

```julia
maxtime(opt::Opt)
maxtime!(opt::Opt, value)
```
Stop when the optimization time (in seconds) exceeds `t`. (0 or negative for no
limit, which is the default.)

### Forced termination

In certain cases, the caller may wish to force the optimization to halt, for
some reason unknown to NLopt. For example, if the user presses Ctrl-C, or there
is an error of some sort in the objective function. You can do this by throwing
any exception inside your objective/constraint functions: the optimization will
be halted gracefully, and the same exception will be thrown to the caller. The
Julia equivalent of `nlopt_forced_stop` from the C API is to throw a `ForcedStop`
exception.

### Performing the optimization

Once all of the desired optimization parameters have been specified in a given
object `opt::Opt`, you can perform the optimization by calling:
```julia
optf, optx, ret = optimize(opt::Opt, x::AbstractVector)
```

On input, `x` is an array of length `n` (the dimension of the problem from the
`Opt` constructor) giving an initial guess for the optimization parameters. The
return value `optx` is a array containing the optimized values of the
optimization parameters. `optf` contains the optimized value of the objective
function, and `ret` contains a symbol indicating the NLopt return code (below).

Alternatively:
```julia
optf, optx, ret = optimize!(opt::Opt, x::Vector{Float64})
```
is the same but modifies `x` in-place (as well as returning `optx = x`).

### Return values

The possible return values are the same as the [return values in the C API](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#Return_values),
except that the `NLOPT_` prefix is replaced with `:`.  That is, the return
values are like `:SUCCESS` instead `NLOPT_SUCCESS`.

### Local/subsidiary optimization algorithm

Some of the algorithms, especially `MLSL` and `AUGLAG`, use a different
optimization algorithm as a subroutine, typically for local optimization. You
can change the local search algorithm and its tolerances by setting:
```julia
local_optimizer!(opt::Opt, local_opt::Opt)
```

Here, `local_opt` is another `Opt` object whose parameters are used to determine
the local search algorithm, its stopping criteria, and other algorithm
parameters. (However, the objective function, bounds, and nonlinear-constraint
parameters of `local_opt` are ignored.) The dimension `n` of `local_opt` must
match that of `opt`.

This makes a copy of the `local_opt` object, so you can freely change your
original `local_opt` afterwards without affecting `opt`.

### Initial step size

Just [as in the C API](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#Initial_step_size),
you can set the initial step sizes for derivative-free optimization algorithms
with:
```julia
initial_step!(opt::Opt, dx::Vector)
```
Here, `dx` is an array of the (nonzero) initial steps for each dimension, or a
single number if you wish to use the same initial steps for all dimensions.

`initial_step(opt::Opt, x::AbstractVector)` returns the initial step that will
be used for a starting guess of `x` in `optimize(opt, x)`.

### Stochastic population

Just [as in the C API](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#Stochastic_population),
you can get and set the initial population for stochastic optimization with:
```julia
population(opt::Opt)
population!(opt::Opt, value)
```
A `population` of zero, the default, implies that the heuristic default will be
used as decided upon by individual algorithms.

### Pseudorandom numbers

For stochastic optimization algorithms, NLopt uses pseudorandom numbers
generated by the Mersenne Twister algorithm, based on code from Makoto Matsumoto.

By default, the seed for the random numbers is generated from the system time,
so that you will get a different sequence of pseudorandom numbers each time you
run your program. If you want to use a "deterministic" sequence of pseudorandom
numbers, that is, the same sequence from run to run, you can set the seed by
calling:
```julia
NLopt.srand(seed::Integer)
```
To reset the seed based on the system time, you can call `NLopt.srand_time()`.

Normally, you don't need to call this as it is called automatically. However, it
might be useful if you want to "re-randomize" the pseudorandom numbers after
calling `nlopt.srand` to set a deterministic seed.

### Vector storage for limited-memory quasi-Newton algorithms

Just [as in the C API](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#Vector_storage_for_limited-memory_quasi-Newton_algorithms),
you can get and set the number M of stored vectors for limited-memory
quasi-Newton algorithms, via integer-valued property
```julia
vector_storage(opt::Opt)
vector_storage!(opt::Opt, value)
```
The default is `0`, in which case NLopt uses a heuristic nonzero value as
determined by individual algorithms.

### Version number

The version number of NLopt is given by the global variable:
```julia
NLOPT_VERSION::VersionNumber
```
where `VersionNumber` is a built-in Julia type from the Julia standard library.

## Thread safety

The underlying NLopt library is threadsafe; however, re-using the same `Opt`
object across multiple threads is not.

As an example, instead of:
```julia
using NLopt
opt = Opt(:LD_MMA, 2)
# Define problem
solutions = Vector{Any}(undef, 10)
Threads.@threads for i in 1:10
    # Not thread-safe because `opt` is re-used
    solutions[i] = optimize(opt, rand(2))
end
```
Do instead:
```julia
solutions = Vector{Any}(undef, 10)
Threads.@threads for i in 1:10
    # Thread-safe because a new `opt` is created for each thread
    opt = Opt(:LD_MMA, 2)
    # Define problem
    solutions[i] = optimize(opt, rand(2))
end
```

## Author

This module was initially written by [Steven G. Johnson](http://math.mit.edu/~stevenj/),
with subsequent contributions by several other authors (see the git history).
