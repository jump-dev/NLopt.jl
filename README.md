# The NLopt module for Julia
[![Build Status](https://travis-ci.org/JuliaOpt/NLopt.jl.svg?branch=master)](https://travis-ci.org/JuliaOpt/NLopt.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/eqw9yb2cyn8sxvo9?svg=true)](https://ci.appveyor.com/project/StevenGJohnson/nlopt-jl)

[![NLopt](http://pkg.julialang.org/badges/NLopt_0.3.svg)](http://pkg.julialang.org/?pkg=NLopt&ver=0.3)
[![NLopt](http://pkg.julialang.org/badges/NLopt_0.4.svg)](http://pkg.julialang.org/?pkg=NLopt&ver=0.4)
[![NLopt](http://pkg.julialang.org/badges/NLopt_0.5.svg)](http://pkg.julialang.org/?pkg=NLopt&ver=0.5)

This module provides a [Julia-language](http://julialang.org/) interface to
the free/open-source [NLopt library](http://ab-initio.mit.edu/nlopt) for
nonlinear optimization. NLopt provides a common interface for many different
optimization algorithms, including:

* Both global and local optimization
* Algorithms using function values only (derivative-free) and also algorithms
  exploiting user-supplied gradients.
* Algorithms for unconstrained optimization, bound-constrained optimization,
  and general nonlinear inequality/equality constraints.

See the [NLopt introduction](http://ab-initio.mit.edu/wiki/index.php/NLopt_Introduction)
for a further overview of the types of problems it addresses.

NLopt can be used either by accessing it's specialized API or by using the generic [MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl) interface for nonlinear
optimization. Both methods are documented below.

## Installation

Within Julia, you can install the NLopt.jl package with the package manager: `Pkg.add("NLopt")`

On Windows and OS X platforms, NLopt binaries will be automatically installed.
On other platforms, Julia will attempt to build NLopt from source;
be sure to have a compiler installed.

## Using with MathProgBase

NLopt implements the [MathProgBase interface](http://mathprogbasejl.readthedocs.org/en/latest/nlp.html) for nonlinear optimization, which means that it can be used interchangeably with other optimization packages from modeling packages like [JuMP](https://github.com/JuliaOpt/JuMP.jl) or when providing hand-written derivatives. Note that NLopt does not exploit sparsity of Jacobians.

The NLopt solver is named ``NLoptSolver`` and takes parameters:

 - ``algorithm``
 - ``stopval``
 - ``ftol_rel``
 - ``ftol_abs``
 - ``xtol_rel``
 - ``xtol_abs``
 - ``constrtol_abs``
 - ``maxeval``
 - ``maxtime``
 - ``initial_step``
 - ``population``
 - ``seed``
 - ``vector_storage``

The ``algorithm`` parameter is required, and all others are optional. The meaning and acceptable values of all parameters, except ``constrtol_abs``, match the descriptions below from the specialized NLopt API. The ``constrtol_abs`` parameter is an absolute feasibility tolerance applied to all constraints.

## Tutorial

The following example code solves the nonlinearly constrained minimization
problem from the [NLopt Tutorial](http://ab-initio.mit.edu/wiki/index.php/NLopt_Tutorial):

```julia
using NLopt

function myfunc(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5/sqrt(x[2])
    end
    return sqrt(x[2])
end

function myconstraint(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

opt = Opt(:LD_MMA, 2)
lower_bounds!(opt, [-Inf, 0.])
xtol_rel!(opt,1e-4)

min_objective!(opt, myfunc)
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

(minf,minx,ret) = optimize(opt, [1.234, 5.678])
count = opt.numevals # the number of function evaluations
println("got $minf at $minx after $count iterations (returned $ret)")
```

The output should be:

```
got 0.5443310476200902 at [0.3333333346933468,0.29629628940318486] after 11 iterations (returned XTOL_REACHED)
```

Much like the NLopt interfaces in other languages, you create an
`Opt` object (analogous to `nlopt_opt` in C) which encapsulates the
dimensionality of your problem (here, 2) and the algorithm to be used
(here, `LD_MMA`) and use various functions to specify the constraints
and stopping criteria (along with any other aspects of the problem).

The same problem can be solved by using the JuMP interface to NLopt:

```julia
using JuMP
using NLopt

m = Model(solver=NLoptSolver(algorithm=:LD_MMA))

a1 = 2
b1 = 0
a2 = -1
b2 = 1

@variable(m, x1)
@variable(m, x2 >= 0)

@NLobjective(m, Min, sqrt(x2))
@NLconstraint(m, x2 >= (a1*x1+b1)^3)
@NLconstraint(m, x2 >= (a2*x1+b2)^3)

setvalue(x1, 1.234)
setvalue(x2, 5.678)

status = solve(m)

println("got ", getobjectivevalue(m), " at ", [getvalue(x1),getvalue(x2)])
```
The output should be:
```
got 0.5443310477213124 at [0.3333333342139688,0.29629628951338166]
```

Note that the MathProgBase interface sets slightly different convergence tolerances by default,
so the outputs from the two problems are not identical.

## Reference

The main purpose of this section is to document the syntax and unique
features of the Julia interface; for more detail on the underlying
features, please refer to the C documentation in the [NLopt
Reference](http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference).

### Using the Julia API

To use NLopt in Julia, your Julia program should include the line:
```julia
using NLopt
```

which imports the NLopt module and its symbols.  (Alternatively, you
can use `import NLopt` if you want to keep all the NLopt symbols in
their own namespace.  You would then prefix all functions below with
`NLopt.`, e.g. `NLopt.Opt` and so on.)

### The `Opt` type

The NLopt API revolves around an object of type `Opt`. Via functions
acting on this object, all of the parameters of the optimization are
specified (dimensions, algorithm, stopping criteria, constraints,
objective function, etcetera), and then one finally calls the
`optimize` function in order to perform the optimization. The object
should normally be created via the constructor:
```julia
opt = Opt(algorithm, n)
```

given an algorithm (see [NLopt
Algorithms](http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms)
for possible values) and the dimensionality of the problem (`n`, the
number of optimization parameters). Whereas in C the algorithms are
specified by `nlopt_algorithm` constants of the form `NLOPT_LD_MMA`,
`NLOPT_LN_COBYLA`, etcetera, the Julia `algorithm` values are symbols
of the form `:LD_MMA`, `:LN_COBYLA`, etcetera (with the `NLOPT_` prefix
replaced by `:` to create a Julia symbol).

There is also a `copy(opt::Opt)` function to make a copy of a given
object (equivalent to `nlopt_copy` in the C API).

If there is an error in these functions, an exception is thrown.

The algorithm and dimension parameters of the object are immutable (cannot be changed without constructing a new object), but you can query them for a given object by the functions:
```julia
ndims(opt::Opt)
algorithm(opt::Opt)
```

You can get a string description of the algorithm via:
```julia
algorithm_name(opt::Opt)
```

### Objective function

The objective function is specified by calling one of the functions:
```julia
min_objective!(opt::Opt, f::Function)
max_objective!(opt::Opt, f::Function)
```

depending on whether one wishes to minimize or maximize the objective function `f`, respectively. The function `f` should be of the form:
```julia
function f(x::Vector, grad::Vector):
    if length(grad) > 0:
        ...set grad to gradient, in-place...
    return ...value of f(x)...
end
```

The return value should be the value of the function at the point `x`,
where `x` is a (`Float64`) array of length `n` of the optimization
parameters (the same as the dimension passed to the `Opt` constructor).

In addition, if the argument `grad` is not empty [i.e. `length(grad)`>0],
then `grad` is a (`Float64`) array of length `n` which should (upon return) be
set to the gradient of the function with respect to the optimization
parameters at `x`. That is, `grad[i]` should upon return contain the
partial derivative &part;`f`/&part;`x`<sub>`i`</sub>, for 1&le;`i`&le;`n`, if
`grad` is non-empty. Not all of the optimization algorithms (below) use
the gradient information: for algorithms listed as "derivative-free,"
the `grad` argument will always be empty and need never be
computed. (For algorithms that do use gradient information, however,
`grad` may still be empty for some calls.)

Note that `grad` must be modified *in-place* by your function `f`. Generally, this means using indexing operations `grad[...] = ...` to overwrite the contents of `grad`.  For example `grad = 2x` will *not* work, because it points `grad` to a new array `2x` rather than overwriting the old contents; instead, use an explicit loop or use `grad[:] = 2x`.

### Bound constraints

The bound constraints can be specified by calling the functions:
```julia
lower_bounds!(opt::Opt, lb::AbstractVector)
upper_bounds!(opt::Opt, ub::AbstractVector)
```

where `lb` and `ub` are real arrays of length `n` (the same as the
dimension passed to the `Opt` constructor). For convenience, these are
overloaded with functions that take a single number as arguments, in
order to set the lower/upper bounds for all optimization parameters to
a single constant.

To retrieve the values of the lower/upper bounds, you can call one of:
```julia
lower_bounds(opt::Opt)
upper_bounds(opt::Opt)
```

both of which return `Vector{Float64}` arrays.

To specify an unbounded dimension, you can use &plusmn;`Inf`.

### Nonlinear constraints

Just as for nonlinear constraints in C, you can specify nonlinear
inequality and equality constraints by the functions:
```julia
inequality_constraint!(opt::Opt, fc, tol=0)
equality_constraint!(opt::Opt, h, tol=0)
```

where the arguments `fc` and `h` have the same form as the objective
function above. The optional `tol` arguments specify a tolerance
(which defaults to zero) in judging feasibility for the purposes of
stopping the optimization, as in C.

To remove all of the inequality and equality constraints from a given problem, you can call the following functions:
```julia
remove_constraints!(opt::Opt)
```

### Vector-valued constraints

Just as for nonlinear constraints in C, you can specify vector-valued
nonlinear inequality and equality constraints by the functions
```julia
inequality_constraint!(opt::Opt, c, tol::AbstractVector)
equality_constraint!(opt::Opt, c, tol::AbstractVector)
```

Here, `tol` is an array of the tolerances in each constraint
dimension; the dimensionality `m` of the constraint is determined by
`length(tol)`. The constraint function `c` must be of the form:
```julia
function c(result::Vector, x::Vector, grad::Matrix):
    if length(grad) > 0:
        ...set grad to gradient, in-place...
    result[1] = ...value of c1(x)...
    result[2] = ...value of c2(x)...
    ...
```

`result` is a (`Float64`) array whose length equals the dimensionality
`m` of the constraint (same as the length of `tol` above), which upon
return should be set *in-place* (see also above) to the constraint
results at the point `x` (a `Float64` array whose length `n` is the
same as the dimension passed to the `Opt` constructor). Any return value of
the function is ignored.

In addition, if the argument `grad` is not empty
[i.e. `length(grad)`>0], then `grad` is a 2d array of size
`n`&times;`m` which should (upon return) be set in-place (see above)
to the gradient of the function with respect to the optimization
parameters at `x`. That is, `grad[j,i]` should upon return contain the
partial derivative &part;c<sub>`i`</sub>/&part;x<sub>`j`</sub> if
`grad` is non-empty. Not all of the optimization algorithms (below)
use the gradient information: for algorithms listed as
"derivative-free," the `grad` argument will always be empty and need
never be computed. (For algorithms that do use gradient information,
however, `grad` may still be empty for some calls.)

An inequality constraint corresponds to c<sub>`i`</sub>&le;0 for
1&le;`i`&le;`m`, and an equality constraint corresponds to
c<sub>i</sub>=0, in both cases with tolerance `tol[i]` for purposes of
termination criteria.

(You can add multiple vector-valued constraints and/or scalar
constraints in the same problem.)

### Stopping criteria

As explained in the [C API
Reference](http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference)
and the
[Introduction](http://ab-initio.mit.edu/wiki/index.php/NLopt_Introduction)),
you have multiple options for different stopping criteria that you can
specify. (Unspecified stopping criteria are disabled; i.e., they have
innocuous defaults.)

For each stopping criteria, there are (at least) two functions: a
"set" function (ending with `!`) to specify the stopping criterion,
and a "get" function to retrieve the current value for that
criterion. The meanings of each criterion are exactly the same as in
the C API.
```julia
stopval!(o::Opt, stopval::Real)
stopval(o::Opt)
```

Stop when an objective value of at least `stopval` is found.
```julia
ftol_rel!(o::Opt, tol::Real)
ftol_rel(o::Opt)
```

Set relative tolerance on function value.
```julia
ftol_abs!(o::Opt, tol::Real)
ftol_abs(o::Opt)
```

Set absolute tolerance on function value.
```julia
xtol_rel!(o::Opt, tol::Real)
xtol_rel(o::Opt)
```

Set relative tolerance on optimization parameters.
```julia
xtol_abs!(o::Opt, tol)
xtol_abs(o::Opt)
```

Set absolute tolerances on optimization parameters. The `tol` input must be an array of length `n` (the dimension specified in the `Opt` constructor); alternatively, you can pass a single number in order to set the same tolerance for all optimization parameters. `xtol_abs(o)` returns the tolerances as a array.
```julia
maxeval!(o::Opt, mev::Integer)
maxeval(o::Opt)
```

Stop when the number of function evaluations exceeds `mev`. (0 or
negative for no limit.)
```julia
maxtime!(o::Opt, t::Real)
maxtime(o::Opt)
```

Stop when the optimization time (in seconds) exceeds `t`. (0 or
negative for no limit.)

### Forced termination

In certain cases, the caller may wish to force the optimization to
halt, for some reason unknown to NLopt. For example, if the user
presses Ctrl-C, or there is an error of some sort in the objective
function. You can do this by throwing any exception inside your
objective/constraint functions: the optimization will be halted
gracefully, and the same exception will be thrown to the caller. See
below regarding exceptions. The Julia equivalent of `nlopt_forced_stop`
from the C API is to throw a `ForcedStop` exception.

### Performing the optimization

Once all of the desired optimization parameters have been specified in
a given object `opt::Opt`, you can perform the optimization by calling:
```julia
(optf,optx,ret) = optimize(opt::Opt, x::AbstractVector)
```

On input, `x` is an array of length `n` (the dimension of the problem
from the `Opt` constructor) giving an initial guess for the
optimization parameters. The return value `optx` is a array containing
the optimized values of the optimization parameters. `optf` contains
the optimized value of the objective function, and `ret` contains a
symbol indicating the NLopt return code (below).

Alternatively,
```julia
(optf,optx,ret) = optimize!(opt::Opt, x::Vector{Float64})
```

is the same but modifies `x` in-place (as well as returning `optx=x`).

On failure (negative return codes), optimize() throws an
exception (see Exceptions, below).

### Return values

The possible return values are the same as the [return values in the C
API](http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Return_values),
except that the `NLOPT_` prefix is replaced with `:`.  That is, the return values are `:SUCCESS`, `:XTOL_REACHED`, etcetera (instead of `NLOPT_SUCCESS` etcetera).

### Exceptions

The error codes in the C API are replaced in the Julia API by thrown
exceptions. The following exceptions are thrown by the various
routines:

If your objective/constraint functions throw any exception during the
execution of `optimize`, it will be caught by NLopt and the
optimization will be halted gracefully, and opt.optimize will re-throw
the same exception to its caller.

### Local/subsidiary optimization algorithm

Some of the algorithms, especially MLSL and AUGLAG, use a different
optimization algorithm as a subroutine, typically for local
optimization. You can change the local search algorithm and its
tolerances by calling:
```julia
local_optimizer!(opt::Opt, local_opt::Opt)
```

Here, `local_opt` is another `Opt` object whose parameters are used to determine the local search algorithm, its stopping criteria, and other algorithm parameters. (However, the objective function, bounds, and nonlinear-constraint parameters of `local_opt` are ignored.) The dimension `n` of `local_opt` must match that of `opt`.

This function makes a copy of the `local_opt` object, so you can freely change your original `local_opt` afterwards without affecting `opt`.

### Initial step size

Just [as in the C
API](http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Initial_step_size),
you can get and set the initial step sizes for derivative-free
optimization algorithms. The Julia equivalents of the C functions are
the following functions:
```julia
initial_step!(opt::Opt, dx::AbstractVector)
```

Here, dx is an array of the (nonzero) initial steps for each
dimension, or a single number if you wish to use the same initial
steps for all dimensions. `get_initial_step(opt::Opt,
x::AbstractVector)` returns the initial step that will be used for a
starting guess of `x` in `optimize(opt,x)`.

### Stochastic population

Just [as in the C
API](http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Stochastic_population),
you can get and set the initial population for stochastic optimization
algorithms, by the functions:
```julia
population!(opt::Opt, pop::Integer)
population(opt::Opt)
```

(A `pop` of zero implies that the heuristic default will be used.)

### Pseudorandom numbers

For stochastic optimization algorithms, NLopt uses pseudorandom numbers
generated by the Mersenne Twister algorithm, based on code from Makoto
Matsumoto. By default, the seed for the random numbers is generated
from the system time, so that you will get a different sequence of
pseudorandom numbers each time you run your program. If you want to
use a "deterministic" sequence of pseudorandom numbers, i.e. the same
sequence from run to run, you can set the seed by calling:
```julia
NLopt.srand(seed::Integer)
```

To reset the seed based on the system time, you can call `NLopt.srand_time()`.

(Normally, you don't need to call this as it is called
automatically. However, it might be useful if you want to
"re-randomize" the pseudorandom numbers after calling `nlopt.srand` to
set a deterministic seed.)

### Vector storage for limited-memory quasi-Newton algorithms

Just [as in the C API](http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Vector_storage_for_limited-memory_quasi-Newton_algorithms), you can get and set the number M of stored vectors for limited-memory quasi-Newton algorithms, via the functions:
```julia
vector_storage!(opt::Opt, M::Integer)
vector_storage(opt::Opt)
```

(The default is `M`=0, in which case NLopt uses a heuristic nonzero value.)

### Version number

The version number of NLopt is given by the global variable:
```julia
NLOPT_VERSION::VersionNumber
```

where `VersionNumber` is a built-in Julia type from the Julia standard library.

## Author

This module was initially written by [Steven G. Johnson](http://math.mit.edu/~stevenj/),
with subsequent contributions by several other authors (see the git history).
