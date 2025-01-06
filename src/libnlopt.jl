# Copyright (c) 2013: Steven G. Johnson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

#! format: off

using CEnum

# typedef double ( * nlopt_func ) ( unsigned n , const double * x , double * gradient , /* NULL if not needed */ void * func_data )
const nlopt_func = Ptr{Cvoid}

# typedef void ( * nlopt_mfunc ) ( unsigned m , double * result , unsigned n , const double * x , double * gradient , /* NULL if not needed */ void * func_data )
const nlopt_mfunc = Ptr{Cvoid}

# typedef void ( * nlopt_precond ) ( unsigned n , const double * x , const double * v , double * vpre , void * data )
const nlopt_precond = Ptr{Cvoid}

@static if _is_version_newer_than_2_9()
    @cenum nlopt_algorithm::UInt32 begin
        NLOPT_GN_DIRECT = 0
        NLOPT_GN_DIRECT_L
        NLOPT_GN_DIRECT_L_RAND
        NLOPT_GN_DIRECT_NOSCAL
        NLOPT_GN_DIRECT_L_NOSCAL
        NLOPT_GN_DIRECT_L_RAND_NOSCAL
        NLOPT_GN_ORIG_DIRECT
        NLOPT_GN_ORIG_DIRECT_L
        NLOPT_GD_STOGO
        NLOPT_GD_STOGO_RAND
        # NLOPT_LD_LBFGS_NOCEDAL = 10
        NLOPT_LD_LBFGS
        NLOPT_LN_PRAXIS
        NLOPT_LD_VAR1
        NLOPT_LD_VAR2
        NLOPT_LD_TNEWTON
        NLOPT_LD_TNEWTON_RESTART
        NLOPT_LD_TNEWTON_PRECOND
        NLOPT_LD_TNEWTON_PRECOND_RESTART
        NLOPT_GN_CRS2_LM
        NLOPT_GN_MLSL
        NLOPT_GD_MLSL
        NLOPT_GN_MLSL_LDS
        NLOPT_GD_MLSL_LDS
        NLOPT_LD_MMA
        NLOPT_LN_COBYLA
        NLOPT_LN_NEWUOA
        NLOPT_LN_NEWUOA_BOUND
        NLOPT_LN_NELDERMEAD
        NLOPT_LN_SBPLX
        NLOPT_LN_AUGLAG
        NLOPT_LD_AUGLAG
        NLOPT_LN_AUGLAG_EQ
        NLOPT_LD_AUGLAG_EQ
        NLOPT_LN_BOBYQA
        NLOPT_GN_ISRES
        NLOPT_AUGLAG
        NLOPT_AUGLAG_EQ
        NLOPT_G_MLSL
        NLOPT_G_MLSL_LDS
        NLOPT_LD_SLSQP
        NLOPT_LD_CCSAQ
        NLOPT_GN_ESCH
        NLOPT_GN_AGS
        NLOPT_NUM_ALGORITHMS
    end
else
    @cenum nlopt_algorithm::UInt32 begin
        NLOPT_GN_DIRECT = 0
        NLOPT_GN_DIRECT_L
        NLOPT_GN_DIRECT_L_RAND
        NLOPT_GN_DIRECT_NOSCAL
        NLOPT_GN_DIRECT_L_NOSCAL
        NLOPT_GN_DIRECT_L_RAND_NOSCAL
        NLOPT_GN_ORIG_DIRECT
        NLOPT_GN_ORIG_DIRECT_L
        NLOPT_GD_STOGO
        NLOPT_GD_STOGO_RAND
        NLOPT_LD_LBFGS_NOCEDAL  # This one exists
        NLOPT_LD_LBFGS
        NLOPT_LN_PRAXIS
        NLOPT_LD_VAR1
        NLOPT_LD_VAR2
        NLOPT_LD_TNEWTON
        NLOPT_LD_TNEWTON_RESTART
        NLOPT_LD_TNEWTON_PRECOND
        NLOPT_LD_TNEWTON_PRECOND_RESTART
        NLOPT_GN_CRS2_LM
        NLOPT_GN_MLSL
        NLOPT_GD_MLSL
        NLOPT_GN_MLSL_LDS
        NLOPT_GD_MLSL_LDS
        NLOPT_LD_MMA
        NLOPT_LN_COBYLA
        NLOPT_LN_NEWUOA
        NLOPT_LN_NEWUOA_BOUND
        NLOPT_LN_NELDERMEAD
        NLOPT_LN_SBPLX
        NLOPT_LN_AUGLAG
        NLOPT_LD_AUGLAG
        NLOPT_LN_AUGLAG_EQ
        NLOPT_LD_AUGLAG_EQ
        NLOPT_LN_BOBYQA
        NLOPT_GN_ISRES
        NLOPT_AUGLAG
        NLOPT_AUGLAG_EQ
        NLOPT_G_MLSL
        NLOPT_G_MLSL_LDS
        NLOPT_LD_SLSQP
        NLOPT_LD_CCSAQ
        NLOPT_GN_ESCH
        NLOPT_GN_AGS
        NLOPT_NUM_ALGORITHMS
    end
end

function nlopt_algorithm_name(a)
    ccall((:nlopt_algorithm_name, libnlopt), Ptr{Cchar}, (nlopt_algorithm,), a)
end

function nlopt_algorithm_to_string(algorithm)
    ccall((:nlopt_algorithm_to_string, libnlopt), Ptr{Cchar}, (nlopt_algorithm,), algorithm)
end

function nlopt_algorithm_from_string(name)
    ccall((:nlopt_algorithm_from_string, libnlopt), nlopt_algorithm, (Ptr{Cchar},), name)
end

@cenum nlopt_result::Int32 begin
    NLOPT_FAILURE = -1
    NLOPT_INVALID_ARGS = -2
    NLOPT_OUT_OF_MEMORY = -3
    NLOPT_ROUNDOFF_LIMITED = -4
    NLOPT_FORCED_STOP = -5
    NLOPT_NUM_FAILURES = -6
    NLOPT_SUCCESS = 1
    NLOPT_STOPVAL_REACHED = 2
    NLOPT_FTOL_REACHED = 3
    NLOPT_XTOL_REACHED = 4
    NLOPT_MAXEVAL_REACHED = 5
    NLOPT_MAXTIME_REACHED = 6
    NLOPT_NUM_RESULTS = 7
end

function nlopt_result_to_string(algorithm)
    ccall((:nlopt_result_to_string, libnlopt), Ptr{Cchar}, (nlopt_result,), algorithm)
end

function nlopt_result_from_string(name)
    ccall((:nlopt_result_from_string, libnlopt), nlopt_result, (Ptr{Cchar},), name)
end

function nlopt_srand(seed)
    ccall((:nlopt_srand, libnlopt), Cvoid, (Culong,), seed)
end

function nlopt_srand_time()
    ccall((:nlopt_srand_time, libnlopt), Cvoid, ())
end

function nlopt_version(major, minor, bugfix)
    ccall((:nlopt_version, libnlopt), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), major, minor, bugfix)
end

mutable struct nlopt_opt_s end

const nlopt_opt = Ptr{Cvoid}

function nlopt_create(algorithm, n)
    ccall((:nlopt_create, libnlopt), nlopt_opt, (nlopt_algorithm, Cuint), algorithm, n)
end

function nlopt_destroy(opt)
    ccall((:nlopt_destroy, libnlopt), Cvoid, (nlopt_opt,), opt)
end

function nlopt_copy(opt)
    ccall((:nlopt_copy, libnlopt), nlopt_opt, (nlopt_opt,), opt)
end

function nlopt_optimize(opt, x, opt_f)
    ccall((:nlopt_optimize, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}, Ptr{Cdouble}), opt, x, opt_f)
end

function nlopt_set_min_objective(opt, f, f_data)
    ccall((:nlopt_set_min_objective, libnlopt), nlopt_result, (nlopt_opt, nlopt_func, Ptr{Cvoid}), opt, f, f_data)
end

function nlopt_set_max_objective(opt, f, f_data)
    ccall((:nlopt_set_max_objective, libnlopt), nlopt_result, (nlopt_opt, nlopt_func, Ptr{Cvoid}), opt, f, f_data)
end

function nlopt_set_precond_min_objective(opt, f, pre, f_data)
    ccall((:nlopt_set_precond_min_objective, libnlopt), nlopt_result, (nlopt_opt, nlopt_func, nlopt_precond, Ptr{Cvoid}), opt, f, pre, f_data)
end

function nlopt_set_precond_max_objective(opt, f, pre, f_data)
    ccall((:nlopt_set_precond_max_objective, libnlopt), nlopt_result, (nlopt_opt, nlopt_func, nlopt_precond, Ptr{Cvoid}), opt, f, pre, f_data)
end

function nlopt_get_algorithm(opt)
    ccall((:nlopt_get_algorithm, libnlopt), nlopt_algorithm, (nlopt_opt,), opt)
end

function nlopt_get_dimension(opt)
    ccall((:nlopt_get_dimension, libnlopt), Cuint, (nlopt_opt,), opt)
end

function nlopt_get_errmsg(opt)
    ccall((:nlopt_get_errmsg, libnlopt), Ptr{Cchar}, (nlopt_opt,), opt)
end

function nlopt_set_param(opt, name, val)
    ccall((:nlopt_set_param, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cchar}, Cdouble), opt, name, val)
end

function nlopt_get_param(opt, name, defaultval)
    ccall((:nlopt_get_param, libnlopt), Cdouble, (nlopt_opt, Ptr{Cchar}, Cdouble), opt, name, defaultval)
end

function nlopt_has_param(opt, name)
    ccall((:nlopt_has_param, libnlopt), Cint, (nlopt_opt, Ptr{Cchar}), opt, name)
end

function nlopt_num_params(opt)
    ccall((:nlopt_num_params, libnlopt), Cuint, (nlopt_opt,), opt)
end

function nlopt_nth_param(opt, n)
    ccall((:nlopt_nth_param, libnlopt), Ptr{Cchar}, (nlopt_opt, Cuint), opt, n)
end

function nlopt_set_lower_bounds(opt, lb)
    ccall((:nlopt_set_lower_bounds, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}), opt, lb)
end

function nlopt_set_lower_bounds1(opt, lb)
    ccall((:nlopt_set_lower_bounds1, libnlopt), nlopt_result, (nlopt_opt, Cdouble), opt, lb)
end

function nlopt_set_lower_bound(opt, i, lb)
    ccall((:nlopt_set_lower_bound, libnlopt), nlopt_result, (nlopt_opt, Cint, Cdouble), opt, i, lb)
end

function nlopt_get_lower_bounds(opt, lb)
    ccall((:nlopt_get_lower_bounds, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}), opt, lb)
end

function nlopt_set_upper_bounds(opt, ub)
    ccall((:nlopt_set_upper_bounds, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}), opt, ub)
end

function nlopt_set_upper_bounds1(opt, ub)
    ccall((:nlopt_set_upper_bounds1, libnlopt), nlopt_result, (nlopt_opt, Cdouble), opt, ub)
end

function nlopt_set_upper_bound(opt, i, ub)
    ccall((:nlopt_set_upper_bound, libnlopt), nlopt_result, (nlopt_opt, Cint, Cdouble), opt, i, ub)
end

function nlopt_get_upper_bounds(opt, ub)
    ccall((:nlopt_get_upper_bounds, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}), opt, ub)
end

function nlopt_remove_inequality_constraints(opt)
    ccall((:nlopt_remove_inequality_constraints, libnlopt), nlopt_result, (nlopt_opt,), opt)
end

function nlopt_add_inequality_constraint(opt, fc, fc_data, tol)
    ccall((:nlopt_add_inequality_constraint, libnlopt), nlopt_result, (nlopt_opt, nlopt_func, Ptr{Cvoid}, Cdouble), opt, fc, fc_data, tol)
end

function nlopt_add_precond_inequality_constraint(opt, fc, pre, fc_data, tol)
    ccall((:nlopt_add_precond_inequality_constraint, libnlopt), nlopt_result, (nlopt_opt, nlopt_func, nlopt_precond, Ptr{Cvoid}, Cdouble), opt, fc, pre, fc_data, tol)
end

function nlopt_add_inequality_mconstraint(opt, m, fc, fc_data, tol)
    ccall((:nlopt_add_inequality_mconstraint, libnlopt), nlopt_result, (nlopt_opt, Cuint, nlopt_mfunc, Ptr{Cvoid}, Ptr{Cdouble}), opt, m, fc, fc_data, tol)
end

function nlopt_remove_equality_constraints(opt)
    ccall((:nlopt_remove_equality_constraints, libnlopt), nlopt_result, (nlopt_opt,), opt)
end

function nlopt_add_equality_constraint(opt, h, h_data, tol)
    ccall((:nlopt_add_equality_constraint, libnlopt), nlopt_result, (nlopt_opt, nlopt_func, Ptr{Cvoid}, Cdouble), opt, h, h_data, tol)
end

function nlopt_add_precond_equality_constraint(opt, h, pre, h_data, tol)
    ccall((:nlopt_add_precond_equality_constraint, libnlopt), nlopt_result, (nlopt_opt, nlopt_func, nlopt_precond, Ptr{Cvoid}, Cdouble), opt, h, pre, h_data, tol)
end

function nlopt_add_equality_mconstraint(opt, m, h, h_data, tol)
    ccall((:nlopt_add_equality_mconstraint, libnlopt), nlopt_result, (nlopt_opt, Cuint, nlopt_mfunc, Ptr{Cvoid}, Ptr{Cdouble}), opt, m, h, h_data, tol)
end

function nlopt_set_stopval(opt, stopval)
    ccall((:nlopt_set_stopval, libnlopt), nlopt_result, (nlopt_opt, Cdouble), opt, stopval)
end

function nlopt_get_stopval(opt)
    ccall((:nlopt_get_stopval, libnlopt), Cdouble, (nlopt_opt,), opt)
end

function nlopt_set_ftol_rel(opt, tol)
    ccall((:nlopt_set_ftol_rel, libnlopt), nlopt_result, (nlopt_opt, Cdouble), opt, tol)
end

function nlopt_get_ftol_rel(opt)
    ccall((:nlopt_get_ftol_rel, libnlopt), Cdouble, (nlopt_opt,), opt)
end

function nlopt_set_ftol_abs(opt, tol)
    ccall((:nlopt_set_ftol_abs, libnlopt), nlopt_result, (nlopt_opt, Cdouble), opt, tol)
end

function nlopt_get_ftol_abs(opt)
    ccall((:nlopt_get_ftol_abs, libnlopt), Cdouble, (nlopt_opt,), opt)
end

function nlopt_set_xtol_rel(opt, tol)
    ccall((:nlopt_set_xtol_rel, libnlopt), nlopt_result, (nlopt_opt, Cdouble), opt, tol)
end

function nlopt_get_xtol_rel(opt)
    ccall((:nlopt_get_xtol_rel, libnlopt), Cdouble, (nlopt_opt,), opt)
end

function nlopt_set_xtol_abs1(opt, tol)
    ccall((:nlopt_set_xtol_abs1, libnlopt), nlopt_result, (nlopt_opt, Cdouble), opt, tol)
end

function nlopt_set_xtol_abs(opt, tol)
    ccall((:nlopt_set_xtol_abs, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}), opt, tol)
end

function nlopt_get_xtol_abs(opt, tol)
    ccall((:nlopt_get_xtol_abs, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}), opt, tol)
end

function nlopt_set_x_weights1(opt, w)
    ccall((:nlopt_set_x_weights1, libnlopt), nlopt_result, (nlopt_opt, Cdouble), opt, w)
end

function nlopt_set_x_weights(opt, w)
    ccall((:nlopt_set_x_weights, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}), opt, w)
end

function nlopt_get_x_weights(opt, w)
    ccall((:nlopt_get_x_weights, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}), opt, w)
end

function nlopt_set_maxeval(opt, maxeval)
    ccall((:nlopt_set_maxeval, libnlopt), nlopt_result, (nlopt_opt, Cint), opt, maxeval)
end

function nlopt_get_maxeval(opt)
    ccall((:nlopt_get_maxeval, libnlopt), Cint, (nlopt_opt,), opt)
end

function nlopt_get_numevals(opt)
    ccall((:nlopt_get_numevals, libnlopt), Cint, (nlopt_opt,), opt)
end

function nlopt_set_maxtime(opt, maxtime)
    ccall((:nlopt_set_maxtime, libnlopt), nlopt_result, (nlopt_opt, Cdouble), opt, maxtime)
end

function nlopt_get_maxtime(opt)
    ccall((:nlopt_get_maxtime, libnlopt), Cdouble, (nlopt_opt,), opt)
end

function nlopt_force_stop(opt)
    ccall((:nlopt_force_stop, libnlopt), nlopt_result, (nlopt_opt,), opt)
end

function nlopt_set_force_stop(opt, val)
    ccall((:nlopt_set_force_stop, libnlopt), nlopt_result, (nlopt_opt, Cint), opt, val)
end

function nlopt_get_force_stop(opt)
    ccall((:nlopt_get_force_stop, libnlopt), Cint, (nlopt_opt,), opt)
end

function nlopt_set_local_optimizer(opt, local_opt)
    ccall((:nlopt_set_local_optimizer, libnlopt), nlopt_result, (nlopt_opt, nlopt_opt), opt, local_opt)
end

function nlopt_set_population(opt, pop)
    ccall((:nlopt_set_population, libnlopt), nlopt_result, (nlopt_opt, Cuint), opt, pop)
end

function nlopt_get_population(opt)
    ccall((:nlopt_get_population, libnlopt), Cuint, (nlopt_opt,), opt)
end

function nlopt_set_vector_storage(opt, dim)
    ccall((:nlopt_set_vector_storage, libnlopt), nlopt_result, (nlopt_opt, Cuint), opt, dim)
end

function nlopt_get_vector_storage(opt)
    ccall((:nlopt_get_vector_storage, libnlopt), Cuint, (nlopt_opt,), opt)
end

function nlopt_set_default_initial_step(opt, x)
    ccall((:nlopt_set_default_initial_step, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}), opt, x)
end

function nlopt_set_initial_step(opt, dx)
    ccall((:nlopt_set_initial_step, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}), opt, dx)
end

function nlopt_set_initial_step1(opt, dx)
    ccall((:nlopt_set_initial_step1, libnlopt), nlopt_result, (nlopt_opt, Cdouble), opt, dx)
end

function nlopt_get_initial_step(opt, x, dx)
    ccall((:nlopt_get_initial_step, libnlopt), nlopt_result, (nlopt_opt, Ptr{Cdouble}, Ptr{Cdouble}), opt, x, dx)
end

# typedef void * ( * nlopt_munge ) ( void * p )
const nlopt_munge = Ptr{Cvoid}

function nlopt_set_munge(opt, munge_on_destroy, munge_on_copy)
    ccall((:nlopt_set_munge, libnlopt), Cvoid, (nlopt_opt, nlopt_munge, nlopt_munge), opt, munge_on_destroy, munge_on_copy)
end

# typedef void * ( * nlopt_munge2 ) ( void * p , void * data )
const nlopt_munge2 = Ptr{Cvoid}

function nlopt_munge_data(opt, munge, data)
    ccall((:nlopt_munge_data, libnlopt), Cvoid, (nlopt_opt, nlopt_munge2, Ptr{Cvoid}), opt, munge, data)
end

# typedef double ( * nlopt_func_old ) ( int n , const double * x , double * gradient , /* NULL if not needed */ void * func_data )
const nlopt_func_old = Ptr{Cvoid}

function nlopt_minimize(algorithm, n, f, f_data, lb, ub, x, minf, minf_max, ftol_rel, ftol_abs, xtol_rel, xtol_abs, maxeval, maxtime)
    ccall((:nlopt_minimize, libnlopt), nlopt_result, (nlopt_algorithm, Cint, nlopt_func_old, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Cint, Cdouble), algorithm, n, f, f_data, lb, ub, x, minf, minf_max, ftol_rel, ftol_abs, xtol_rel, xtol_abs, maxeval, maxtime)
end

function nlopt_minimize_constrained(algorithm, n, f, f_data, m, fc, fc_data, fc_datum_size, lb, ub, x, minf, minf_max, ftol_rel, ftol_abs, xtol_rel, xtol_abs, maxeval, maxtime)
    ccall((:nlopt_minimize_constrained, libnlopt), nlopt_result, (nlopt_algorithm, Cint, nlopt_func_old, Ptr{Cvoid}, Cint, nlopt_func_old, Ptr{Cvoid}, Cptrdiff_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Cint, Cdouble), algorithm, n, f, f_data, m, fc, fc_data, fc_datum_size, lb, ub, x, minf, minf_max, ftol_rel, ftol_abs, xtol_rel, xtol_abs, maxeval, maxtime)
end

function nlopt_minimize_econstrained(algorithm, n, f, f_data, m, fc, fc_data, fc_datum_size, p, h, h_data, h_datum_size, lb, ub, x, minf, minf_max, ftol_rel, ftol_abs, xtol_rel, xtol_abs, htol_rel, htol_abs, maxeval, maxtime)
    ccall((:nlopt_minimize_econstrained, libnlopt), nlopt_result, (nlopt_algorithm, Cint, nlopt_func_old, Ptr{Cvoid}, Cint, nlopt_func_old, Ptr{Cvoid}, Cptrdiff_t, Cint, nlopt_func_old, Ptr{Cvoid}, Cptrdiff_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Cdouble, Cdouble, Cint, Cdouble), algorithm, n, f, f_data, m, fc, fc_data, fc_datum_size, p, h, h_data, h_datum_size, lb, ub, x, minf, minf_max, ftol_rel, ftol_abs, xtol_rel, xtol_abs, htol_rel, htol_abs, maxeval, maxtime)
end

function nlopt_get_local_search_algorithm(deriv, nonderiv, maxeval)
    ccall((:nlopt_get_local_search_algorithm, libnlopt), Cvoid, (Ptr{nlopt_algorithm}, Ptr{nlopt_algorithm}, Ptr{Cint}), deriv, nonderiv, maxeval)
end

function nlopt_set_local_search_algorithm(deriv, nonderiv, maxeval)
    ccall((:nlopt_set_local_search_algorithm, libnlopt), Cvoid, (nlopt_algorithm, nlopt_algorithm, Cint), deriv, nonderiv, maxeval)
end

function nlopt_get_stochastic_population()
    ccall((:nlopt_get_stochastic_population, libnlopt), Cint, ())
end

function nlopt_set_stochastic_population(pop)
    ccall((:nlopt_set_stochastic_population, libnlopt), Cvoid, (Cint,), pop)
end

const NLOPT_MINF_MAX_REACHED = NLOPT_STOPVAL_REACHED

# Skipping MacroDefinition: NLOPT_DEPRECATED __attribute__ ( ( deprecated ) )
