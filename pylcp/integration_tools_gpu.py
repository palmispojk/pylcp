import inspect
import jax
import jax.numpy as jnp
from diffrax import (
    diffeqsolve,
    ODETerm, 
    Dopri5, 
    Bosh3, 
    Kvaerno5, 
    SaveAt, 
    PIDController,
    LinearInterpolation
)

class parallelIntegrator(object):
    """
    parallelIntegrator: A GPU-native, JAX-compatible class to integrate a 
    function as it is being called.

    This class replaces the SciPy-based parallelIntegrator. It uses the Diffrax 
    library to solve ODEs natively on the GPU (or CPU via XLA). Unlike the 
    original version, this class is strictly stateless to remain compatible 
    with JAX's JIT and vmap transformations.

    Parameters
    ----------
    func : callable
        The function that is to be integrated. It can have the form func(t) 
        or func(t, y).
    y0 : float or array_like, optional
        The initial value of y. Default value is 0.
    method : string, optional
        Integration method to use, mapped to Diffrax GPU solvers:
            * 'Dopri' (default): Dopri5 (Dormand-Prince 5th order).
            * 'Bosh3': Bosh3 (Bogacki-Shampine 3rd order).
            * 'Kvaerno5': Kvaerno5 (Implicit solver for 
              stiff equations).
    tmax : float, optional
        Maximum magnitude of the time. Included for API compatibility with 
        legacy pylcp code.
    rtol : float, optional
        Relative tolerance for the adaptive step size controller. Default 1e-5.
    atol : float, optional
        Absolute tolerance for the adaptive step size controller. Default 1e-5.

    Attributes
    ----------
    y0 : jax.Array
        The initial state of the integration.
    term : diffrax.ODETerm
        The wrapped differential equation term for Diffrax.
    solver : diffrax.AbstractSolver
        The Diffrax solver instance used for integration.
    """
    
    def __init__(self, func, y0=[0.], method="Dopri5", tmax=1e9, rtol=1e-5, atol=1e-5, **kwargs):
        sig = str(inspect.signature(func))
        if '(t, y)' in sig or '(t, y' in sig:
            self.func = lambda t, y, args: func(t, y)
        elif '(t)' in sig or '(t' in sig:
            self.func = lambda t, y, args: func(t)
        else:
            raise ValueError(f"signature {sig} for func not recognized. Must be (t) or (t, y).")
        
        self.y0 = jnp.array(y0)
        self.tmax = tmax
        self.rtol = rtol
        self.atol = atol
        
        if method == "Dopri5":
            self.solver = Dopri5()
        elif method == 'Bosh3':
            self.solver = Bosh3()
        elif method == "Kvaerno5":
            self.solver = Kvaerno5() 
        else:
            raise ValueError(f"Method {method} not recognized.")
        
        self.term = ODETerm(self.func)
        
        self._batched_solve = jax.vmap(self._solve_single, in_axes=(0,))
        
        
    def _solve_single(self, t):
        """
        Internal stateless solver for a single point in time.

        Parameters
        ----------
        t : float or jax.Array
            The target time for the integration.

        Returns
        -------
        y : jax.Array
            The result of the integration from 0 to t.
        """
        is_zero = jnp.isclose(t, 0.0)
        
        def do_solve(t_val):
            sol = diffeqsolve(
                self.term,
                self.solver,
                t0=0.0,
                t1=t_val,
                dt0=1e-3, # Initial step size guess
                y0=self.y0,
                stepsize_controller=PIDController(rtol=self.rtol, atol=self.atol),
                saveat=SaveAt(t1=True) # Only save the final value at t
            )
            return sol.ys[-1]
    
        def return_y0(t_val):
            return self.y0
        
        return jax.lax.cond(is_zero, return_y0, do_solve, t)

    def __call__(self, t):
        """
        Return the value of the integral at time t.

        If t is an array, the integration is automatically vectorized across 
        the GPU using jax.vmap, solving all time points in parallel.

        Parameters
        ----------
        t : float or jax.Array
            Time or array of times at which to evaluate the function.

        Returns
        -------
        y : jax.Array
            Value of the function at time t.
        """
        t_arr = jnp.asarray(t)
        
        if t_arr.ndim > 0:
            return self._batched_solve(t_arr)
        else:
            return self._solve_single(t_arr)
        
    
    def dense_output(self, t_span, n_points=1000):
        """
        Computes a continuous solution over a specified interval.

        This method pre-integrates the function over a grid and returns a 
        JAX-native interpolation function. This is significantly more 
        efficient when the integrated value must be called repeatedly 
        inside another ODE solver.

        Parameters
        ----------
        t_span : 2-tuple of floats
            The start and end times for the pre-computation.
        n_points : int, optional
            Number of points used for the interpolation grid. Default 1000.

        Returns
        -------
        interp_func : callable
            A function that takes time t and returns the interpolated 
            integral value using jax.linear_interpolation.
        """
        ts = jnp.linspace(t_span[0], t_span[1], n_points)
        
        sol = diffeqsolve(
            self.term,
            self.solver,
            t0=t_span[0],
            t1=t_span[1],
            dt0=1e-3,
            y0=self.y0,
            stepsize_controller=PIDController(rtol=self.rtol, atol=self.atol),
            saveat=SaveAt(ts=ts)
        )
        
        # Return a pure JAX interpolation function that can be called anywhere
        interp = LinearInterpolation(ts, sol.ys)
        return lambda t: interp.evaluate(t)
        
        