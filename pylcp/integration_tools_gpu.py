import inspect
import jax
import jax.numpy as jnp
import functools
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
        

class RandomOdeResult:
    def __init__(self, t, y, t_random, n_random, inds_random, success, status=0, message="", nfev=0):
        self.t = t
        self.y = y
        self.t_random = t_random
        self.n_random = n_random
        self.inds_random = inds_random
        self.success = success
        self.status = status
        self.message = message
        self.nfev = nfev


@functools.partial(jax.jit, static_argnames=("func", "random_func", "max_steps", "solver_type"))
def _batched_random_trajectories(
    func,
    random_func,
    t0,
    tf,
    y0_batch, 
    keys_batch,
    max_steps,
    max_step_global, 
    rtol,
    atol,
    dt0,
    solver_type="Dopri5"
    ):
    """
    JIT-compiled batched execution for simulating stochastic trajectories.
    
    This function runs a purely functional `jax.lax.while_loop` concurrently 
    for every atom/trajectory in the batch using XLA vectorization (`jax.vmap`).

    Args:
        func (callable): 
            Right-hand side of the system ODE (the continuous physics). 
            The calling signature is ``fun(t, y)``. Here `t` is a scalar, 
            and `y` is a JAX array representing the state of a single atom. 
        random_func (callable): 
            A JAX-compatible function that simulates discrete, stochastic events 
            (such as random photon recoils). Must accept ``(t, y, dt, key)`` 
            and return a tuple: ``(y_jump, n_scatters, dt_max_suggested, key_new)``.
            * `y_jump`: A new copy of the state array containing the updated velocities.
            * `n_scatters`: The number of scattering events that occurred.
            * `dt_max_suggested`: The maximum safe time step for the next loop.
            * `key_new`: A freshly split JAX PRNGKey.
        t0 (float or jnp.float64): 
            The initial time of the integration interval.
        tf (float or jnp.float64): 
            The final time of the integration interval.
        y0_batch (jax.Array): 
            A batch of initial state arrays. Shape must be `(N, state_dim)` 
            where `N` is the number of simultaneous trajectories (atoms) to simulate.
        keys_batch (jax.Array): 
            A batch of JAX PRNG keys, one for each trajectory. Shape must be `(N, ...)`.
        max_steps (int): 
            The maximum number of integration steps. In JAX, `while_loop` arrays 
            must have a statically known size, so arrays are pre-allocated to this length.
        max_step_global (float): 
            The absolute maximum continuous time step (`dt`) the solver is allowed to take.
        rtol (float): 
            Relative tolerance for the ODE solver's PID step-size controller.
        atol (float): 
            Absolute tolerance for the ODE solver's PID step-size controller.
        dt0 (float): 
            The initial time step size to use at `t0`.
        solver_type (str, optional): 
            The Diffrax solver to use ("Dopri5", "Bosh3", or "Kvaerno5"). Defaults to "Dopri5".

    Raises:
        ValueError: 
            If an unrecognized `solver_type` is provided.

    Returns:
        dict: 
            A dictionary containing the batched, zero-padded JAX arrays for the entire 
            integration run across all `N` trajectories. Keys include:
            - `t` (float): Final time reached.
            - `y` (jax.Array): Final state reached.
            - `step_idx` (int): The actual number of steps taken before terminating.
            - `ts` (jax.Array): The time history array of shape `(N, max_steps)`.
            - `ys` (jax.Array): The state history array of shape `(N, max_steps, state_dim)`.
            - `t_random` (jax.Array): The times stochastic jumps occurred.
            - `n_random` (jax.Array): The number of jumps at each corresponding time.
            - `nfev` (int): Number of function evaluations performed by the solver.
    """
    if solver_type == "Dopri5":
        solver = Dopri5()
    elif solver_type == "Bosh3":
        solver = Bosh3()
    elif solver_type == "Kvaerno5":
        solver = Kvaerno5()
    else:
        raise ValueError(f"Solver {solver} is not one of the specified solvers implemented in the function!")
    
    term = ODETerm(lambda t, y, args: func(t, y))
    
    def cond_fun(state):
        return (state['t'] < tf) & (state['step_idx'] < max_steps)
    
    def body_fun(state):
        t_curr = state['t']
        dt_curr = state['dt']
        t_next = jnp.minimum(t_curr + dt_curr, tf)
        actual_dt = t_next - t_curr
        
        sol = diffeqsolve(
            term,
            solver,
            t0=t_curr,
            t1=t_next,
            dt0=dt_curr,
            y0=state['y'],
            stepsize_controller=PIDController(rtol=rtol, atol=atol),
            saveat=SaveAt(t1=True)
        )
        
        y_next = sol.ys[-1]
        nfev_add = sol.stats['num_steps']
        
        y_jump, n_scatters, dt_max_suggested, key_new = random_func(
            t_next, y_next, actual_dt, state['key']
        )
        
        dt_next = jnp.minimum(dt_max_suggested, max_step_global)
        idx = state['step_idx']
        
        ts_new = state['ts'].at[idx].set(t_next)
        ys_new = state['ys'].at[idx].set(y_jump)
        t_rand_new = jnp.where(n_scatters > 0, state['t_random'].at[idx].set(t_next), state['t_random'])
        n_rand_new = jnp.where(n_scatters > 0, state['n_random'].at[idx].set(n_scatters), state['n_random'])
        
        return {
            't': t_next, 'y': y_jump, 'dt': dt_next, 'key': key_new,
            'step_idx': idx + 1, 'ts': ts_new, 'ys': ys_new,
            't_random': t_rand_new, 'n_random': n_rand_new,
            'nfev': state['nfev'] + nfev_add
        }
    
    def single_trajectory(y0, key):
        initial_state = {
            't': jnp.float64(t0),
            'y': y0,
            'dt': dt0,
            'key': key,
            'step_idx': jnp.int32(1),
            'ts': jnp.zeros(max_steps, dtype=jnp.float64).at[0].set(t0),
            'ys': jnp.zeros((max_steps,) + y0.shape, dtype=y0.dtype).at[0].set(y0),
            't_random': jnp.zeros(max_steps, dtype=jnp.float64),
            'n_random': jnp.zeros(max_steps, dtype=jnp.int32),
            'nfev': jnp.int32(0)
        }
        
        return jax.lax.while_loop(cond_fun, body_fun, initial_state)
    
    return jax.vmap(single_trajectory)(y0_batch, keys_batch)

def solve_ivp_random(
    fun,
    random_func,
    t_span,
    y0_batch,
    keys_batch,
    solver_type="Dopri5",
    max_steps=100000,
    max_step=jnp.inf,
    rtol=1e-5,
    atol=1e-5,
    **options
    ):
    """
    Solve an initial value problem for a system of ODEs natively in JAX with stochastic jumps.
    
    This function is the parallel, GPU-accelerated replacement for standard 
    `scipy.integrate.solve_ivp`. It takes a batch of initial conditions and 
    simulates thousands of trajectories concurrently using Diffrax.

    Args:
        fun (callable): 
            Right-hand side of the continuous system ODE. 
            Calling signature: ``fun(t, y)``.
        random_func (callable): 
            A JAX-compatible stochastic event detector and applier. 
            Calling signature: ``(y_jump, n_scatters, dt_max, key_new) = random_func(t, y, dt, key)``.
        t_span (tuple of floats): 
            Interval of integration `(t0, tf)`. The solver starts with `t=t0` 
            and integrates until it reaches `t=tf`.
        y0_batch (jax.Array): 
            A batch of initial states. Shape must be `(N, state_dim)`, where `N` 
            is the number of atoms/trajectories to simulate concurrently.
        keys_batch (jax.Array): 
            A batch of JAX PRNGKeys for evaluating Poisson scattering variables. 
            Must be mapped to the `N` trajectories.
        solver_type (str, optional): 
            Integration method mapped to Diffrax. Options include "Dopri5" (RK45), 
            "Bosh3" (RK23), or "Kvaerno5" (Radau/BDF). Defaults to "Dopri5".
        max_steps (int, optional): 
            Maximum buffer size for preallocating JAX arrays. If an atom's 
            integration requires more steps, it will safely terminate early. 
            Defaults to 100,000.
        max_step (float, optional): 
            Maximum global allowed step size. Defaults to `jnp.inf`.
        rtol (float, optional): 
            Relative tolerance for the step size controller. Defaults to 1e-5.
        atol (float, optional): 
            Absolute tolerance for the step size controller. Defaults to 1e-5.
        **options: 
            Additional keyword arguments to maintain API compatibility with SciPy 
            (e.g., standard `method` overrides).

    Returns:
        list of RandomOdeResult: 
            A list containing `N` result objects (one for each atom in the batch). 
            Each `RandomOdeResult` mimics standard SciPy behavior with padded 
            zeroes cleanly trimmed off. Key attributes of each result include:
            - `t` (ndarray): Time points.
            - `y` (ndarray): Values of the solution at `t`.
            - `t_random` (ndarray): Times at which stochastic scattering jumps occurred.
            - `inds_random` (ndarray): Boolean mask mapping jump events to the `t` and `y` grids.
            - `success` (bool): True if the solver reached `tf`.
            - `status` (int): 0 if successful, -1 if terminated early.
    """
    
    # cast to jnp just in case
    y0_batch = jnp.asarray(y0_batch)
    keys_batch = jnp.asarray(keys_batch)
    t0, tf = jnp.asarray(t_span[0], dtype=jnp.float64), jnp.asarray(t_span[1], dtype=jnp.float64)
    dt0 = jnp.minimum(1e-3, max_step)
    N = y0_batch.shape[0]
    
    batched_state = _batched_random_trajectories(
        fun,
        random_func,
        t0,
        tf,
        y0_batch,
        keys_batch,
        max_steps,
        max_step,
        rtol,
        atol,
        dt0,
        solver_type
    )
    
    results = []
    for i in range(N):
        num_steps = int(batched_state['step_idx'][i])
        
        ts_final = batched_state['ts'][i, :num_steps]
        ys_final = jnp.moveaxis(batched_state['ys'][i, :num_steps], 0, -1)
        
        t_rand_raw = batched_state['t_random'][i, :num_steps]
        n_rand_raw = batched_state['n_random'][i, :num_steps]
        
        scatter_mask = n_rand_raw > 0
        t_random_clean = t_rand_raw[scatter_mask]
        n_random_clean = n_rand_raw[scatter_mask]
        inds_random = scatter_mask
        
        success = bool(batched_state['t'][i] >= tf)
        status = 0 if success else -1
        message = "Success" if success else "Terminated early (max_steps limit)."
        
        results.append(RandomOdeResult(
            t=ts_final,
            y=ys_final,
            t_random=t_random_clean,
            n_random=n_random_clean,
            inds_random=inds_random,
            success=success,
            status=status,
            message=message,
            nfev=int(batched_state['nfev'][i])
        ))

    return results


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp

    def dydt(t, y):
        return jnp.array([-y[1], y[0]])

    def func2(t, y, dt, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        roll = jax.random.uniform(subkey1)
        did_scatter = roll < 2 * dt
        kick = 5 * jax.random.normal(subkey2)
        y_jump = jnp.where(did_scatter, y.at[1].add(kick), y)
        n_events = jnp.where(did_scatter, 1, 0)
        max_dt = jnp.maximum(0.1, jnp.abs(y_jump[1]))
        return y_jump, n_events, max_dt, key

    # initial conditions
    y0_batch = jnp.array([
        [0., 1.],  # Atom 1
        [0., 1.],  # Atom 2
        [0., 1.]   # Atom 3
    ])
    
    # 3 unique keys, one for each atom
    initial_key = jax.random.PRNGKey(42)
    keys_batch = jax.random.split(initial_key, 3)

    # Runs the batched solver on 3 atoms
    sols = solve_ivp_random(dydt, func2, [0., 10 * jnp.pi], y0_batch, keys_batch, max_step=0.1, method='Dopri5')

    # only plot one atom instead of all 3
    sol = sols[0] 
    
    plt.figure()
    plt.plot(sol.t, sol.y.T)
    
    if len(sol.t_random) > 0:
        plt.plot(sol.t_random, sol.y[:, sol.inds_random].T, 'o', color='black', label='Random Kicks')
        plt.legend()
        
    plt.xlabel('Time')
    plt.ylabel('State (Position & Velocity)')
    plt.title('Batched GPU Integration (Atom 1)')
    plt.show()