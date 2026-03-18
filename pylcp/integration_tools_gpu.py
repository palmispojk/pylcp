import inspect

import jax
jax.config.update("jax_enable_x64", True)
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


def _ensure_3arg(func):
    """Wrap ``func`` so it matches the ``func(t, y, args)`` signature that
    diffrax's ``ODETerm`` requires.

    Functions with fewer than three *required* (positional, no-default)
    parameters are wrapped so that the ``args`` value from diffrax is
    ignored and the original function is called as ``func(t, y)``.  This
    lets callers pass either the simple ``(t, y)`` form or the full
    ``(t, y, args)`` form — the latter is needed when static physics
    data is supplied via the ``args`` parameter of :func:`solve_ivp_dense`
    or :func:`solve_ivp_random`.

    A function like ``func(t, y, _H=default)`` counts as 2-arg (the third
    parameter has a default) and is wrapped, so diffrax's ``args=None``
    does not clobber the default.
    """
    sig = inspect.signature(func)
    n_required = sum(
        1 for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    if n_required < 3:
        return lambda t, y, args: func(t, y)
    return func


class RandomOdeResult:
    """Result of a single ODE trajectory.

    Can be constructed eagerly (all arrays passed in) or lazily from a
    batched state dict returned by ``_batched_random_trajectories``.  In
    the lazy case, per-atom slicing and device-to-host transfer are
    deferred until an attribute is first accessed, avoiding N separate
    GPU→host copies during result construction.
    """

    def __init__(self, t=None, y=None, t_random=None, n_random=None,
                 inds_random=None, success=None, status=0, message="",
                 nfev=0, _batched_state=None, _index=None, _tf=None):
        if _batched_state is not None:
            # Lazy mode: defer slicing until attribute access.
            self._batched_state = _batched_state
            self._index = _index
            self._tf = _tf
            self._cache = {}
        else:
            # Eager mode: store pre-sliced arrays directly.
            self._batched_state = None
            self.t = t
            self.y = y
            self.t_random = t_random
            self.n_random = n_random
            self.inds_random = inds_random
            self.success = success
            self.status = status
            self.message = message
            self.nfev = nfev

    def _materialise(self):
        """Slice this atom's data from the batched state (once)."""
        if 't' in self._cache:
            return
        i = self._index
        bs = self._batched_state
        num_steps = int(bs['step_idx'][i])

        self._cache['t'] = bs['ts'][i, :num_steps]
        self._cache['y'] = jnp.moveaxis(bs['ys'][i, :num_steps], 0, -1)

        t_rand_raw = bs['t_random'][i, :num_steps]
        n_rand_raw = bs['n_random'][i, :num_steps]
        scatter_mask = n_rand_raw > 0

        self._cache['t_random'] = t_rand_raw[scatter_mask]
        self._cache['n_random'] = n_rand_raw[scatter_mask]
        self._cache['inds_random'] = scatter_mask
        self._cache['success'] = bool(bs['t'][i] >= self._tf)
        self._cache['status'] = 0 if self._cache['success'] else -1
        self._cache['message'] = ("Success" if self._cache['success']
                                  else "Terminated early (max_steps limit).")
        self._cache['nfev'] = int(bs['nfev'][i])

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails (lazy mode).
        if name.startswith('_'):
            raise AttributeError(name)
        self._materialise()
        try:
            val = self._cache[name]
        except KeyError:
            raise AttributeError(f"RandomOdeResult has no attribute {name!r}")
        # Promote to a real attribute so subsequent accesses skip __getattr__.
        setattr(self, name, val)
        return val


@functools.partial(jax.jit, static_argnames=("func", "random_func", "max_steps", "inner_max_steps", "solver_type"))
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
    solver_type="Dopri5",
    inner_max_steps=64,
    ):
    """
    JIT-compiled batched execution for simulating stochastic trajectories.
    
    This function runs a purely functional `jax.lax.while_loop` concurrently 
    for every atom/trajectory in the batch using XLA vectorization (`jax.vmap`).

    Args:
        func (callable):
            Right-hand side of the system ODE (the continuous physics).
            Accepts either ``func(t, y)`` or ``func(t, y, args)``.  The
            3-arg form receives the ``args`` pytree passed to the solver;
            the 2-arg form is wrapped automatically.
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
        inner_max_steps (int, optional):
            Maximum number of adaptive steps the inner diffrax solver is
            allowed to take per outer time step.  The outer ``while_loop``
            already controls time progression; the inner solver only needs
            enough steps to integrate one ``dt`` interval.  Defaults to 64.

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
        raise ValueError(f"Solver '{solver_type}' is not one of the specified solvers. Use 'Dopri5', 'Bosh3', or 'Kvaerno5'.")
    
    term = ODETerm(_ensure_3arg(func))
    
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
            saveat=SaveAt(t1=True),
            max_steps=inner_max_steps,
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
        # Always-write pattern: single scatter update per array instead of
        # materialising two full-array copies via jnp.where(cond, arr1, arr2).
        t_rand_new = state['t_random'].at[idx].set(
            jnp.where(n_scatters > 0, t_next, jnp.float64(0.)))
        n_rand_new = state['n_random'].at[idx].set(jnp.int32(n_scatters))
        
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

def optimal_batch_size(state_dim, max_steps, inner_max_steps=64, safety=0.6):
    """Return the largest batch size that fits in available GPU memory.

    Estimates peak per-atom allocation inside ``_batched_random_trajectories``
    and divides the currently free device memory (with a safety margin) by that
    figure.  Call this *after* JIT warm-up so that JAX's internal pool is
    already reflected in ``bytes_in_use``.

    Args:
        state_dim (int): Length of the per-atom state vector (``y0.shape[-1]``).
        max_steps (int): Outer buffer size passed to the solver.
        inner_max_steps (int, optional): Inner diffrax buffer size. Defaults to 64.
        safety (float, optional): Fraction of free memory to use. Defaults to 0.6
            to leave headroom for XLA workspace and compilation buffers.

    Returns:
        int: Recommended batch size, or ``None`` if no GPU is present.
    """
    gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpu_devices:
        return None
    stats = gpu_devices[0].memory_stats()
    free_bytes = stats['bytes_limit'] - stats['bytes_in_use']
    # Each atom pre-allocates:
    #   outer ts:  (max_steps,)           float64 → 8 bytes each
    #   outer ys:  (max_steps, state_dim) float64
    #   inner ys:  (inner_max_steps, state_dim) float64 (per diffrax call)
    bytes_per_atom = 8 * (max_steps * (1 + state_dim) + inner_max_steps * state_dim)
    return max(1, int(free_bytes * safety / bytes_per_atom))


def solve_ivp_random(
    fun,
    random_func,
    t_span,
    y0_batch,
    keys_batch,
    solver_type="Dopri5",
    max_steps=100000,
    inner_max_steps=None,
    max_step=float('inf'),
    rtol=1e-5,
    atol=1e-6,
    batch_size=None,
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
        inner_max_steps (int, optional):
            Maximum adaptive steps the inner diffrax solver may take per
            outer time step.  The outer loop already controls time
            progression; the inner solver only integrates one ``dt``
            interval, so a small value (default 64) is sufficient and
            avoids massive over-allocation on GPU.
        batch_size (int, optional):
            Maximum number of atoms to integrate simultaneously.  When
            set, atoms are processed in chunks of this size to limit peak
            GPU memory.  Each chunk reuses the same compiled kernel so
            there is no extra JIT cost.  Default: ``None`` (all atoms in
            one batch).
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
    dt0 = jnp.minimum(jnp.asarray((tf - t0) * 1e-3, dtype=jnp.float64),
                       jnp.asarray(max_step, dtype=jnp.float64))
    N = y0_batch.shape[0]

    if inner_max_steps is None:
        inner_max_steps = max_steps

    if batch_size is None:
        state_dim = y0_batch.shape[-1]
        batch_size = optimal_batch_size(state_dim, max_steps, inner_max_steps) or N

    if batch_size >= N:
        # Single batch — all atoms at once.
        batched_state = _batched_random_trajectories(
            fun, random_func, t0, tf,
            y0_batch, keys_batch,
            max_steps, max_step, rtol, atol, dt0,
            solver_type, inner_max_steps,
        )
        return [
            RandomOdeResult(_batched_state=batched_state, _index=i, _tf=tf)
            for i in range(N)
        ]

    # Chunked execution to limit peak memory.
    results = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk_state = _batched_random_trajectories(
            fun, random_func, t0, tf,
            y0_batch[start:end], keys_batch[start:end],
            max_steps, max_step, rtol, atol, dt0,
            solver_type, inner_max_steps,
        )
        for i in range(end - start):
            results.append(
                RandomOdeResult(_batched_state=chunk_state, _index=i, _tf=tf)
            )
    return results


@functools.partial(jax.jit, static_argnames=('func', 'n_points', 'max_steps', 'rtol', 'atol', 'solver_type'))
def _batched_dense_trajectories(func, t0, t1, y0_batch, n_points, max_steps=4096, rtol=1e-5, atol=1e-6, solver_type='Dopri5', args=None):
    """
    JIT-compiled batched ODE solve returning solution on a fixed time grid.

    This is the performance-critical inner function used by solve_ivp_dense.
    ``func`` and ``n_points`` are declared static so JAX traces and compiles
    once per unique (func, n_points, rtol, atol, solver_type) combination and
    then reuses the compiled XLA kernel for all subsequent calls.

    ``func`` must have the calling signature ``func(t, y, args) -> dy/dt``.
    Pass physics data via ``args`` (a JAX pytree) to enable sharing a single
    compiled kernel across multiple OBE instances that have the same Hamiltonian
    structure (same matrix shapes).  When ``func`` is a stable module-level
    function object, JAX's JIT cache key is the same for every caller,
    so compilation happens once per (n_points, args-pytree-shape) pair.

    Args:
        func: RHS of the ODE.  Accepts either ``func(t, y)`` or
              ``func(t, y, args) -> dy/dt``.  The 3-arg form receives
              the ``args`` pytree; the 2-arg form is wrapped automatically.
              Must be a stable Python object (e.g. a module-level function)
              so the JIT cache key stays constant across calls.
        t0, t1: Start and end time (JAX float64 scalars).
        y0_batch: Initial conditions, shape (N, state_dim).
        n_points: Number of equally-spaced output time points (static).
        max_steps: Maximum number of adaptive solver steps (static). Default: 4096.
        rtol, atol: Adaptive step-size controller tolerances (static).
        solver_type: 'Dopri5', 'Bosh3', or 'Kvaerno5' (static).
        args: JAX pytree passed as the third argument to func at every step.
              Traced dynamically — different values with the same pytree
              structure reuse the compiled kernel without recompilation.
              Default: None.

    Returns:
        ys: shape (N, n_points, state_dim)
        ts: shape (n_points,)
    """
    if solver_type == 'Dopri5':
        solver = Dopri5()
    elif solver_type == 'Bosh3':
        solver = Bosh3()
    elif solver_type == 'Kvaerno5':
        solver = Kvaerno5()
    else:
        raise ValueError(f"Solver '{solver_type}' not recognised. "
                         f"Use 'Dopri5', 'Bosh3', or 'Kvaerno5'.")

    term = ODETerm(_ensure_3arg(func))

    # t0/t1 can be scalars (shared) or per-atom arrays (shape (N,)).
    # When per-atom, each atom gets its own time grid and integration
    # window — all run in parallel via vmap with a single JIT trace.
    per_atom = (jnp.ndim(t0) > 0 or jnp.ndim(t1) > 0)

    if not per_atom:
        # Shared t_span: single ts_grid, vmap over y0 only
        ts_grid = jnp.linspace(t0, t1, n_points)

        def solve_one(y0):
            sol = diffeqsolve(
                term,
                solver,
                t0=t0,
                t1=t1,
                dt0=jnp.asarray((t1 - t0) / n_points, dtype=t0.dtype),
                y0=y0,
                args=args,
                stepsize_controller=PIDController(rtol=rtol, atol=atol),
                saveat=SaveAt(ts=ts_grid),
                max_steps=max_steps,
            )
            return sol.ys, ts_grid

        ys, ts_batch = jax.vmap(solve_one)(y0_batch)
        return ys, ts_batch[0]  # ts identical for all atoms
    else:
        # Per-atom t0/t1: each atom has its own time grid
        # Broadcast scalars to match batch dimension
        if jnp.ndim(t0) == 0:
            t0 = jnp.broadcast_to(t0, t1.shape)
        if jnp.ndim(t1) == 0:
            t1 = jnp.broadcast_to(t1, t0.shape)

        def solve_one(y0, t0_i, t1_i):
            ts_grid_i = jnp.linspace(t0_i, t1_i, n_points)
            sol = diffeqsolve(
                term,
                solver,
                t0=t0_i,
                t1=t1_i,
                dt0=jnp.asarray((t1_i - t0_i) / n_points, dtype=t0_i.dtype),
                y0=y0,
                args=args,
                stepsize_controller=PIDController(rtol=rtol, atol=atol),
                saveat=SaveAt(ts=ts_grid_i),
                max_steps=max_steps,
            )
            return sol.ys, ts_grid_i

        ys, ts_batch = jax.vmap(solve_one)(y0_batch, t0, t1)
        return ys, ts_batch  # (N, n_points)


def solve_ivp_dense(func, t_span, y0_batch, n_points=1001,
                    max_steps=4096, rtol=1e-5, atol=1e-6,
                    solver_type='Dopri5', args=None):
    """
    Solve a batched ODE and return the solution on a fixed time grid.

    GPU-native, JIT-compiled replacement for scipy's solve_ivp when you need
    dense output at N equally-spaced time points.  The underlying kernel is
    compiled once per unique (func, n_points, rtol, atol, solver_type)
    combination; subsequent calls with different t_span or y0_batch reuse the
    compiled kernel directly.

    For best performance, pass a module-level function as ``func`` and supply
    all physics data via ``args`` (a JAX pytree).  A module-level function has
    a stable Python identity so the JIT cache key is shared across all callers
    with the same Hamiltonian structure, reducing total compilation time.

    Args:
        func: RHS of the ODE.  Accepts either ``func(t, y)`` or
              ``func(t, y, args) -> dy/dt``.  The 3-arg form receives
              the ``args`` pytree; the 2-arg form is wrapped automatically.
        t_span: (t0, t1) integration interval.  t1 may be a scalar
                (shared endpoint for all atoms) or an array of shape (N,)
                giving a per-atom endpoint.  When per-atom, the returned
                ``ts`` has shape (N, n_points) instead of (n_points,).
        y0_batch: Initial conditions, shape (N, state_dim).
        n_points: Number of equally-spaced output time points. Default 1001.
        max_steps: Maximum number of adaptive solver steps. Default: 4096.
        rtol: Relative tolerance. Default 1e-5.
        atol: Absolute tolerance. Default 1e-5.
        solver_type: 'Dopri5' (default), 'Bosh3', or 'Kvaerno5'.
        args: JAX pytree passed through to func as its third argument.
              Default: None.

    Returns:
        ts: jax.Array, shape (n_points,) or (N, n_points) for per-atom t1
        ys: jax.Array, shape (N, n_points, state_dim)
    """
    t0 = jnp.asarray(t_span[0], dtype=jnp.float64)
    t1 = jnp.asarray(t_span[1], dtype=jnp.float64)
    y0_batch = jnp.asarray(y0_batch)
    ys, ts = _batched_dense_trajectories(
        func, t0, t1, y0_batch, n_points, max_steps, rtol, atol, solver_type, args
    )
    return ts, ys


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
    sols = solve_ivp_random(dydt, func2, [0., 10 * jnp.pi], y0_batch, keys_batch, max_step=0.1, solver_type='Dopri5')

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