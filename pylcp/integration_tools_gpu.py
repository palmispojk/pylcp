import inspect
import os
import logging
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import functools
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
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

_log = logging.getLogger(__name__)

# Enable verbose GPU diagnostics with PYLCP_GPU_DEBUG=1
_GPU_DEBUG = os.environ.get('PYLCP_GPU_DEBUG', '0') == '1'


def _gpu_devices():
    """Return all available GPU devices on this node."""
    return [d for d in jax.devices() if d.platform == 'gpu']


def _gpu_device_info(gpu_devs=None):
    """Return a list of per-GPU info dicts.

    Each dict contains:
        device, device_kind, process_index,
        bytes_limit, bytes_in_use, peak_bytes_in_use, bytes_free
    """
    if gpu_devs is None:
        gpu_devs = _gpu_devices()
    info = []
    for d in gpu_devs:
        entry = {
            'device': d,
            'device_kind': getattr(d, 'device_kind', 'unknown'),
            'process_index': getattr(d, 'process_index', 0),
        }
        stats = d.memory_stats()
        if stats is not None:
            entry['bytes_limit'] = stats['bytes_limit']
            entry['bytes_in_use'] = stats['bytes_in_use']
            entry['peak_bytes_in_use'] = stats['peak_bytes_in_use']
            entry['bytes_free'] = stats['bytes_limit'] - stats['bytes_in_use']
        else:
            entry['bytes_limit'] = 0
            entry['bytes_in_use'] = 0
            entry['peak_bytes_in_use'] = 0
            entry['bytes_free'] = 0
        info.append(entry)
    return info


def _log_gpu_debug(gpu_devs=None, label=""):
    """Print detailed GPU device info when PYLCP_GPU_DEBUG=1."""
    if not _GPU_DEBUG:
        return
    infos = _gpu_device_info(gpu_devs)
    prefix = f"[GPU DEBUG{': ' + label if label else ''}]"
    _log.info(f"{prefix} {len(infos)} GPU(s) detected")
    for i, info in enumerate(infos):
        d = info['device']
        _log.info(
            f"{prefix}   GPU {i}: {d}"
            f"  kind={info['device_kind']}"
            f"  process={info['process_index']}"
            f"  pool={info['bytes_limit']/2**30:.2f} GiB"
            f"  in_use={info['bytes_in_use']/2**20:.1f} MiB"
            f"  peak={info['peak_bytes_in_use']/2**20:.1f} MiB"
            f"  free={info['bytes_free']/2**20:.1f} MiB"
        )


def _shard_batch(arr, gpu_devs):
    """Shard ``arr`` along axis 0 across ``gpu_devs``.

    Pads the batch dimension to be evenly divisible by ``len(gpu_devs)``
    if necessary.  Returns ``(sharded_arr, original_N)``.
    """
    n_gpus = len(gpu_devs)
    if n_gpus <= 1:
        return arr, arr.shape[0]

    N = arr.shape[0]
    remainder = N % n_gpus
    pad_size = 0
    if remainder != 0:
        pad_size = n_gpus - remainder
        pad_shape = (pad_size,) + arr.shape[1:]
        arr = jnp.concatenate([arr, jnp.zeros(pad_shape, dtype=arr.dtype)])

    if _GPU_DEBUG:
        per_gpu = arr.shape[0] // n_gpus
        _log.info(
            f"[GPU DEBUG: _shard_batch] N={N}, n_gpus={n_gpus}, "
            f"padded={pad_size}, per_gpu={per_gpu}, "
            f"arr_shape={arr.shape}"
        )

    mesh = Mesh(gpu_devs, axis_names=('batch',))
    sharding = NamedSharding(mesh, P('batch'))
    return jax.device_put(arr, sharding), N


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
        n_saved = int(bs['save_idx'][i])

        # Initial state is already in slot 0 of ts/ys (written by the
        # while_loop init), so just slice up to n_saved.
        self._cache['t'] = bs['ts'][i, :n_saved]
        self._cache['y'] = np.moveaxis(bs['ys'][i, :n_saved], 0, -1)

        t_rand_raw = bs['t_random'][i, :n_saved]
        n_rand_raw = bs['n_random'][i, :n_saved]
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
    args=None,
    save_every=1,
    progress=False,
    ):
    """
    Batched stochastic trajectory solver with host-side output accumulation.

    Runs groups of ``save_every`` ODE steps on GPU via a JIT-compiled
    ``jax.lax.while_loop`` vmapped across atoms.  After each group the
    snapshot (one point per atom) is transferred to CPU and the next
    group is dispatched.  **No output history arrays are ever allocated
    on GPU** — only the small per-atom carry (y, t, dt, key, counters)
    and diffrax scratch buffers reside on device.  Peak GPU memory is
    therefore independent of ``max_steps`` and ``save_every``.

    The JIT kernel compiles once on the first group call and is reused
    for all subsequent groups.  Host-loop dispatch overhead is ~0.5 ms
    per group, typically <1 %% of total wall time.

    Args:
        progress (bool): Print progress every ~5 %%.  Default: False.

    Returns:
        dict of **CPU numpy arrays** (already transferred from GPU):
            ``t``, ``y``, ``step_idx``, ``save_idx``,
            ``ts``, ``ys``, ``t_random``, ``n_random``, ``nfev``.
    """
    if solver_type == "Dopri5":
        solver = Dopri5()
    elif solver_type == "Bosh3":
        solver = Bosh3()
    elif solver_type == "Kvaerno5":
        solver = Kvaerno5()
    else:
        raise ValueError(f"Solver '{solver_type}' is not one of the specified "
                         f"solvers. Use 'Dopri5', 'Bosh3', or 'Kvaerno5'.")

    term = ODETerm(_ensure_3arg(func))
    n_save = max_steps // save_every
    N = y0_batch.shape[0]
    state_dim = y0_batch.shape[1]

    # -- JIT kernel: one save group (save_every steps) ---------------------
    # The while_loop carry is TINY (no output arrays).  XLA double-buffers
    # this carry, but at ~1 KiB/atom that is negligible.

    def _single_group(carry):
        """Run up to save_every ODE steps for one atom."""
        def cond_fn(s):
            return ((s['t'] < tf) &
                    (s['count'] < save_every) &
                    (s['step_idx'] < max_steps))

        def body_fn(s):
            t_curr = s['t']
            dt_curr = s['dt']
            t_next = jnp.minimum(t_curr + dt_curr, tf)
            actual_dt = t_next - t_curr

            sol = diffeqsolve(
                term, solver,
                t0=t_curr, t1=t_next, dt0=dt_curr,
                y0=s['y'], args=args,
                stepsize_controller=PIDController(rtol=rtol, atol=atol),
                saveat=SaveAt(t1=True),
                max_steps=inner_max_steps,
            )
            y_next = sol.ys[-1]
            nfev_add = sol.stats['num_steps']

            y_jump, n_scatters, dt_max_suggested, key_new = random_func(
                t_next, y_next, actual_dt, s['key'], args
            )
            dt_next = jnp.minimum(dt_max_suggested, max_step_global)

            return {
                't': t_next, 'y': y_jump, 'dt': dt_next, 'key': key_new,
                'step_idx': s['step_idx'] + 1,
                'count': s['count'] + 1,
                'nfev': s['nfev'] + nfev_add,
                'last_t_random': jnp.where(
                    n_scatters > 0, t_next, s['last_t_random']),
                'last_n_random': jnp.where(
                    n_scatters > 0, jnp.int32(n_scatters),
                    s['last_n_random']),
            }

        init = {**carry, 'count': jnp.int32(0)}
        final = jax.lax.while_loop(cond_fn, body_fn, init)
        # Drop the loop counter — not needed outside the group.
        return {k: v for k, v in final.items() if k != 'count'}

    @jax.jit
    def _run_group(carry_batch):
        return jax.vmap(_single_group)(carry_batch)

    # -- Initialise carry (tiny, on GPU) -----------------------------------
    carry = {
        't': jnp.full(N, t0, dtype=jnp.float64),
        'y': y0_batch,
        'dt': jnp.full(N, dt0, dtype=jnp.float64),
        'key': keys_batch,
        'step_idx': jnp.ones(N, dtype=jnp.int32),
        'nfev': jnp.zeros(N, dtype=jnp.int32),
        'last_t_random': jnp.zeros(N, dtype=jnp.float64),
        'last_n_random': jnp.zeros(N, dtype=jnp.int32),
    }

    # -- Pre-allocate CPU output arrays ------------------------------------
    n_total = n_save + 1  # +1 for initial state in slot 0
    ts_cpu = np.zeros((N, n_total), dtype=np.float64)
    ys_cpu = np.zeros((N, n_total, state_dim), dtype=np.float64)
    t_random_cpu = np.zeros((N, n_total), dtype=np.float64)
    n_random_cpu = np.zeros((N, n_total), dtype=np.int32)

    # Slot 0: initial state
    ts_cpu[:, 0] = np.asarray(carry['t'])
    ys_cpu[:, 0, :] = np.asarray(carry['y'])

    # -- Host loop: run groups, snapshot to CPU ----------------------------
    tf_float = float(tf)
    save_idx = 1
    _log_interval = max(1, n_save // 20)  # ~5 % increments
    _t_first_group = None

    for _gi in range(n_save):
        carry = _run_group(carry)

        # Start timer after first group (which includes JIT compilation).
        if _gi == 0:
            _t_first_group = time.monotonic()

        if progress and _gi > 0 and _gi % _log_interval == 0:
            elapsed = time.monotonic() - _t_first_group
            rate = _gi / elapsed
            eta = (n_save - _gi) / rate if rate > 0 else 0
            m, s = divmod(int(eta), 60)
            h, m = divmod(m, 60)
            print(f"\r  [{_gi}/{n_save}] {100*_gi/n_save:.0f}%  "
                  f"ETA {h}h{m:02d}m{s:02d}s", end="", flush=True)

        # Tiny transfer: scalars + one state vector per atom.
        ts_cpu[:, save_idx] = np.asarray(carry['t'])
        ys_cpu[:, save_idx, :] = np.asarray(carry['y'])
        t_random_cpu[:, save_idx] = np.asarray(carry['last_t_random'])
        n_random_cpu[:, save_idx] = np.asarray(carry['last_n_random'])
        save_idx += 1

        # Reset scatter accumulators for the next group.
        carry = {
            **carry,
            'last_t_random': jnp.zeros(N, dtype=jnp.float64),
            'last_n_random': jnp.zeros(N, dtype=jnp.int32),
        }

        # Early exit when every atom has finished.
        t_now = ts_cpu[:, save_idx - 1]
        steps_now = np.asarray(carry['step_idx'])
        if np.all((t_now >= tf_float) | (steps_now >= max_steps)):
            break

    if progress:
        print(f"\r  [{n_save}/{n_save}] 100%  done.          ")

    # Compute per-atom save_idx (number of valid output points).
    # All atoms save the same number of groups, so save_idx is uniform.
    save_idx_arr = np.full(N, save_idx, dtype=np.int32)

    return {
        't': np.asarray(carry['t']),
        'y': np.asarray(carry['y']),
        'step_idx': np.asarray(carry['step_idx']),
        'save_idx': save_idx_arr,
        'ts': ts_cpu,
        'ys': ys_cpu,
        't_random': t_random_cpu,
        'n_random': n_random_cpu,
        'nfev': np.asarray(carry['nfev']),
    }

def _bytes_per_atom(state_dim, max_steps, inner_max_steps=64, save_every=1):
    """Estimated peak GPU allocation per atom in the batched solver.

    The solver accumulates output on **CPU** via a host loop, so no
    output history arrays reside on GPU.  Only the small per-atom
    while_loop carry and diffrax scratch buffers are on device.  Peak
    GPU memory is therefore independent of ``max_steps`` and
    ``save_every``.

    While_loop carry (double-buffered by XLA):
        y, t, dt, key, counters, scatter info     ~(state_dim * 8 + 100) bytes × 2

    Diffrax internals per inner solve:
        RK stages:      ~7 * state_dim             float64  (Dopri5 + PID)
        step buffers:   (inner_max_steps, state_dim) float64
    """
    # Carry state (double-buffered by XLA's while_loop):
    #   y (state_dim f64), t (f64), dt (f64), key (2×u32),
    #   step_idx (i32), count (i32), nfev (i32),
    #   last_t_random (f64), last_n_random (i32)  ≈ state_dim*8 + 48
    carry = 2 * (state_dim * 8 + 100)
    # Diffrax internal buffers inside the body:
    diffrax_buf = inner_max_steps * state_dim * 8 + 7 * state_dim * 8
    per_atom = carry + diffrax_buf
    return int(per_atom)


def optimal_batch_size(state_dim, max_steps, inner_max_steps=64, safety=0.6, save_every=1):
    """Return the largest batch size that fits across all available GPUs.

    Estimates peak per-atom allocation inside ``_batched_random_trajectories``
    and computes how many atoms each GPU can hold.  When multiple GPUs are
    present the total batch size is the per-GPU capacity multiplied by the
    number of devices (using the minimum free memory across GPUs in case
    they are not identical).

    Call this *after* JIT warm-up so that JAX's internal pool is already
    reflected in ``bytes_in_use``.

    Args:
        state_dim (int): Length of the per-atom state vector (``y0.shape[-1]``).
        max_steps (int): Outer buffer size passed to the solver.
        inner_max_steps (int, optional): Inner diffrax buffer size. Defaults to 64.
        safety (float, optional): Fraction of free memory to use. Defaults to 0.6
            to leave headroom for XLA workspace and compilation buffers.
        save_every (int, optional): Stride passed to ``_batched_random_trajectories``.
            Output buffers are sized ``max_steps // save_every + 1``. Defaults to 1.

    Returns:
        int: Recommended total batch size, or ``None`` if no GPU is present.
    """
    gpu_devs = _gpu_devices()
    if not gpu_devs:
        return None

    _log_gpu_debug(gpu_devs, "optimal_batch_size")

    infos = _gpu_device_info(gpu_devs)
    bpa = _bytes_per_atom(state_dim, max_steps, inner_max_steps, save_every)

    # Per-GPU capacity.
    capacities = []
    for info in infos:
        if info['bytes_limit'] == 0:
            return None
        cap = max(1, int(info['bytes_free'] * safety / bpa))
        capacities.append(cap)

    if _GPU_DEBUG:
        for i, (info, cap) in enumerate(zip(infos, capacities)):
            _log.info(
                f"[GPU DEBUG: optimal_batch_size]   GPU {i}: "
                f"free={info['bytes_free']/2**20:.1f} MiB, "
                f"capacity={cap} atoms "
                f"({cap * bpa / 2**20:.1f} MiB at {bpa/2**20:.3f} MiB/atom)"
            )

    # Even sharding: limited by the smallest GPU.
    min_cap = min(capacities)
    total = min_cap * len(gpu_devs)

    if _GPU_DEBUG and len(gpu_devs) > 1:
        max_cap = max(capacities)
        wasted = sum(c - min_cap for c in capacities)
        _log.info(
            f"[GPU DEBUG: optimal_batch_size]   Even-shard total: {total} atoms "
            f"(min_cap={min_cap}, max_cap={max_cap}, "
            f"wasted capacity={wasted} atoms across {len(gpu_devs)} GPUs)"
        )

    return total


def optimal_batch_size_per_gpu(state_dim, max_steps, inner_max_steps=64, safety=0.6, save_every=1):
    """Return per-GPU atom capacities for heterogeneous multi-GPU setups.

    Unlike :func:`optimal_batch_size` which returns a single total capped by
    the smallest GPU, this function returns each GPU's individual capacity so
    callers can assign proportional work.

    Args:
        state_dim, max_steps, inner_max_steps, safety: Same as
            :func:`optimal_batch_size`.
        save_every (int, optional): Stride for output decimation. Defaults to 1.

    Returns:
        list of (jax.Device, int) pairs, or ``None`` if no GPU is present.
        The int is the number of atoms that GPU can handle.
    """
    gpu_devs = _gpu_devices()
    if not gpu_devs:
        return None

    infos = _gpu_device_info(gpu_devs)
    bpa = _bytes_per_atom(state_dim, max_steps, inner_max_steps, save_every)

    result = []
    for info in infos:
        if info['bytes_limit'] == 0:
            return None
        cap = max(1, int(info['bytes_free'] * safety / bpa))
        result.append((info['device'], cap))
    return result


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
    args=None,
    save_every=1,
    progress=False,
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
        save_every (int, optional):
            Output decimation stride.  Only every ``save_every``-th outer
            step is written to the history buffers, so the output contains
            ``max_steps // save_every`` points while the loop still runs
            for ``max_steps`` steps.  Memory scales with the output size,
            not the loop count.  Defaults to 1 (save every step).
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
        batch_size = optimal_batch_size(state_dim, max_steps, inner_max_steps, save_every=save_every) or N

    gpu_devs = _gpu_devices()
    n_gpus = len(gpu_devs)

    if _GPU_DEBUG:
        _log_gpu_debug(gpu_devs, "solve_ivp_random")
        _log.info(
            f"[GPU DEBUG: solve_ivp_random] N={N}, batch_size={batch_size}, "
            f"n_gpus={n_gpus}, state_dim={y0_batch.shape[-1]}, "
            f"max_steps={max_steps}"
        )

    def _run_chunk(y0_chunk, keys_chunk, chunk_N):
        """Run a single chunk, sharding across GPUs if multiple are available."""
        if n_gpus > 1:
            y0_chunk, _ = _shard_batch(y0_chunk, gpu_devs)
            keys_chunk, _ = _shard_batch(keys_chunk, gpu_devs)
        # _batched_random_trajectories returns CPU numpy arrays directly
        # (output is accumulated on host, not GPU).
        batched_state = _batched_random_trajectories(
            fun, random_func, t0, tf,
            y0_chunk, keys_chunk,
            max_steps, max_step, rtol, atol, dt0,
            solver_type, inner_max_steps, args,
            save_every=save_every,
            progress=progress,
        )
        tf_float = float(tf)
        return [
            RandomOdeResult(_batched_state=batched_state, _index=i, _tf=tf_float)
            for i in range(chunk_N)
        ]

    # When sharding across multiple GPUs, round chunk sizes up to be
    # divisible by n_gpus so that _shard_batch padding is minimal.
    if n_gpus > 1 and batch_size >= n_gpus:
        batch_size = (batch_size // n_gpus) * n_gpus

    if batch_size >= N:
        return _run_chunk(y0_batch, keys_batch, N)

    # Chunked execution to limit peak memory.
    results = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        results.extend(_run_chunk(
            y0_batch[start:end], keys_batch[start:end], end - start
        ))
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

    # Shard across multiple GPUs when available.
    gpu_devs = _gpu_devices()
    if _GPU_DEBUG:
        _log_gpu_debug(gpu_devs, "solve_ivp_dense")
        _log.info(
            f"[GPU DEBUG: solve_ivp_dense] N={y0_batch.shape[0]}, "
            f"n_gpus={len(gpu_devs)}, state_dim={y0_batch.shape[-1]}, "
            f"n_points={n_points}"
        )
    if len(gpu_devs) > 1:
        y0_batch, orig_N = _shard_batch(y0_batch, gpu_devs)
        if jnp.ndim(t1) > 0:
            t1, _ = _shard_batch(t1, gpu_devs)
        if jnp.ndim(t0) > 0:
            t0, _ = _shard_batch(t0, gpu_devs)
    else:
        orig_N = y0_batch.shape[0]

    ys, ts = _batched_dense_trajectories(
        func, t0, t1, y0_batch, n_points, max_steps, rtol, atol, solver_type, args
    )
    # Strip padding atoms added by _shard_batch.
    if ys.shape[0] > orig_N:
        ys = ys[:orig_N]
        if ts.ndim > 1:
            ts = ts[:orig_N]
    return ts, ys


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp

    def dydt(t, y):
        return jnp.array([-y[1], y[0]])

    def func2(t, y, dt, key, args):
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