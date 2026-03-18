"""
Benchmark: CPU vs GPU for evolve motion.

Evolve motion: serial (N_SERIAL atoms) vs pathos CPU parallel vs GPU batched.

Speedup is reported as time-per-atom so the runs don't need equal atom counts.

Amdahl's Law estimate
---------------------
From the measured serial and pathos wall times we estimate the parallelizable
fraction p of the workload and project to arbitrary core counts:

    S(n) = 1 / ((1 - p) + p/n)

where n is the number of cores and p is solved from the two measurements:

    S_measured = T_serial / T_pathos(n_test)
    p = (1 - 1/S_measured) / (1 - 1/n_test)

Each pathos worker is a fresh process that JIT-compiles on its first atom call.
Workers are given N_ATOMS_PER_WORKER atoms each so that JIT overhead is
amortised over real work.  Wall time reflects realistic CPU parallel throughput.

Uses the notebook 03 parameters (alpha=1e-4, det=-2.5, s=1.25).
"""
import os

# Must happen before pylcp/jax are imported.  When multiprocess spawns a worker
# it re-executes this file; the _PYLCP_BENCH_WORKER flag (set by run_pathos
# before pool creation) tells us to pin JAX to CPU so workers never touch GPU.
if os.environ.get('_PYLCP_BENCH_WORKER') == '1':
    os.environ['JAX_PLATFORMS'] = 'cpu'
    # Minimise per-worker thread count: parallelism comes from multiple
    # processes, so each worker needs only minimal intra-op threads.  This
    # keeps total PID usage within cgroup limits even at high core counts.
    # No performance impact: workers process atoms sequentially and the
    # matrices are tiny, so intra-op parallelism is not beneficial.
    os.environ['XLA_FLAGS'] = (
        os.environ.get('XLA_FLAGS', '')
        + ' --xla_cpu_multi_thread_eigen=false'
        + ' --xla_force_host_platform_device_count=1'
    )
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')

# Limit per-process thread counts to avoid exhausting the OS thread limit
# when pathos spawns many workers (each would otherwise create 64+ threads
# for OpenBLAS/OMP/XLA).  Workers get parallelism from multiple processes.
for _tvar in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_tvar, '1')

import time
import numpy as np
import pylcp
from pylcp.integration_tools_gpu import optimal_batch_size
import jax
import jax.numpy as jnp

DET = -2.5
S = 1.25
ALPHA = 1e-4
N_SERIAL = 4              # atoms for serial (shared y0s used for numeric check)
PATHOS_CORE_COUNTS = [2, 4, 8]
# Each worker handles this many atoms so JIT overhead is amortised over real work.
# N_PATHOS_ATOMS is fixed independently of PATHOS_CORE_COUNTS so that adding or
# removing core counts does not change the atom pool and skew per-atom timings.
# Must be divisible by all entries in _ALL_PATHOS_CORE_COUNTS.
N_ATOMS_PER_WORKER = 4
N_PATHOS_ATOMS = 32  # fixed; divisible by 2, 4, 8, 16
SEED = 42
MAX_STEPS = 10000

# Core counts to project via Amdahl's Law
AMDAHL_CORE_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, os.cpu_count()]

# Sweep parameters
# Evolve motion: atom counts for CPU and GPU sweeps.
# CPU serial is expensive (~3s/atom) so capped lower than GPU.
EVOLVE_SWEEP_CPU_ATOMS = [4, 8, 16, 32, 64, 128]
EVOLVE_SWEEP_GPU_ATOMS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]


def setup():
    Hg, Bgq = pylcp.hamiltonians.singleF(F=0, gF=0, muB=1)
    He, Beq = pylcp.hamiltonians.singleF(F=1, gF=1, muB=1)
    dijq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)
    hamiltonian = pylcp.hamiltonian(
        Hg, -DET * np.eye(3) + He, Bgq, Beq, dijq, mass=100
    )
    laserBeams = pylcp.conventional3DMOTBeams(
        s=S, delta=0., beam_type=pylcp.infinitePlaneWaveBeam
    )
    magField = pylcp.quadrupoleMagneticField(ALPHA)
    obe = pylcp.obe(laserBeams, magField, hamiltonian,
                    transform_into_re_im=True)
    return obe


def run_evolve_sweep(obe):
    """Sweep over atom counts; return {n: (t_cpu_per_atom, t_gpu_per_atom)}.

    CPU is measured only for EVOLVE_SWEEP_CPU_ATOMS (expensive at large N).
    GPU is measured for EVOLVE_SWEEP_GPU_ATOMS.  Missing values are None.
    A single warmup call is made before each new atom count so JIT cost is
    not included in the measured time.
    """
    t_span = [0, 2 * np.pi * 500]
    kw_cpu = dict(freeze_axis=[True, True, False], max_steps=MAX_STEPS, backend='cpu')
    kw_gpu = dict(freeze_axis=[True, True, False], max_steps=MAX_STEPS, backend='gpu')

    all_counts = sorted(set(EVOLVE_SWEEP_CPU_ATOMS) | set(EVOLVE_SWEEP_GPU_ATOMS))
    results = {n: [None, None] for n in all_counts}

    for n in all_counts:
        y0_list = make_y0_list(obe, n)
        y0_batch = jnp.stack(y0_list)
        keys = jax.random.split(jax.random.PRNGKey(SEED), n)
        print(f"\n  --- {n} atoms ---")

        if n in EVOLVE_SWEEP_CPU_ATOMS:
            # Warmup
            obe.evolve_motion(t_span, y0_batch=y0_batch[:min(n, 2)],
                              keys_batch=keys[:min(n, 2)], **kw_cpu)
            t0 = time.perf_counter()
            obe.evolve_motion(t_span, y0_batch=y0_batch, keys_batch=keys, **kw_cpu)
            t_cpu = (time.perf_counter() - t0) / n
            results[n][0] = t_cpu
            print(f"    CPU: {t_cpu:.4f}s/atom")

        if n in EVOLVE_SWEEP_GPU_ATOMS:
            # Warmup
            obe.evolve_motion(t_span, y0_batch=y0_batch[:min(n, 2)],
                              keys_batch=keys[:min(n, 2)], **kw_gpu)
            t0 = time.perf_counter()
            obe.evolve_motion(t_span, y0_batch=y0_batch, keys_batch=keys, **kw_gpu)
            t_gpu = (time.perf_counter() - t0) / n
            results[n][1] = t_gpu
            print(f"    GPU: {t_gpu:.4f}s/atom")

    return results


def make_plots(evolve_data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    ns_e = sorted(evolve_data.keys())
    cpu_pts = [(n, evolve_data[n][0]) for n in ns_e if evolve_data[n][0] is not None]
    gpu_pts = [(n, evolve_data[n][1]) for n in ns_e if evolve_data[n][1] is not None]
    if cpu_pts:
        ax.plot(*zip(*cpu_pts), 'o-', label='Serial CPU')
    if gpu_pts:
        ax.plot(*zip(*gpu_pts), 's-', label='GPU batched')
    ax.set_xlabel('Number of atoms (N)')
    ax.set_ylabel('Time per atom (s)')
    ax.set_title('Evolve Motion: CPU vs GPU')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'benchmark_cpu_vs_gpu_{_ts}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to {out}")
    plt.close()


def warmup_evolve(obe):
    """JIT warmup for evolve_motion (single and batched)."""
    obe.set_initial_rho_equally()
    rho0 = jnp.array(obe.rho0)
    y0 = jnp.concatenate([rho0, jnp.zeros(3), jnp.zeros(3)])

    obe.evolve_motion([0, 100], y0_batch=y0[jnp.newaxis, :],
                      freeze_axis=[True, True, False], max_steps=200, backend='cpu')

    y0_batch = jnp.stack([y0, y0])
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    obe.evolve_motion([0, 100], y0_batch=y0_batch, keys_batch=keys,
                      freeze_axis=[True, True, False], max_steps=200, backend='gpu')


def make_y0_list(obe, n_atoms):
    np.random.seed(SEED)
    rho0 = jnp.array(obe.rho0)
    r_init = np.random.uniform(-2 / ALPHA, 2 / ALPHA, size=(n_atoms, 3))
    r_init[:, :2] = 0.
    y0_list = [
        jnp.concatenate([rho0, jnp.zeros(3), jnp.array(r_init[i])])
        for i in range(n_atoms)
    ]
    return y0_list


def run_serial(obe, y0_list):
    """Run atoms one at a time via backend='cpu'; return (time_per_atom, final_z)."""
    n_atoms = len(y0_list)
    kw = dict(freeze_axis=[True, True, False], max_steps=MAX_STEPS, backend='cpu')
    t_span = [0, 2 * np.pi * 500]

    y0_batch = jnp.stack(y0_list)
    keys = jax.random.split(jax.random.PRNGKey(SEED), n_atoms)

    t0 = time.perf_counter()
    obe.evolve_motion(t_span, y0_batch=y0_batch, keys_batch=keys, **kw)
    elapsed = time.perf_counter() - t0

    final_z = np.array([sol.r[2, -1] for sol in obe.sols])
    time_per_atom = elapsed / n_atoms
    print(f"\n  Serial CPU ({n_atoms} atoms, backend='cpu'):")
    print(f"    Total time:      {elapsed:.1f}s")
    print(f"    Time per atom:   {time_per_atom:.3f}s")
    print(f"    Final z:         mean={np.mean(final_z):.2f}, std={np.std(final_z):.2f}")
    return time_per_atom, final_z


def run_batched(obe, y0_list):
    """Run atoms in one GPU-batched call via backend='gpu'; return (time_per_atom, final_z)."""
    n_atoms = len(y0_list)
    kw = dict(freeze_axis=[True, True, False], max_steps=MAX_STEPS, backend='gpu')
    t_span = [0, 2 * np.pi * 500]

    y0_batch = jnp.stack(y0_list)
    keys = jax.random.split(jax.random.PRNGKey(SEED), n_atoms)

    t0 = time.perf_counter()
    obe.evolve_motion(t_span, y0_batch=y0_batch, keys_batch=keys, **kw)
    elapsed = time.perf_counter() - t0

    final_z = np.array([sol.r[2, -1] for sol in obe.sols])
    time_per_atom = elapsed / n_atoms
    print(f"\n  Batched GPU ({n_atoms} atoms):")
    print(f"    Total time:      {elapsed:.1f}s")
    print(f"    Time per atom:   {time_per_atom:.3f}s")
    print(f"    Final z:         mean={np.mean(final_z):.2f}, std={np.std(final_z):.2f}")
    return time_per_atom, final_z


# ---------------------------------------------------------------------------
# Pathos worker — must be a module-level function so dill can pickle it.
# Each worker process is fresh: it builds its own obe and JIT-compiles on the
# first evolve_motion call.  That overhead is included deliberately.
# ---------------------------------------------------------------------------
def _pathos_worker(y0_batch_np):
    """Run a batch of atom trajectories in a worker process; return list of final z.

    Accepts a list of numpy arrays (one per atom) and runs them sequentially.
    Building obe and the first evolve_motion call trigger JIT compilation; that
    cost is amortised over all atoms in the batch.  JAX_PLATFORMS=cpu is already
    set at module-import time via the _PYLCP_BENCH_WORKER env flag.
    """
    import numpy as _np
    import pylcp as _pylcp
    import jax.numpy as _jnp

    Hg, Bgq = _pylcp.hamiltonians.singleF(F=0, gF=0, muB=1)
    He, Beq = _pylcp.hamiltonians.singleF(F=1, gF=1, muB=1)
    dijq = _pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)
    hamiltonian = _pylcp.hamiltonian(
        Hg, -DET * _np.eye(3) + He, Bgq, Beq, dijq, mass=100
    )
    laserBeams = _pylcp.conventional3DMOTBeams(
        s=S, delta=0., beam_type=_pylcp.infinitePlaneWaveBeam
    )
    obe = _pylcp.obe(laserBeams, _pylcp.quadrupoleMagneticField(ALPHA),
                     hamiltonian, transform_into_re_im=True)
    obe.set_initial_rho_equally()

    results = []
    for y0_np in y0_batch_np:
        y0 = _jnp.array(y0_np)
        obe.evolve_motion(
            [0, 2 * _np.pi * 500],
            y0_batch=y0[_jnp.newaxis, :],
            freeze_axis=[True, True, False],
            max_steps=MAX_STEPS,
            backend='cpu',
        )
        results.append(float(obe.sol.r[2, -1]))
    return results


def run_pathos(y0_list, n_cores):
    """Run atoms in parallel using pathos; return (time_per_atom, final_z).

    Atoms are split evenly across n_cores workers.  Each worker runs its chunk
    sequentially so JIT-compile overhead (~10s) is paid once per worker and
    amortised over N_ATOMS_PER_WORKER atoms.

    _PYLCP_BENCH_WORKER=1 is set before pool creation; spawned children inherit
    it and the module-level guard sets JAX_PLATFORMS=cpu before pylcp imports,
    preventing GPU init in workers while the parent holds all GPU memory.
    """
    import multiprocess
    multiprocess.set_start_method('spawn', force=True)
    from pathos.multiprocessing import ProcessPool

    n_atoms = len(y0_list)
    # Convert to plain numpy so dill can serialise without JAX device refs.
    y0_np_list = [np.array(y0) for y0 in y0_list]

    # Split atoms into n_cores chunks (one per worker).
    chunks = [y0_np_list[i::n_cores] for i in range(n_cores)]

    # Set worker flag before spawning so children inherit it.
    os.environ['_PYLCP_BENCH_WORKER'] = '1'
    pool = ProcessPool(nodes=n_cores)
    try:
        t0 = time.perf_counter()
        results_nested = pool.map(_pathos_worker, chunks)
        elapsed = time.perf_counter() - t0
    finally:
        # Always terminate and join workers so no stale processes are left
        # holding XLA thread pools after a crash or exception.
        pool.terminate()
        pool.join()
        pool.clear()
        os.environ.pop('_PYLCP_BENCH_WORKER', None)

    # Reassemble: chunks[i][j] → atom index i + j*n_cores
    final_z = np.empty(n_atoms)
    for i, chunk_results in enumerate(results_nested):
        for j, z in enumerate(chunk_results):
            final_z[i + j * n_cores] = z

    time_per_atom = elapsed / n_atoms
    print(f"\n  Pathos CPU ({n_atoms} atoms, {n_cores} cores, {n_atoms//n_cores} atoms/worker):")
    print(f"    Total time:      {elapsed:.1f}s")
    print(f"    Time per atom:   {time_per_atom:.3f}s")
    print(f"    Final z:         mean={np.mean(final_z):.2f}, std={np.std(final_z):.2f}")
    return time_per_atom, final_z


def amdahl_speedup(p, n):
    """Amdahl's Law: predicted speedup for n cores given parallelisable fraction p.

    Formula
    -------
        S(n) = 1 / ((1 - p) + p / n)

    where:
        p  — fraction of work that can run in parallel  (0 ≤ p ≤ 1)
        n  — number of cores
        S  — wall-time speedup relative to 1 core

    Solving for p from two measurements (serial T_1 and parallel T_n):
        S_measured = T_1 / T_n
        p = (1 - 1 / S_measured) / (1 - 1 / n)
    """
    return 1.0 / ((1.0 - p) + p / n)


if __name__ == '__main__':
    import sys
    import datetime

    # Output directory: same folder as this script.
    # Each run gets a timestamp suffix so multiple runs don't overwrite each other.
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
    _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _log_path = os.path.join(OUT_DIR, f'benchmark_output_{_ts}.txt')
    _log_file = open(_log_path, 'w')

    class _Tee:
        """Write to both stdout and the log file."""
        def __init__(self, stream, log):
            self._stream, self._log = stream, log
        def write(self, data):
            self._stream.write(data)
            self._log.write(data)
            self._log.flush()
        def flush(self):
            self._stream.flush()
            self._log.flush()
        def __getattr__(self, attr):
            return getattr(self._stream, attr)

    sys.stdout = _Tee(sys.stdout, _log_file)

    print(f"Run started: {datetime.datetime.now().isoformat()}")

    # Hardware info
    print(f"\nHardware:")
    print(f"  CPU cores (logical): {os.cpu_count()}")
    _gpu_devs = [d for d in jax.devices() if d.platform == 'gpu']
    _cpu_devs = [d for d in jax.devices() if d.platform == 'cpu']
    for d in _gpu_devs:
        mem = d.memory_stats()
        print(f"  GPU: {d}  memory={mem['bytes_limit']/2**30:.1f} GiB")
    if not _gpu_devs:
        print(f"  GPU: none detected")
    for d in _cpu_devs:
        print(f"  CPU device: {d}")

    print(f"\nParameters: det={DET}, s={S}, alpha={ALPHA}")
    print(f"Seed: {SEED}")

    obe = setup()

    # --- Evolve motion benchmark ---
    print("\n" + "=" * 50)
    print("  Evolve Motion Benchmark")
    print("=" * 50)

    print("\nWarming up JIT (evolve motion)...")
    warmup_evolve(obe)

    # Determine optimal GPU batch size from live memory after warmup.
    # state_dim is exact: rho0 length (set by obe internals) + v(3) + r(3).
    state_dim = len(obe.rho0) + 6
    n_batched = optimal_batch_size(state_dim, MAX_STEPS, inner_max_steps=64, safety=0.6)
    if n_batched is None:
        n_batched = 64  # CPU fallback
    print(f"\n  state_dim={state_dim}, optimal GPU batch size: {n_batched}")

    # Build all y0s once with a fixed seed.
    # pathos_y0: N_PATHOS_ATOMS atoms (fixed, independent of PATHOS_CORE_COUNTS)
    # reused across all core-count runs so wall-times are directly comparable.
    # GPU batch uses n_batched atoms; first N_SERIAL shared for numeric check.
    all_y0    = make_y0_list(obe, max(n_batched, N_PATHOS_ATOMS))
    pathos_y0 = all_y0[:N_PATHOS_ATOMS]
    gpu_y0    = all_y0[:n_batched]

    t_per_atom_serial, z_serial = run_serial(obe, pathos_y0)

    # Run pathos at each core count; collect (n_cores, t_per_atom, z).
    # Wrap in try/except so stale worker processes are killed on error
    # and the benchmark can continue with whatever results were collected.
    pathos_results = {}
    try:
        for n_cores in PATHOS_CORE_COUNTS:
            t_pa, z_pa = run_pathos(pathos_y0, n_cores)
            pathos_results[n_cores] = (t_pa, z_pa)
    except Exception as e:
        print(f"\n  Pathos failed at {n_cores} cores: {e}")
        print(f"  Continuing with {len(pathos_results)} successful core counts.")
        # Kill any orphaned worker processes spawned by pathos.
        import subprocess
        subprocess.run(
            ['pkill', '-P', str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    t_per_atom_batch, z_batch = run_batched(obe, gpu_y0)

    # --- Numerical check (first N_SERIAL atoms identical across all runs) ---
    print(f"\n  Numerical check (first {N_SERIAL} atoms vs serial):")
    for n_cores, (_, z_pa) in pathos_results.items():
        diff = np.max(np.abs(z_serial[:N_SERIAL] - z_pa[:N_SERIAL]))
        print(f"    Pathos {n_cores:>2} cores  max|z diff|:  {diff:.4e}")
    zdiff_gpu = z_serial[:N_SERIAL] - z_batch[:N_SERIAL]
    print(f"    GPU             max|z diff|:  {np.max(np.abs(zdiff_gpu)):.4e}")

    # --- Amdahl's Law projection ---
    #
    # S(n) = 1 / ((1 - p) + p/n)
    #
    # Linearised form: let x = 1 - 1/n, y = 1 - 1/S  →  y = p·x
    # Closed-form least-squares through the origin: p = Σ(xᵢ·yᵢ) / Σ(xᵢ²)
    # This fits all measurements simultaneously and naturally down-weights
    # low core counts (small x) which carry less information about p.
    #
    if len(pathos_results) >= 2:
        measured_speedups = {}
        xs, ys = [], []
        for n_cores, (t_pa, _) in pathos_results.items():
            s_i = t_per_atom_serial / t_pa
            measured_speedups[n_cores] = s_i
            xs.append(1.0 - 1.0 / n_cores)
            ys.append(1.0 - 1.0 / s_i)

        xs, ys = np.array(xs), np.array(ys)
        p_parallel = float(np.clip(np.dot(xs, ys) / np.dot(xs, xs), 0.0, 1.0))

        print(f"\n  Amdahl's Law (CPU parallel projection)")
        print(f"  Formula: S(n) = 1 / ((1 - p) + p/n)")
        print(f"  Fit: linearised least-squares  p = Σ(xᵢyᵢ)/Σ(xᵢ²)  where x=1-1/n, y=1-1/S")
        print(f"  Measurements used for fit:")
        for n_cores in sorted(pathos_results.keys()):
            print(f"    {n_cores:>2} cores: S={measured_speedups[n_cores]:.2f}x")
        print(f"  Fitted p = {p_parallel:.4f}  ({p_parallel*100:.1f}% of work is parallel)")
        print(f"\n  {'Cores':>8}  {'Predicted S':>12}  {'Time/atom (s)':>15}")
        print(f"  {'-'*8}  {'-'*12}  {'-'*15}")
        measured_set = set(pathos_results.keys())
        seen: set = set()
        for n in AMDAHL_CORE_COUNTS:
            if n in seen:
                continue
            seen.add(n)
            s_pred = amdahl_speedup(p_parallel, n)
            t_pred = t_per_atom_serial / s_pred
            marker = f"  ← measured S={measured_speedups[n]:.2f}x" if n in measured_set else ""
            print(f"  {n:>8}  {s_pred:>12.2f}  {t_pred:>15.3f}{marker}")
    else:
        print(f"\n  Amdahl's Law: skipped (need ≥2 pathos measurements, got {len(pathos_results)})")

    s_gpu = t_per_atom_serial / t_per_atom_batch
    print(f"\n  GPU batched speedup for reference: {s_gpu:.1f}x  "
          f"({t_per_atom_batch:.4f}s / atom)")

    # --- Evolve motion sweep ---
    print("\n" + "=" * 50)
    print("  Evolve Motion Sweep (CPU vs GPU across atom counts)")
    print("=" * 50)
    evolve_sweep = run_evolve_sweep(obe)

    # --- Plots ---
    print("\n" + "=" * 50)
    print("  Generating plots")
    print("=" * 50)
    make_plots(evolve_sweep)

    print(f"\nRun finished: {datetime.datetime.now().isoformat()}")
    print(f"Output saved to {_log_path}")
    sys.stdout = sys.stdout._stream
    _log_file.close()
