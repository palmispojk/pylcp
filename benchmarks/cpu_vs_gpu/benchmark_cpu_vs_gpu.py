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
N_ATOMS_PER_WORKER = 4

# Amdahl overhead sweep: run pathos at different atoms-per-worker counts to see
# how JIT/spawn overhead is amortised.  Each level uses (atoms_per_worker * max_cores)
# total atoms.  More atoms per worker → higher fitted p (less overhead).
AMDAHL_ATOMS_PER_WORKER = [2, 4, 8, 16]
SEED = 42
MAX_STEPS = 10000
# Time spans to sweep: each is [0, 2*pi*T]; longer spans mean more solver work.
SWEEP_T_FACTORS = [100, 500, 2000]

# Core counts to project via Amdahl's Law
AMDAHL_CORE_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, os.cpu_count()]

# Sweep parameters
# Evolve motion: atom counts for CPU and GPU sweeps.
# CPU serial is expensive (~3s/atom) so capped lower than GPU.
SWEEP_CPU_ATOMS = [4, 8, 16, 32, 64, 128]
SWEEP_GPU_ATOMS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]


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


def run_evolve_sweep(obe, t_factor=500):
    """Unified sweep: serial CPU, pathos (all core counts), and GPU.

    For each atom count, runs (in order):
      1. Serial CPU           (if n in SWEEP_CPU_ATOMS)
      2. Pathos at each core  (if n in SWEEP_CPU_ATOMS and n >= cores)
      3. GPU batched           (if n in SWEEP_GPU_ATOMS)

    Args:
        t_factor: Time span is [0, 2*pi*t_factor].

    Returns:
        {n: {'serial': t|None, 'gpu': t|None, 'pathos': {cores: t, ...},
             'z_serial': array|None, 'z_gpu': array|None}}
    """
    t_span = [0, 2 * np.pi * t_factor]
    kw_cpu = dict(freeze_axis=[True, True, False], max_steps=MAX_STEPS, backend='cpu')
    kw_gpu = dict(freeze_axis=[True, True, False], max_steps=MAX_STEPS, backend='gpu')

    all_counts = sorted(set(SWEEP_CPU_ATOMS) | set(SWEEP_GPU_ATOMS))
    results = {}

    for n in all_counts:
        results[n] = {'serial': None, 'gpu': None, 'pathos': {},
                      'z_serial': None, 'z_gpu': None}
        y0_list = make_y0_list(obe, n)
        y0_batch = jnp.stack(y0_list)
        keys = jax.random.split(jax.random.PRNGKey(SEED), n)
        print(f"\n  --- {n} atoms ---")

        if n in SWEEP_CPU_ATOMS:
            # Serial CPU
            obe.evolve_motion(t_span, y0_batch=y0_batch[:min(n, 2)],
                              keys_batch=keys[:min(n, 2)], **kw_cpu)
            t0 = time.perf_counter()
            obe.evolve_motion(t_span, y0_batch=y0_batch, keys_batch=keys, **kw_cpu)
            t_cpu = (time.perf_counter() - t0) / n
            z_serial = np.array([sol.r[2, -1] for sol in obe.sols])
            results[n]['serial'] = t_cpu
            results[n]['z_serial'] = z_serial
            print(f"    CPU serial: {t_cpu:.4f}s/atom")

            # Pathos at each core count (only where n >= cores)
            for n_cores in PATHOS_CORE_COUNTS:
                if n >= n_cores:
                    try:
                        t_pa, _ = run_pathos(y0_list, n_cores, t_factor=t_factor)
                        results[n]['pathos'][n_cores] = t_pa
                    except Exception as e:
                        print(f"    Pathos {n_cores} cores failed: {e}")

        if n in SWEEP_GPU_ATOMS:
            # GPU
            obe.evolve_motion(t_span, y0_batch=y0_batch[:min(n, 2)],
                              keys_batch=keys[:min(n, 2)], **kw_gpu)
            t0 = time.perf_counter()
            obe.evolve_motion(t_span, y0_batch=y0_batch, keys_batch=keys, **kw_gpu)
            t_gpu = (time.perf_counter() - t0) / n
            z_gpu = np.array([sol.r[2, -1] for sol in obe.sols])
            results[n]['gpu'] = t_gpu
            results[n]['z_gpu'] = z_gpu
            print(f"    GPU: {t_gpu:.4f}s/atom")

    return results


def make_plots(sweep_data, t_factor):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    ns = sorted(sweep_data.keys())

    # Serial CPU
    cpu_pts = [(n, sweep_data[n]['serial']) for n in ns
               if sweep_data[n]['serial'] is not None]
    if cpu_pts:
        ax.plot(*zip(*cpu_pts), 'o-', label='Serial CPU')

    # Pathos lines (one per core count)
    all_cores = sorted({c for n in ns for c in sweep_data[n]['pathos']})
    markers = ['^', 'v', 'D', 'p', 'h', '*']
    for i, n_cores in enumerate(all_cores):
        pts = [(n, sweep_data[n]['pathos'][n_cores]) for n in ns
               if n_cores in sweep_data[n]['pathos']]
        if pts:
            m = markers[i % len(markers)]
            ax.plot(*zip(*pts), f'{m}-', label=f'Pathos CPU ({n_cores} cores)')

    # GPU
    gpu_pts = [(n, sweep_data[n]['gpu']) for n in ns
               if sweep_data[n]['gpu'] is not None]
    if gpu_pts:
        ax.plot(*zip(*gpu_pts), 's-', label='GPU batched')

    ax.set_xlabel('Number of atoms (N)')
    ax.set_ylabel('Time per atom (s)')
    ax.set_title(f'Evolve Motion: CPU vs GPU  (t=2\u03c0\u00d7{t_factor})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'benchmark_cpu_vs_gpu_t{t_factor}_{_ts}.png')
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


def run_serial(obe, y0_list, t_factor=500):
    """Run atoms one at a time via backend='cpu'; return (time_per_atom, final_z)."""
    n_atoms = len(y0_list)
    kw = dict(freeze_axis=[True, True, False], max_steps=MAX_STEPS, backend='cpu')
    t_span = [0, 2 * np.pi * t_factor]

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

    t_factor = float(os.environ.get('_PYLCP_BENCH_T_FACTOR', '500'))
    results = []
    for y0_np in y0_batch_np:
        y0 = _jnp.array(y0_np)
        obe.evolve_motion(
            [0, 2 * _np.pi * t_factor],
            y0_batch=y0[_jnp.newaxis, :],
            freeze_axis=[True, True, False],
            max_steps=MAX_STEPS,
            backend='cpu',
        )
        results.append(float(obe.sol.r[2, -1]))
    return results


def run_pathos(y0_list, n_cores, t_factor=500):
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

    # Set worker flags before spawning so children inherit them.
    os.environ['_PYLCP_BENCH_WORKER'] = '1'
    os.environ['_PYLCP_BENCH_T_FACTOR'] = str(t_factor)
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
        os.environ.pop('_PYLCP_BENCH_T_FACTOR', None)

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


def fit_amdahl_p(t_per_atom_serial, pathos_results):
    """Fit parallelisable fraction p from pathos measurements.

    Args:
        t_per_atom_serial: Serial time per atom.
        pathos_results: dict {n_cores: (t_per_atom, final_z)}.

    Returns:
        (p, measured_speedups) or (None, {}) if fewer than 2 measurements.
    """
    if len(pathos_results) < 2:
        return None, {}
    measured_speedups = {}
    xs, ys = [], []
    for n_cores, (t_pa, _) in pathos_results.items():
        s_i = t_per_atom_serial / t_pa
        measured_speedups[n_cores] = s_i
        xs.append(1.0 - 1.0 / n_cores)
        ys.append(1.0 - 1.0 / s_i)
    xs, ys = np.array(xs), np.array(ys)
    p = float(np.clip(np.dot(xs, ys) / np.dot(xs, xs), 0.0, 1.0))
    return p, measured_speedups


def run_amdahl_overhead_sweep(obe, t_factor=500):
    """Run pathos at multiple atoms-per-worker levels to show overhead amortisation.

    For each level in AMDAHL_ATOMS_PER_WORKER, runs all PATHOS_CORE_COUNTS and
    fits a separate p.  Each level gets its own serial baseline so the
    comparison is fair at that atom count.

    Returns:
        list of (atoms_per_worker, p, measured_speedups) tuples.
    """
    max_cores = max(PATHOS_CORE_COUNTS)
    results = []

    for apw in AMDAHL_ATOMS_PER_WORKER:
        n_total = apw * max_cores
        print(f"\n  --- Amdahl sweep: {apw} atoms/worker, {n_total} total atoms ---")
        y0_list = make_y0_list(obe, n_total)

        # Serial baseline at this atom count.
        t_serial, _ = run_serial(obe, y0_list, t_factor=t_factor)

        pathos_res = {}
        try:
            for n_cores in PATHOS_CORE_COUNTS:
                t_pa, z_pa = run_pathos(y0_list, n_cores, t_factor=t_factor)
                pathos_res[n_cores] = (t_pa, z_pa)
        except Exception as e:
            print(f"    Pathos failed at {n_cores} cores: {e}")

        p, speedups = fit_amdahl_p(t_serial, pathos_res)
        if p is not None:
            print(f"    Fitted p = {p:.4f}  ({p*100:.1f}% parallel)")
            results.append((apw, p, speedups))
        else:
            print(f"    Not enough data to fit p (got {len(pathos_res)} measurements)")

    return results


def make_amdahl_plot(amdahl_sweep_results, out_dir, ts, t_factor):
    """Plot Amdahl's Law curves for different atoms-per-worker levels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not amdahl_sweep_results:
        print("  No Amdahl sweep data to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    core_range = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(amdahl_sweep_results)))

    # --- Left panel: Amdahl curves ---
    for (apw, p, speedups), color in zip(amdahl_sweep_results, colors):
        predicted = [amdahl_speedup(p, n) for n in core_range]
        ax1.plot(core_range, predicted, '-', color=color, lw=2,
                 label=f'{apw} atoms/worker (p={p:.3f})')
        # Overlay measured points.
        for nc, s in speedups.items():
            ax1.plot(nc, s, 'o', color=color, markersize=7, zorder=5)

    ax1.set_xlabel('Number of cores')
    ax1.set_ylabel('Speedup S(n)')
    ax1.set_title(f"Amdahl's Law: Overhead Amortisation  (t=2\u03c0\u00d7{t_factor})")
    ax1.set_xscale('log', base=2)
    ax1.legend(fontsize=9)
    ax1.grid(True, which='both', ls='--', alpha=0.4)

    # --- Right panel: p vs atoms/worker ---
    apws = [r[0] for r in amdahl_sweep_results]
    ps = [r[1] for r in amdahl_sweep_results]
    ax2.plot(apws, [p * 100 for p in ps], 'o-', color='tab:blue', markersize=8, lw=2)
    ax2.set_xlabel('Atoms per worker')
    ax2.set_ylabel('Parallelisable fraction p (%)')
    ax2.set_title('Overhead vs Work per Worker')
    ax2.set_xscale('log', base=2)
    ax2.set_ylim(0, 105)
    ax2.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()
    out = os.path.join(out_dir, f'benchmark_amdahl_overhead_t{t_factor}_{ts}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Amdahl overhead plot saved to {out}")
    plt.close()


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

    print("\nWarming up JIT (evolve motion)...")
    warmup_evolve(obe)

    # Determine optimal GPU batch size and add it to the sweep.
    state_dim = len(obe.rho0) + 6
    n_batched = optimal_batch_size(state_dim, MAX_STEPS, inner_max_steps=64, safety=0.6)
    if n_batched is not None and n_batched not in SWEEP_GPU_ATOMS:
        SWEEP_GPU_ATOMS.append(n_batched)
        SWEEP_GPU_ATOMS.sort()
    print(f"\n  state_dim={state_dim}, optimal GPU batch size: {n_batched}")

    # --- Loop over time spans ---
    for t_factor in SWEEP_T_FACTORS:
        print("\n" + "=" * 50)
        print(f"  t = 2pi x {t_factor}  (Evolve Motion Sweep)")
        print("=" * 50)
        sweep = run_evolve_sweep(obe, t_factor=t_factor)

        # Numerical check (first N_SERIAL atoms, smallest count with both)
        check_candidates = [n for n in sweep
                            if sweep[n]['serial'] is not None
                            and sweep[n]['gpu'] is not None]
        if check_candidates:
            check_n = min(check_candidates)
            z_serial = sweep[check_n]['z_serial']
            z_gpu = sweep[check_n]['z_gpu']
            print(f"\n  Numerical check ({check_n} atoms, first {N_SERIAL} vs serial):")
            print(f"    GPU  max|z diff|:  "
                  f"{np.max(np.abs(z_serial[:N_SERIAL] - z_gpu[:N_SERIAL])):.4e}")

        # Amdahl's Law from sweep data
        amdahl_candidates = [
            n for n in sorted(sweep.keys())
            if sweep[n]['serial'] is not None
            and all(c in sweep[n]['pathos'] for c in PATHOS_CORE_COUNTS)
        ]
        if amdahl_candidates:
            amdahl_n = amdahl_candidates[-1]
            t_serial = sweep[amdahl_n]['serial']
            pathos_for_fit = {c: (sweep[amdahl_n]['pathos'][c], None)
                              for c in PATHOS_CORE_COUNTS}
            p_parallel, measured_speedups = fit_amdahl_p(t_serial, pathos_for_fit)

            if p_parallel is not None:
                print(f"\n  Amdahl's Law (from {amdahl_n}-atom sweep point)")
                print(f"  Formula: S(n) = 1 / ((1 - p) + p/n)")
                print(f"  Measurements used for fit:")
                for n_cores in sorted(PATHOS_CORE_COUNTS):
                    print(f"    {n_cores:>2} cores: S={measured_speedups[n_cores]:.2f}x")
                print(f"  Fitted p = {p_parallel:.4f}  ({p_parallel*100:.1f}% of work is parallel)")
                print(f"\n  {'Cores':>8}  {'Predicted S':>12}  {'Time/atom (s)':>15}")
                print(f"  {'-'*8}  {'-'*12}  {'-'*15}")
                measured_set = set(PATHOS_CORE_COUNTS)
                seen: set = set()
                for n in AMDAHL_CORE_COUNTS:
                    if n in seen:
                        continue
                    seen.add(n)
                    s_pred = amdahl_speedup(p_parallel, n)
                    t_pred = t_serial / s_pred
                    marker = f"  ← measured S={measured_speedups[n]:.2f}x" if n in measured_set else ""
                    print(f"  {n:>8}  {s_pred:>12.2f}  {t_pred:>15.3f}{marker}")
        else:
            print("\n  Amdahl's Law: skipped (no atom count has all core counts measured)")

        # Amdahl overhead sweep
        print(f"\n  Amdahl Overhead Sweep (t=2pi x {t_factor})")
        print(f"  Atoms per worker levels: {AMDAHL_ATOMS_PER_WORKER}")
        print(f"  Core counts per level:   {PATHOS_CORE_COUNTS}")
        amdahl_sweep = run_amdahl_overhead_sweep(obe, t_factor=t_factor)

        if amdahl_sweep:
            print(f"\n  {'Atoms/worker':>13}  {'p':>7}  {'p (%)':>7}  {'Max speedup':>12}")
            print(f"  {'-'*13}  {'-'*7}  {'-'*7}  {'-'*12}")
            for apw, p, _ in amdahl_sweep:
                s_max = 1.0 / (1.0 - p) if p < 1.0 else float('inf')
                print(f"  {apw:>13}  {p:>7.4f}  {p*100:>6.1f}%  {s_max:>12.1f}x")

        # Plots for this t_factor
        print(f"\n  Generating plots (t=2pi x {t_factor})...")
        make_plots(sweep, t_factor)
        make_amdahl_plot(amdahl_sweep, OUT_DIR, _ts, t_factor)

    print(f"\nRun finished: {datetime.datetime.now().isoformat()}")
    print(f"Output saved to {_log_path}")
    sys.stdout = sys.stdout._stream
    _log_file.close()
