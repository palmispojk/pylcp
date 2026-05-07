"""CPU-only benchmark: serial + multiprocess timings for evolve_motion.

Sweeps over multiple transitions at small atom counts (CPU is expensive).

Runs on a CPU node; never touches GPU. Output: cpu_results.pkl.
"""
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
for _tvar in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_tvar, '1')

# Worker-mode guard: when multiprocessing spawns a child it re-imports this module.
if os.environ.get('_PYLCP_BENCH_WORKER') == '1':
    os.environ['XLA_FLAGS'] = (
        os.environ.get('XLA_FLAGS', '')
        + ' --xla_cpu_multi_thread_eigen=false'
        + ' --xla_force_host_platform_device_count=1'
    )
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')

import sys
import time
import pickle
import datetime
import argparse

import numpy as np
import jax
import jax.numpy as jnp

from common import (
    N_POINTS, SEED, SWEEP_CPU_ATOMS, SWEEP_T_FACTORS, PARALLEL_CORE_COUNTS,
    TRANSITIONS, setup_obe, make_y0_list, run_parallel,
)


def warmup(obe):
    rho0 = jnp.array(obe.rho0)
    y0 = jnp.concatenate([rho0, jnp.zeros(3), jnp.zeros(3)])
    obe.evolve_motion([0, 100], n_points=8, y0_batch=y0[jnp.newaxis, :],
                      freeze_axis=[True, True, False], backend='cpu')


def run_cpu_sweep(obe, atom_counts, core_counts, t_factor, transition):
    kw = dict(freeze_axis=[True, True, False], backend='cpu')
    t_span = [0, 2 * np.pi * t_factor]
    out = {}
    for n in atom_counts:
        print(f"\n  --- N={n}, t=2pi x {t_factor} ---", flush=True)
        y0_list = make_y0_list(obe, n)
        y0_batch = jnp.stack(y0_list)
        keys = jax.random.split(jax.random.PRNGKey(SEED), n)

        # Serial.
        obe.evolve_motion(t_span, n_points=N_POINTS,
                          y0_batch=y0_batch[:min(n, 2)],
                          keys_batch=keys[:min(n, 2)], **kw)
        t0 = time.perf_counter()
        obe.evolve_motion(t_span, n_points=N_POINTS,
                          y0_batch=y0_batch, keys_batch=keys, **kw)
        t_serial = (time.perf_counter() - t0) / n
        z_serial = np.array([sol.r[2, -1] for sol in obe.sols])
        print(f"    serial:  {t_serial:.4f} s/atom", flush=True)

        entry = {'serial': t_serial, 'z_serial': z_serial, 'parallel': {}}

        for nc in core_counts:
            if n < nc:
                continue
            try:
                t_pa, _ = run_parallel(y0_list, nc, t_factor=t_factor,
                                     transition=transition)
                entry['parallel'][nc] = t_pa
                print(f"    parallel {nc:>2}: {t_pa:.4f} s/atom", flush=True)
            except Exception as exc:
                print(f"    parallel {nc}: FAILED ({exc})", flush=True)

        out[n] = entry
    return out


def main():
    if os.environ.get('_PYLCP_BENCH_WORKER') == '1':
        return

    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--cores', type=int, nargs='+', default=None,
                    help=f"Override core counts (default: {PARALLEL_CORE_COUNTS}).")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out or os.path.join(here, 'cpu_results.pkl')
    core_counts = args.cores if args.cores else list(PARALLEL_CORE_COUNTS)

    print(f"Run started: {datetime.datetime.now().isoformat()}")
    print(f"  CPU cores (logical): {os.cpu_count()}")
    print(f"  JAX backend:         {jax.default_backend()}")
    print(f"  Parallel core sweep:   {core_counts}")

    results_by_transition = {}
    for name, Fg, Fe, gFg, gFe in TRANSITIONS:
        print(f"\n########## Transition {name} "
              f"(Fg={Fg}, Fe={Fe}) ##########", flush=True)
        obe = setup_obe(Fg=Fg, Fe=Fe, gFg=gFg, gFe=gFe)
        state_dim = len(obe.rho0) + 6

        print("Warming up JIT...", flush=True)
        warmup(obe)

        runs = {}
        for t_factor in SWEEP_T_FACTORS:
            print(f"\n=== {name}: t = 2pi x {t_factor} ===", flush=True)
            runs[t_factor] = run_cpu_sweep(
                obe, SWEEP_CPU_ATOMS, core_counts, t_factor,
                transition=(Fg, Fe, gFg, gFe),
            )

        results_by_transition[name] = {
            'transition': {'Fg': Fg, 'Fe': Fe, 'gFg': gFg, 'gFe': gFe},
            'state_dim': state_dim,
            'atom_counts': list(SWEEP_CPU_ATOMS),
            'runs': runs,
        }

    payload = {
        'meta': {
            'timestamp': datetime.datetime.now().isoformat(),
            't_factors': list(SWEEP_T_FACTORS),
            'core_counts': core_counts,
            'max_steps': N_POINTS,
            'seed': SEED,
            'cpu_cores_available': os.cpu_count(),
        },
        'transitions': results_by_transition,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
