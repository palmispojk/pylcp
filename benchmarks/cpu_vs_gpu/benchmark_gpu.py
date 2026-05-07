"""GPU-only benchmark: evolve_motion per-atom times, saved to a pickle.

Sweeps over multiple transitions (state dimensions) and extends atom counts
up to `optimal_batch_size` for each transition.

Output: gpu_results.pkl (next to this file, or --out PATH).
"""
import os
import sys
import time
import pickle
import datetime
import argparse

import numpy as np
import jax
import jax.numpy as jnp

from common import (
    N_POINTS, SEED, SWEEP_GPU_ATOMS, SWEEP_T_FACTORS, TRANSITIONS,
    setup_obe, make_y0_list,
)
from pylcp.integration_tools_gpu import optimal_batch_size


def warmup(obe):
    rho0 = jnp.array(obe.rho0)
    y0 = jnp.concatenate([rho0, jnp.zeros(3), jnp.zeros(3)])
    y0_batch = jnp.stack([y0, y0])
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    obe.evolve_motion([0, 100], n_points=8, y0_batch=y0_batch, keys_batch=keys,
                      freeze_axis=[True, True, False], backend='gpu')


def run_gpu_sweep(obe, atom_counts, t_factor):
    kw = dict(freeze_axis=[True, True, False], backend='gpu')
    t_span = [0, 2 * np.pi * t_factor]
    out = {}
    for n in atom_counts:
        y0_list = make_y0_list(obe, n)
        y0_batch = jnp.stack(y0_list)
        keys = jax.random.split(jax.random.PRNGKey(SEED), n)
        try:
            # Prime JIT for this batch shape.
            obe.evolve_motion(t_span, n_points=N_POINTS,
                              y0_batch=y0_batch[:min(n, 2)],
                              keys_batch=keys[:min(n, 2)], **kw)
            t0 = time.perf_counter()
            obe.evolve_motion(t_span, n_points=N_POINTS,
                              y0_batch=y0_batch, keys_batch=keys, **kw)
            t_per_atom = (time.perf_counter() - t0) / n
            z_gpu = np.array([sol.r[2, -1] for sol in obe.sols])
            out[n] = {'gpu': t_per_atom, 'z_gpu': z_gpu}
            print(f"    N={n:>6}: {t_per_atom:.4f} s/atom", flush=True)
        except Exception as exc:
            print(f"    N={n:>6}: FAILED ({type(exc).__name__}: {exc})",
                  flush=True)
            out[n] = {'gpu': None, 'z_gpu': None, 'error': str(exc)}
    return out


def run_one_transition(name, Fg, Fe, gFg, gFe):
    """Sweep a single transition. Returns the per-transition result dict."""
    print(f"\n########## Transition {name} "
          f"(Fg={Fg}, Fe={Fe}) ##########", flush=True)
    obe = setup_obe(Fg=Fg, Fe=Fe, gFg=gFg, gFe=gFe)
    state_dim = len(obe.rho0) + 6

    print("Warming up JIT...", flush=True)
    warmup(obe)

    n_batched = optimal_batch_size(state_dim, safety=0.6)
    print(f"  state_dim={state_dim}, optimal GPU batch size: {n_batched}")

    atom_counts = sorted(set(SWEEP_GPU_ATOMS))
    if n_batched:
        atom_counts = [n for n in atom_counts if n <= n_batched]
        if atom_counts and n_batched > 2 * atom_counts[-1]:
            mid = int(np.sqrt(atom_counts[-1] * n_batched))
            atom_counts.append(mid)
        atom_counts.append(n_batched)
        atom_counts = sorted(set(atom_counts))

    runs = {}
    for t_factor in SWEEP_T_FACTORS:
        print(f"\n=== {name}: t = 2pi x {t_factor} ===", flush=True)
        runs[t_factor] = run_gpu_sweep(obe, atom_counts, t_factor)

    return {
        'transition': {'Fg': Fg, 'Fe': Fe, 'gFg': gFg, 'gFe': gFe},
        'state_dim': state_dim,
        'optimal_batch_size': n_batched,
        'atom_counts': atom_counts,
        'runs': runs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None,
                    help="Output pickle path. Default depends on --transition: "
                         "with one named transition, gpu_results_<name>.pkl; "
                         "without, gpu_results.pkl.")
    ap.add_argument('--transition', default=None,
                    help="Name of a single transition to run "
                         "(e.g. F0_F1). Omit to run all in one process.")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))

    if args.transition:
        matches = [t for t in TRANSITIONS if t[0] == args.transition]
        if not matches:
            valid = ', '.join(t[0] for t in TRANSITIONS)
            print(f"ERROR: unknown transition '{args.transition}'. "
                  f"Valid: {valid}", file=sys.stderr)
            sys.exit(2)
        wanted = matches
        default_name = f'gpu_results_{args.transition}.pkl'
    else:
        wanted = list(TRANSITIONS)
        default_name = 'gpu_results.pkl'
    out_path = args.out or os.path.join(here, default_name)

    print(f"Run started: {datetime.datetime.now().isoformat()}")
    gpu_devs = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpu_devs:
        print("ERROR: no GPU devices visible to JAX.", file=sys.stderr)
        sys.exit(1)
    for d in gpu_devs:
        mem = d.memory_stats()
        print(f"  GPU: {d}  memory={mem['bytes_limit']/2**30:.1f} GiB")

    results_by_transition = {
        name: run_one_transition(name, Fg, Fe, gFg, gFe)
        for name, Fg, Fe, gFg, gFe in wanted
    }

    payload = {
        'meta': {
            'timestamp': datetime.datetime.now().isoformat(),
            't_factors': list(SWEEP_T_FACTORS),
            'n_points': N_POINTS,
            'seed': SEED,
            'gpu_devices': [str(d) for d in gpu_devs],
            'gpu_memory_bytes': [d.memory_stats()['bytes_limit']
                                 for d in gpu_devs],
        },
        'transitions': results_by_transition,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
