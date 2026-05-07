"""Combine cpu_results.pkl + gpu_results.pkl into plots and Amdahl tables.

Expects the multi-transition schema:
    payload['transitions'][name]['runs'][t_factor][n] -> {...}

Usage:
    python analyze.py [--cpu cpu_results.pkl] [--gpu gpu_results.pkl] [--out OUTDIR]
"""
import os
import sys
import pickle
import argparse
import datetime

import numpy as np

from common import (
    AMDAHL_ATOMS_PER_WORKER, AMDAHL_CORE_COUNTS, PARALLEL_CORE_COUNTS, N_SERIAL,
    amdahl_speedup, fit_amdahl_p,
)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_gpu(path_or_dir):
    """Load GPU results.

    Accepts either a single pickle path (combined run) or a directory
    containing per-transition pickles (gpu_results_<NAME>.pkl). Returns a
    payload with a merged 'transitions' dict; meta is taken from the first
    transition pickle found and its t_factors expanded to the union.
    """
    import glob as _glob
    if path_or_dir is None:
        return None
    if os.path.isfile(path_or_dir):
        return load(path_or_dir)
    if os.path.isdir(path_or_dir):
        search_dir = path_or_dir
    else:
        # Treat as a base path; look in its parent for split pickles.
        search_dir = os.path.dirname(path_or_dir) or '.'

    # Single-file fallback first.
    if os.path.isfile(path_or_dir):
        return load(path_or_dir)

    parts = sorted(_glob.glob(os.path.join(search_dir, 'gpu_results_*.pkl')))
    if not parts:
        return None

    merged_transitions = {}
    meta = None
    all_t_factors = set()
    for p in parts:
        payload = load(p)
        if payload is None:
            continue
        if meta is None:
            meta = dict(payload['meta'])
        all_t_factors.update(payload['meta'].get('t_factors', []))
        for name, tdata in payload.get('transitions', {}).items():
            merged_transitions[name] = tdata

    if not meta:
        return None
    meta['t_factors'] = sorted(all_t_factors)
    meta['source_files'] = parts
    return {'meta': meta, 'transitions': merged_transitions}


def plot_sweep(cpu_runs, gpu_runs, name, state_dim, t_factor, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))

    if cpu_runs:
        ns = sorted(cpu_runs.keys())
        serial_pts = [(n, cpu_runs[n]['serial']) for n in ns
                      if cpu_runs[n].get('serial') is not None]
        if serial_pts:
            ax.plot(*zip(*serial_pts), 'o-', label='Serial CPU')

        all_cores = sorted({c for n in ns for c in cpu_runs[n].get('parallel', {})})
        markers = ['^', 'v', 'D', 'p', 'h', '*']
        for i, nc in enumerate(all_cores):
            pts = [(n, cpu_runs[n]['parallel'][nc]) for n in ns
                   if nc in cpu_runs[n].get('parallel', {})]
            if pts:
                ax.plot(*zip(*pts), f'{markers[i % len(markers)]}-',
                        label=f'Parallel CPU ({nc} cores)')

    if gpu_runs:
        ns = sorted(gpu_runs.keys())
        gpu_pts = [(n, gpu_runs[n]['gpu']) for n in ns
                   if gpu_runs[n].get('gpu') is not None]
        if gpu_pts:
            ax.plot(*zip(*gpu_pts), 's-', label='GPU batched')

    ax.set_xlabel('Number of atoms (N)')
    ax.set_ylabel('Time per atom (s)')
    ax.set_title(f'Evolve Motion — {name} (state_dim={state_dim})  '
                 f't=2\u03c0\u00d7{t_factor}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, f'sweep_{name}_t{t_factor}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {out}")


def plot_amdahl(amdahl_sweep, name, t_factor, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not amdahl_sweep:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    core_range = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(amdahl_sweep)))

    for (apw, p, speedups), color in zip(amdahl_sweep, colors):
        predicted = [amdahl_speedup(p, n) for n in core_range]
        ax1.plot(core_range, predicted, '-', color=color, lw=2,
                 label=f'{apw} atoms/worker (p={p:.3f})')
        for nc, s in speedups.items():
            ax1.plot(nc, s, 'o', color=color, markersize=7, zorder=5)

    ax1.set_xlabel('Number of cores'); ax1.set_ylabel('Speedup S(n)')
    ax1.set_title(f"Amdahl — {name}  t=2\u03c0\u00d7{t_factor}")
    ax1.set_xscale('log', base=2)
    ax1.legend(fontsize=9); ax1.grid(True, which='both', ls='--', alpha=0.4)

    apws = [r[0] for r in amdahl_sweep]
    ps = [r[1] for r in amdahl_sweep]
    ax2.plot(apws, [p * 100 for p in ps], 'o-', color='tab:blue',
             markersize=8, lw=2)
    ax2.set_xlabel('Atoms per worker'); ax2.set_ylabel('p (%)')
    ax2.set_title('Overhead vs work per worker')
    ax2.set_xscale('log', base=2); ax2.set_ylim(0, 105)
    ax2.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()
    out = os.path.join(out_dir, f'amdahl_{name}_t{t_factor}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {out}")


def numerical_check(cpu_runs, gpu_runs):
    if not cpu_runs or not gpu_runs:
        return
    candidates = [n for n in sorted(set(cpu_runs) & set(gpu_runs))
                  if cpu_runs[n].get('z_serial') is not None
                  and gpu_runs[n].get('z_gpu') is not None]
    if not candidates:
        return
    n = candidates[0]
    z_s = cpu_runs[n]['z_serial']; z_g = gpu_runs[n]['z_gpu']
    k = min(N_SERIAL, len(z_s), len(z_g))
    diff = float(np.max(np.abs(z_s[:k] - z_g[:k])))
    print(f"    Numerical check (N={n}, first {k}): "
          f"GPU max|z diff| = {diff:.4e}")


def amdahl_tables(cpu_runs, fit_cores):
    candidates = [
        n for n in sorted(cpu_runs)
        if cpu_runs[n].get('serial') is not None
        and all(c in cpu_runs[n].get('parallel', {}) for c in fit_cores)
    ]
    if candidates:
        n_fit = candidates[-1]
        t_ser = cpu_runs[n_fit]['serial']
        pa = {c: cpu_runs[n_fit]['parallel'][c] for c in fit_cores}
        p, speedups = fit_amdahl_p(t_ser, pa)
        if p is not None:
            print(f"    Amdahl fit (N={n_fit}):  p = {p:.4f}  "
                  f"({p*100:.1f}% parallel)")
            print(f"    {'Cores':>6}  {'S_pred':>8}  {'t/atom(s)':>11}")
            seen = set()
            for n in AMDAHL_CORE_COUNTS:
                if n in seen:
                    continue
                seen.add(n)
                s_pred = amdahl_speedup(p, n)
                tag = (f"  ← measured S={speedups[n]:.2f}x"
                       if n in speedups else "")
                print(f"    {n:>6}  {s_pred:>8.2f}  "
                      f"{t_ser/s_pred:>11.3f}{tag}")

    max_cores = max(fit_cores)
    amdahl_sweep = []
    for apw in AMDAHL_ATOMS_PER_WORKER:
        n_total = apw * max_cores
        if n_total not in cpu_runs:
            continue
        e = cpu_runs[n_total]
        if e.get('serial') is None:
            continue
        if not all(c in e.get('parallel', {}) for c in fit_cores):
            continue
        pa = {c: e['parallel'][c] for c in fit_cores}
        p, speedups = fit_amdahl_p(e['serial'], pa)
        if p is not None:
            amdahl_sweep.append((apw, p, speedups))
    return amdahl_sweep


def plot_state_dim_comparison(gpu_data, out_dir):
    """One curve per transition on a single plot, showing GPU scaling."""
    if not gpu_data:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_factors = gpu_data['meta']['t_factors']
    for t_factor in t_factors:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for name, tdata in gpu_data['transitions'].items():
            runs = tdata['runs'].get(t_factor, {})
            pts = [(n, runs[n]['gpu']) for n in sorted(runs)
                   if runs[n].get('gpu') is not None]
            if pts:
                sd = tdata['state_dim']
                ax.plot(*zip(*pts), 'o-',
                        label=f'{name} (state_dim={sd})')
        ax.set_xlabel('Number of atoms (N)')
        ax.set_ylabel('GPU time per atom (s)')
        ax.set_title(f'GPU scaling across transitions  t=2\u03c0\u00d7{t_factor}')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4)
        plt.tight_layout()
        out = os.path.join(out_dir, f'gpu_transitions_t{t_factor}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpu', default=os.path.join(here, 'cpu_results.pkl'))
    ap.add_argument('--gpu', default=here,
                    help="Path to gpu_results.pkl OR a directory containing "
                         "per-transition gpu_results_<name>.pkl files. "
                         "Defaults to the script directory.")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cpu = load(args.cpu)
    # GPU input is either a single combined pickle or a directory of
    # per-transition pickles (the new default produced by benchmark_gpu.sh).
    if os.path.isdir(args.gpu):
        single = os.path.join(args.gpu, 'gpu_results.pkl')
        gpu = load(single) if os.path.exists(single) else load_gpu(args.gpu)
    else:
        gpu = load_gpu(args.gpu)
    if cpu is None and gpu is None:
        print("No input data found.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out or os.path.join(
        here, f'run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output → {out_dir}")

    if cpu:
        print(f"CPU data:  {args.cpu}  ({cpu['meta']['timestamp']})")
    if gpu:
        print(f"GPU data:  {args.gpu}  ({gpu['meta']['timestamp']})")

    fit_cores = (cpu['meta']['core_counts']
                 if cpu else list(PARALLEL_CORE_COUNTS))

    all_names = sorted(set((cpu['transitions'].keys() if cpu else set()))
                       | set((gpu['transitions'].keys() if gpu else set())))
    t_factors = sorted(set((cpu['meta']['t_factors'] if cpu else []))
                       | set((gpu['meta']['t_factors'] if gpu else [])))

    for name in all_names:
        cpu_t = cpu['transitions'].get(name) if cpu else None
        gpu_t = gpu['transitions'].get(name) if gpu else None
        state_dim = ((gpu_t or cpu_t) or {}).get('state_dim', '?')
        optimal = gpu_t['optimal_batch_size'] if gpu_t else None

        print(f"\n######## {name}  "
              f"(state_dim={state_dim}, optimal={optimal}) ########")

        for t_factor in t_factors:
            print(f"\n  === t = 2pi x {t_factor} ===")
            cpu_runs = cpu_t['runs'].get(t_factor) if cpu_t else None
            gpu_runs = gpu_t['runs'].get(t_factor) if gpu_t else None

            numerical_check(cpu_runs, gpu_runs)
            plot_sweep(cpu_runs, gpu_runs, name, state_dim, t_factor, out_dir)

            if cpu_runs:
                amdahl_sweep = amdahl_tables(cpu_runs, fit_cores)
                plot_amdahl(amdahl_sweep, name, t_factor, out_dir)

    # Cross-transition GPU comparison.
    plot_state_dim_comparison(gpu, out_dir)

    print(f"\nDone. Output in {out_dir}")


if __name__ == '__main__':
    main()
