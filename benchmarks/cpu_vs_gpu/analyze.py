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
    PARALLEL_CORE_COUNTS, N_SERIAL,
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


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_sweep(cpu_runs, gpu_runs, name, state_dim, t_factor, out_dir):
    """One sweep plot per (transition, t_factor)."""
    plt = _mpl()
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
    ax.set_title(f'Evolve motion — {name} (state_dim={state_dim})  '
                 f't=2\u03c0\u00d7{t_factor}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.legend(fontsize=9); ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, f'sweep_{name}_t{t_factor}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved {out}")


def plot_amdahl_combined(fits, out_dir):
    """Single Amdahl plot: all (transition, t_factor) fits overlaid.

    fits: list of (name, state_dim, t_factor, p, speedups).
    color = transition, linestyle = t_factor.
    """
    if not fits:
        return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 6))
    core_range = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])

    names = sorted({f[0] for f in fits})
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    color_for = dict(zip(names, cmap))
    linestyles = {100: ':', 500: '--', 2000: '-'}

    for name, state_dim, t_factor, p, speedups in fits:
        color = color_for[name]
        ls = linestyles.get(t_factor, '-')
        predicted = [amdahl_speedup(p, n) for n in core_range]
        ax.plot(core_range, predicted, ls, color=color, lw=2,
                label=f'{name} (dim={state_dim})  t=2π×{t_factor}  p={p:.2f}')
        for nc, s in speedups.items():
            ax.plot(nc, s, 'o', color=color, markersize=5, zorder=5)

    ax.set_xlabel('Number of cores'); ax.set_ylabel('Speedup S(n)')
    ax.set_title('Amdahl fits across transitions and t_factors')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=8, ncol=2); ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, 'amdahl_all.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved {out}")


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


def _amdahl_at_best_n(cpu_runs, fit_cores, t_factor):
    """Fit Amdahl p at the largest N with all fit_cores measured. Prints summary."""
    candidates = [
        n for n in sorted(cpu_runs)
        if cpu_runs[n].get('serial') is not None
        and all(c in cpu_runs[n].get('parallel', {}) for c in fit_cores)
    ]
    if not candidates:
        return None
    n_fit = candidates[-1]
    t_ser = cpu_runs[n_fit]['serial']
    pa = {c: cpu_runs[n_fit]['parallel'][c] for c in fit_cores}
    p, speedups = fit_amdahl_p(t_ser, pa)
    if p is None:
        return None
    print(f"    t=2π×{t_factor}  Amdahl fit (N={n_fit}):  p = {p:.4f}  "
          f"({p*100:.1f}% parallel)")
    return (p, speedups)


def plot_state_dim_comparison(gpu_data, out_dir):
    """All GPU curves on one plot: color=transition, linestyle=t_factor."""
    if not gpu_data:
        return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    linestyles = {100: ':', 500: '--', 2000: '-'}
    names = list(gpu_data['transitions'].keys())
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))

    for name, color in zip(names, cmap):
        tdata = gpu_data['transitions'][name]
        sd = tdata['state_dim']
        for t_factor, ls in sorted(linestyles.items()):
            runs = tdata['runs'].get(t_factor, {})
            pts = [(n, runs[n]['gpu']) for n in sorted(runs)
                   if runs[n].get('gpu') is not None]
            if pts:
                ax.plot(*zip(*pts), ls, marker='o', color=color, markersize=4,
                        label=f'{name} (dim={sd})  t=2π×{t_factor}')

    ax.set_xlabel('Number of atoms (N)')
    ax.set_ylabel('GPU time per atom (s)')
    ax.set_title('GPU scaling across transitions')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=False)
    plt.tight_layout()
    out = os.path.join(out_dir, 'gpu_transitions.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
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

    all_fits = []
    for name in all_names:
        cpu_t = cpu['transitions'].get(name) if cpu else None
        gpu_t = gpu['transitions'].get(name) if gpu else None
        state_dim = ((gpu_t or cpu_t) or {}).get('state_dim', '?')
        optimal = gpu_t['optimal_batch_size'] if gpu_t else None

        print(f"\n######## {name}  "
              f"(state_dim={state_dim}, optimal={optimal}) ########")

        for t_factor in t_factors:
            cpu_runs = cpu_t['runs'].get(t_factor) if cpu_t else None
            gpu_runs = gpu_t['runs'].get(t_factor) if gpu_t else None
            numerical_check(cpu_runs, gpu_runs)
            plot_sweep(cpu_runs, gpu_runs, name, state_dim, t_factor, out_dir)
            if cpu_runs:
                fit = _amdahl_at_best_n(cpu_runs, fit_cores, t_factor)
                if fit is not None:
                    p, speedups = fit
                    all_fits.append((name, state_dim, t_factor, p, speedups))

    plot_amdahl_combined(all_fits, out_dir)
    plot_state_dim_comparison(gpu, out_dir)

    print(f"\nDone. Output in {out_dir}")


if __name__ == '__main__':
    main()
