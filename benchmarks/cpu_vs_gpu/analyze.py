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


def plot_sweep(cpu_runs, gpu_runs, name, state_dim, t_factor, out_dir,
               knee=None):
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

    if knee:
        ax.axvline(knee, color='gray', ls=':', lw=1,
                   label=f'GPU saturation knee (N={knee})')

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


def fit_state_dim_power_law(gpu_data):
    """Fit t/atom ~ state_dim^k per (N, t_factor) across transitions.

    Exponent ~2 → memory-streaming dominated (rho-vector loads/stores, O(D^2)).
    Exponent ~3 → dense matmul / Liouvillian apply dominated (O(D^3)).
    Mixed values land between.

    Returns list of dicts: {t_factor, n, points[(state_dim, t)], k, log_a}.
    """
    if not gpu_data:
        return []
    transitions = gpu_data['transitions']
    by_key = {}  # (t_factor, n) -> [(state_dim, t_per_atom), ...]
    for name, tdata in transitions.items():
        sd = tdata['state_dim']
        for t_factor, runs in tdata['runs'].items():
            for n, r in runs.items():
                t = r.get('gpu')
                if t is None:
                    continue
                by_key.setdefault((t_factor, n), []).append((sd, t))
    fits = []
    for (t_factor, n), pts in sorted(by_key.items()):
        if len(pts) < 2:
            continue
        sds = np.array([p[0] for p in pts], dtype=float)
        ts = np.array([p[1] for p in pts], dtype=float)
        # log-log linear fit: log t = k * log D + log a
        x = np.log(sds); y = np.log(ts)
        k, log_a = np.polyfit(x, y, 1)
        fits.append({'t_factor': t_factor, 'n': n,
                     'points': pts, 'k': float(k), 'log_a': float(log_a)})
    return fits


def plot_power_law_fits(fits, out_dir):
    """One log-log plot of t/atom vs state_dim with fitted lines."""
    if not fits:
        return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(fits)))
    for color, fit in zip(cmap, fits):
        sds = np.array([p[0] for p in fit['points']], dtype=float)
        ts = np.array([p[1] for p in fit['points']], dtype=float)
        order = np.argsort(sds)
        sds, ts = sds[order], ts[order]
        ax.plot(sds, ts, 'o', color=color, markersize=5)
        sd_grid = np.linspace(sds.min(), sds.max(), 50)
        t_fit = np.exp(fit['log_a']) * sd_grid ** fit['k']
        ax.plot(sd_grid, t_fit, '-', color=color, lw=1.5,
                label=f"N={fit['n']}, t=2π×{fit['t_factor']}: k={fit['k']:.2f}")
    ax.set_xlabel('state_dim D')
    ax.set_ylabel('GPU time per atom (s)')
    ax.set_title('Power-law fit  t/atom ∝ D^k\n(k≈2 memory-bound, k≈3 compute-bound)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=False)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, 'power_law_state_dim.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved {out}")


def plot_arithmetic_intensity(gpu_data, out_dir):
    """AI per (transition, N, t_factor) vs N, with knee marker per transition."""
    if not gpu_data:
        return
    has_any = any(
        r.get('cost', {}).get('ai') is not None
        for tdata in gpu_data['transitions'].values()
        for runs in tdata['runs'].values()
        for r in runs.values()
    )
    if not has_any:
        return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    linestyles = {100: ':', 500: '--', 2000: '-'}
    names = list(gpu_data['transitions'].keys())
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    for name, color in zip(names, cmap):
        tdata = gpu_data['transitions'][name]
        sd = tdata['state_dim']
        knee = tdata.get('optimal_batch_size')
        for t_factor, ls in sorted(linestyles.items()):
            runs = tdata['runs'].get(t_factor, {})
            pts = [(n, runs[n]['cost']['ai']) for n in sorted(runs)
                   if runs[n].get('cost', {}).get('ai') is not None]
            if pts:
                ax.plot(*zip(*pts), ls, marker='o', color=color, markersize=4,
                        label=f'{name} (D={sd}) t=2π×{t_factor}')
        if knee:
            ax.axvline(knee, color=color, alpha=0.25, lw=1)
    ax.set_xlabel('Number of atoms (N)')
    ax.set_ylabel('Arithmetic intensity (FLOPs / byte)')
    ax.set_title('Kernel arithmetic intensity vs batch size\n'
                 '(vertical lines = saturation knee per transition)')
    ax.set_xscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=False)
    plt.tight_layout()
    out = os.path.join(out_dir, 'arithmetic_intensity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved {out}")


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


def _cpu_best_at(cpu_runs):
    """(t_best, label, n) over all (nc, N) in cpu_runs; label is 'serial' or 'X workers'."""
    if not cpu_runs:
        return None
    best = None
    for n, r in cpu_runs.items():
        t_ser = r.get('serial')
        if t_ser is not None:
            if best is None or t_ser < best[0]:
                best = (t_ser, 'serial', n)
        for nc, t in r.get('parallel', {}).items():
            if best is None or t < best[0]:
                best = (t, f'{nc} workers', n)
    return best


def _gpu_best_at(gpu_runs):
    """(t_best, N) over all N in gpu_runs."""
    if not gpu_runs:
        return None
    best = None
    for n, r in gpu_runs.items():
        t = r.get('gpu')
        if t is None:
            continue
        if best is None or t < best[0]:
            best = (t, n)
    return best


def write_summary(cpu, gpu, all_fits, power_fits, out_path):
    """Write a human-readable summary.txt with speedup tables and stats."""
    lines = []
    w = lines.append

    w("=" * 72)
    w(f"Benchmark summary  generated {datetime.datetime.now().isoformat()}")
    w("=" * 72)
    if cpu:
        w(f"CPU data timestamp: {cpu['meta']['timestamp']}")
        w(f"  cores swept:      {cpu['meta']['core_counts']}")
        w(f"  t_factors:        {cpu['meta']['t_factors']}")
    if gpu:
        w(f"GPU data timestamp: {gpu['meta']['timestamp']}")
        w(f"  t_factors:        {gpu['meta']['t_factors']}")
    w("")

    names = sorted(set((cpu['transitions'].keys() if cpu else set()))
                   | set((gpu['transitions'].keys() if gpu else set())))
    t_factors = sorted(set((cpu['meta']['t_factors'] if cpu else []))
                       | set((gpu['meta']['t_factors'] if gpu else [])))

    # ---- Headline table: CPU best vs GPU best per (transition, t_factor) ----
    if cpu and gpu:
        w("-" * 72)
        w("End-to-end speedup: GPU (best batch) vs best parallel CPU (any N, nc)")
        w("-" * 72)
        w(f"{'Transition':<14}{'t':<16}"
          f"{'t_CPU_best':>14}{'cfg':>15}"
          f"{'t_GPU':>12}{'N_GPU':>10}{'Speedup':>10}")
        for name in names:
            cpu_t = cpu['transitions'].get(name)
            gpu_t = gpu['transitions'].get(name)
            for tf in t_factors:
                c_runs = cpu_t['runs'].get(tf) if cpu_t else None
                g_runs = gpu_t['runs'].get(tf) if gpu_t else None
                cb = _cpu_best_at(c_runs)
                gb = _gpu_best_at(g_runs)
                if not cb or not gb:
                    continue
                t_cpu, cfg, n_cpu = cb
                t_gpu, n_gpu = gb
                sp = t_cpu / t_gpu
                w(f"{name:<14}2π×{tf:<12}"
                  f"{t_cpu:>12.5f}  {cfg:>13}"
                  f"{t_gpu:>12.5f}{n_gpu:>10}{sp:>9.1f}×")
        w("")

    # ---- Per-transition detail ----
    for name in names:
        cpu_t = cpu['transitions'].get(name) if cpu else None
        gpu_t = gpu['transitions'].get(name) if gpu else None
        sd = ((gpu_t or cpu_t) or {}).get('state_dim', '?')
        opt = gpu_t['optimal_batch_size'] if gpu_t else None
        w("-" * 72)
        w(f"### {name}   state_dim={sd}   optimal GPU batch={opt}")
        w("-" * 72)

        # Amdahl p per t_factor
        fits_here = [f for f in all_fits if f[0] == name]
        if fits_here:
            w("  Amdahl fit (ref N=largest with serial + all cores):")
            for _, _, tf, p, _ in fits_here:
                w(f"    t=2π×{tf:<5}  p = {p:.3f}  ({p*100:5.1f}% parallel)")
            w("")

        # CPU parallel speedup table at ref N for the largest t_factor
        if cpu_t:
            w("  CPU parallel speedup at reference N (serial → best workers):")
            for tf in t_factors:
                runs = cpu_t['runs'].get(tf, {})
                cores_used = sorted({c for n in runs
                                     for c in runs[n].get('parallel', {})})
                fit_ns = [n for n in sorted(runs)
                          if runs[n].get('serial') is not None
                          and all(c in runs[n].get('parallel', {}) for c in cores_used)]
                if not fit_ns:
                    continue
                n_ref = fit_ns[-1]
                r = runs[n_ref]
                t_ser = r['serial']
                parts = [f"serial={t_ser:.4f}"]
                for nc in cores_used:
                    t_pa = r['parallel'].get(nc)
                    if t_pa is None:
                        continue
                    parts.append(f"{nc}c:{t_ser/t_pa:>4.2f}×")
                w(f"    t=2π×{tf:<5}  N={n_ref:<5}  {'  '.join(parts)}")
            w("")

        # Throughput at peak (atoms/s)
        if cpu_t or gpu_t:
            w("  Peak throughput (atoms/s):")
            for tf in t_factors:
                c_runs = cpu_t['runs'].get(tf, {}) if cpu_t else None
                g_runs = gpu_t['runs'].get(tf, {}) if gpu_t else None
                cb = _cpu_best_at(c_runs) if c_runs else None
                gb = _gpu_best_at(g_runs) if g_runs else None
                parts = []
                if cb:
                    t, cfg, n = cb
                    parts.append(f"CPU {1/t:>9.1f} ({cfg}, N={n})")
                if gb:
                    t, n = gb
                    parts.append(f"GPU {1/t:>9.1f} (N={n})")
                if parts:
                    w(f"    t=2π×{tf:<5}  " + "   ".join(parts))
            w("")

        # CPU vs GPU at matched N, broken down by CPU worker count.
        # Table rows = matched N, columns = serial + each nc; cells = CPU/GPU ratio.
        if cpu_t and gpu_t:
            cores_all = sorted({c for tf in t_factors
                                for n, r in cpu_t['runs'].get(tf, {}).items()
                                for c in r.get('parallel', {})})
            header_cols = ['serial'] + [f'{c}c' for c in cores_all]
            w("  CPU/GPU speedup ratio at matched N (CPU_time / GPU_time):")
            for tf in t_factors:
                c_runs = cpu_t['runs'].get(tf, {})
                g_runs = gpu_t['runs'].get(tf, {})
                matched = sorted(set(c_runs) & set(g_runs))
                # filter to Ns where GPU has a value
                rows = []
                for n in matched:
                    t_gpu = g_runs[n].get('gpu')
                    if t_gpu is None:
                        continue
                    ratios = {}
                    t_ser = c_runs[n].get('serial')
                    if t_ser is not None:
                        ratios['serial'] = t_ser / t_gpu
                    for nc, t_pa in c_runs[n].get('parallel', {}).items():
                        ratios[nc] = t_pa / t_gpu
                    rows.append((n, t_gpu, ratios))
                if not rows:
                    continue
                w(f"    t=2π×{tf}")
                head = f"      {'N':>6}  {'GPU(s)':>8}  " + \
                       "  ".join(f"{c:>6}" for c in header_cols)
                w(head)
                for n, t_gpu, ratios in rows:
                    cells = []
                    for col in ['serial'] + cores_all:
                        r = ratios.get(col)
                        cells.append(f"{r:>6.2f}" if r is not None else f"{'—':>6}")
                    w(f"      {n:>6}  {t_gpu:>8.4f}  " + "  ".join(cells))
                w("")
            w("")

    # ---- GPU compute-vs-memory diagnostic --------------------------------
    if gpu:
        w("=" * 72)
        w("GPU bottleneck diagnostic")
        w("  k = exponent in t/atom ∝ D^k  (k≈2 mem-bound, k≈3 compute-bound)")
        w("  AI = arithmetic intensity from XLA cost_analysis (FLOPs / byte)")
        w("  knee = saturation N from optimal_batch_size probe")
        w("=" * 72)

        if power_fits:
            w("  Power-law fit  log(t/atom) = k·log(D) + log(a):")
            w(f"    {'t_factor':>10}  {'N':>8}  {'k':>6}  "
              f"{'a (μs/D^k)':>14}  {'classifier':>18}")
            for fit in sorted(power_fits, key=lambda f: (f['n'], f['t_factor'])):
                k = fit['k']
                if k < 2.3:
                    cls = 'memory-streaming'
                elif k > 2.7:
                    cls = 'compute-bound'
                else:
                    cls = 'mixed'
                a_us = float(np.exp(fit['log_a'])) * 1e6
                w(f"    2π×{fit['t_factor']:<7}{fit['n']:>8}  {k:>6.2f}  "
                  f"{a_us:>14.4g}  {cls:>18}")
            w("")

        # Per-(transition, N, t_factor) AI table.
        w("  Arithmetic intensity per run (FLOPs/byte, kernel-invariant):")
        w(f"    {'Transition':<14}{'D':>4}  {'knee':>6}  {'t_factor':>10}"
          f"  {'N':>8}  {'AI':>10}  {'sat':>6}")
        for name in sorted(gpu['transitions']):
            tdata = gpu['transitions'][name]
            sd = tdata.get('state_dim', '?')
            knee = tdata.get('optimal_batch_size') or 0
            for tf in sorted(tdata['runs']):
                runs = tdata['runs'][tf]
                for n in sorted(runs):
                    cost = runs[n].get('cost') or {}
                    ai = cost.get('ai')
                    if ai is None:
                        continue
                    sat = 'yes' if knee and n >= knee // 2 else 'no'
                    w(f"    {name:<14}{sd:>4}  {knee:>6}  2π×{tf:<7}"
                      f"  {n:>8}  {ai:>10.2f}  {sat:>6}")
        w("")
        w("  How to read this:")
        w("    • k≈2 + AI low (<5)  → bandwidth-limited streaming kernel")
        w("    • k≈3 + AI high (>10) → compute-limited dense matmul")
        w("    • N << knee           → launch/latency limited (GPU not saturated)")
        w("    • N ≥ knee            → hardware-limited; AI/k tell which roof")
        w("")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(here, 'run')
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpu', default=os.path.join(run_dir, 'cpu_results.pkl'))
    ap.add_argument('--gpu', default=run_dir,
                    help="Path to gpu_results.pkl OR a directory containing "
                         "per-transition gpu_results_<name>.pkl files. "
                         "Defaults to the run/ directory.")
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

    out_dir = args.out or run_dir
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
            plot_sweep(cpu_runs, gpu_runs, name, state_dim, t_factor, out_dir,
                       knee=optimal)
            if cpu_runs:
                fit = _amdahl_at_best_n(cpu_runs, fit_cores, t_factor)
                if fit is not None:
                    p, speedups = fit
                    all_fits.append((name, state_dim, t_factor, p, speedups))

    plot_amdahl_combined(all_fits, out_dir)
    plot_state_dim_comparison(gpu, out_dir)

    power_fits = fit_state_dim_power_law(gpu)
    plot_power_law_fits(power_fits, out_dir)
    plot_arithmetic_intensity(gpu, out_dir)

    write_summary(cpu, gpu, all_fits, power_fits,
                  os.path.join(out_dir, 'summary.txt'))

    print(f"\nDone. Output in {out_dir}")


if __name__ == '__main__':
    main()
