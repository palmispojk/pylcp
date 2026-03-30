"""
GPU atom throughput benchmark.

Measures wall-clock time vs atom count for blue MOT (dim=22) and
green MOT (dim=62). Identifies scaling regimes and saves results
to log, JSON, and plots.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import time
import platform
import subprocess
import json
from datetime import datetime

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pylcp

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
#  Hardware info
# ---------------------------------------------------------------------------

def get_hardware_info():
    info = {
        'timestamp': datetime.now().isoformat(),
        'jax_backend': jax.default_backend(),
        'jax_version': jax.__version__,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
    }

    gpu_devs = [d for d in jax.devices() if d.platform == 'gpu']
    info['n_gpus'] = len(gpu_devs)
    if gpu_devs:
        d = gpu_devs[0]
        info['gpu_name'] = getattr(d, 'device_kind', 'unknown')
        try:
            mem = d.memory_stats()
            info['gpu_memory_bytes'] = mem.get('bytes_limit', 0)
            info['gpu_memory_gb'] = info['gpu_memory_bytes'] / 2**30
        except Exception:
            info['gpu_memory_gb'] = 0

    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap',
             '--format=csv,noheader,nounits'], text=True, timeout=5).strip()
        parts = [p.strip() for p in out.split(',')]
        info['gpu_name_smi'] = parts[0]
        info['gpu_memory_mb_smi'] = int(parts[1])
        info['driver_version'] = parts[2]
        info['compute_capability'] = parts[3]
    except Exception:
        pass

    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if 'model name' in line:
                    info['cpu_name'] = line.split(':')[1].strip()
                    break
        info['cpu_count'] = os.cpu_count()
    except Exception:
        info['cpu_name'] = platform.processor()
        info['cpu_count'] = os.cpu_count()

    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if 'MemTotal' in line:
                    info['ram_gb'] = int(line.split()[1]) / 2**20
                    break
    except Exception:
        pass

    return info


def format_hardware(info):
    lines = [
        "=== Hardware ===",
        f"  GPU:     {info.get('gpu_name_smi', info.get('gpu_name', 'unknown'))}",
        f"  VRAM:    {info.get('gpu_memory_mb_smi', 0)} MB",
        f"  Driver:  {info.get('driver_version', 'unknown')}",
        f"  Compute: {info.get('compute_capability', 'unknown')}",
        f"  CPU:     {info.get('cpu_name', 'unknown')}",
        f"  Cores:   {info.get('cpu_count', 'unknown')}",
        f"  RAM:     {info.get('ram_gb', 0):.1f} GB",
        f"  JAX:     {info.get('jax_version', 'unknown')} ({info.get('jax_backend', 'unknown')})",
        f"  Python:  {info.get('python_version', 'unknown')}",
        f"  Date:    {info.get('timestamp', 'unknown')}",
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
#  Problem definitions
# ---------------------------------------------------------------------------

def build_blue_mot():
    import scipy.constants as const
    wavelength = 460.862e-9
    gamma_real = 2 * np.pi * 30.5e6
    kmag_real = 2 * np.pi / wavelength
    muB_real = const.physical_constants["Bohr magneton"][0]
    mass_real = const.value('atomic mass constant') * 88
    gamma, kmag, muB = 1, 1, 1
    mass = mass_real * gamma_real / const.hbar / kmag_real**2
    det = -1.31 * gamma
    s = 0.173
    alpha_nat = 0.37 * muB_real / (gamma_real * kmag_real * const.hbar)

    laserBeams = pylcp.conventional3DMOTBeams(k=kmag, s=s, delta=0., beam_type=pylcp.infinitePlaneWaveBeam)
    magField = pylcp.quadrupoleMagneticField(alpha_nat)
    H_g, muq_g = pylcp.hamiltonians.singleF(F=0, gF=0, muB=muB)
    H_e, muq_e = pylcp.hamiltonians.singleF(F=1, gF=1, muB=muB)
    d_q = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)
    hamiltonian = pylcp.hamiltonian(
        H_g, -det * np.eye(3) + H_e, muq_g, muq_e, d_q,
        mass=mass, muB=muB, gamma=gamma, k=kmag)
    obe = pylcp.obe(laserBeams, magField, hamiltonian, transform_into_re_im=True)
    state_dim = hamiltonian.n**2 + 6
    return obe, state_dim, np.array([1000., 1000., 1000.]), np.array([0.1, 0.1, 0.3])


def build_green_mot():
    import scipy.constants as const
    frq_real = 603976506.6e6 * 2 * np.pi
    gamma_real = 61.4e6
    kmag_real = frq_real / const.c
    muB_real = const.physical_constants["Bohr magneton"][0]
    mass_real = const.value('atomic mass constant') * 88
    gamma, kmag, muB = 1, 1, 1
    mass = mass_real * gamma_real / const.hbar / kmag_real**2
    alpha_real = 0.4
    alpha = alpha_real * muB_real / (gamma_real * kmag_real * const.hbar)
    det = -2.1 * gamma
    s = 2

    laserBeams = pylcp.conventional3DMOTBeams(k=kmag, s=s, delta=0., beam_type=pylcp.infinitePlaneWaveBeam)
    magField = pylcp.quadrupoleMagneticField(alpha)
    H_g, muq_g = pylcp.hamiltonians.singleF(F=2, gF=1.5, muB=muB)
    H_e, muq_e = pylcp.hamiltonians.singleF(F=3, gF=1 + 1/3, muB=muB)
    d_q = pylcp.hamiltonians.dqij_two_bare_hyperfine(2, 3)
    hamiltonian = pylcp.hamiltonian(
        H_g, -det * np.eye(7) + H_e, muq_g, muq_e, d_q,
        mass=mass, muB=muB, gamma=gamma, k=kmag)
    obe = pylcp.obe(laserBeams, magField, hamiltonian, transform_into_re_im=True)
    state_dim = hamiltonian.n**2 + 6
    return obe, state_dim, np.array([2, 2, 2]) / alpha, np.array([0.1, 0.1, 0.1])


# ---------------------------------------------------------------------------
#  Benchmark runner
# ---------------------------------------------------------------------------

def make_initial_conditions(obe, N, rscale, vscale, seed=42):
    rng = np.random.default_rng(seed)
    r0 = rscale[None, :] * rng.standard_normal((N, 3))
    v0 = vscale[None, :] * rng.standard_normal((N, 3))
    rho0_all = []
    for i in range(N):
        obe.set_initial_position(r0[i])
        obe.set_initial_velocity(v0[i])
        obe.set_initial_rho_from_rateeq()
        rho0_all.append(obe.rho0)
    y0 = jnp.array(np.concatenate([np.stack(rho0_all), v0, r0], axis=1))
    keys = jax.random.split(jax.random.PRNGKey(seed), N)
    return y0, keys


def run_scaling(obe, state_dim, rscale, vscale, label, atom_counts,
                tmax=2e3, n_output=50):
    header = f"\n{'='*70}\n  {label}   (state_dim={state_dim})\n{'='*70}"
    col = f"{'N':>8s}  {'total':>8s}  {'per atom':>10s}  {'speedup':>8s}  {'success':>8s}"
    div = f"{'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}"

    lines = [header, col, div]
    for l in lines:
        print(l)

    results = []
    prev_pa = None

    for N in atom_counts:
        try:
            y0, keys = make_initial_conditions(obe, N, rscale, vscale)
            t0 = time.monotonic()
            sols = obe.evolve_motion(
                [0, tmax], y0_batch=y0, keys_batch=keys,
                random_recoil=True, max_scatter_probability=0.5,
                n_output=n_output, progress=False)
            t_total = time.monotonic() - t0
            n_ok = sum(1 for s in sols if s.success)
        except Exception as e:
            row = f"{N:>8d}  OOM -- {str(e)[:50]}"
            lines.append(row)
            print(row)
            break

        pa = t_total / N
        sp = prev_pa / pa if prev_pa and pa > 0 else 1.0
        prev_pa = pa

        row = f"{N:>8d}  {t_total:>7.1f}s  {pa:>9.4f}s  {sp:>7.2f}x  {n_ok:>5d}/{N}"
        lines.append(row)
        print(row)
        results.append({'N': N, 'total': t_total, 'per_atom': pa,
                        'speedup': sp, 'success': n_ok})

    # Analysis
    if len(results) >= 3:
        per_atoms = [r['per_atom'] for r in results]
        best = min(per_atoms)
        best_n = results[per_atoms.index(best)]['N']
        analysis = [f"\n  Analysis:",
                    f"    Best efficiency: {best:.4f} s/atom at N={best_n}"]
        for i, r in enumerate(results):
            if i > 0 and r['speedup'] < 1.5:
                analysis.append(
                    f"    GPU saturates around N={results[i-1]['N']}-{r['N']} "
                    f"(speedup drops to {r['speedup']:.2f}x)")
                break
        for l in analysis:
            lines.append(l)
            print(l)

    return results, '\n'.join(lines)


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def make_plots(all_results, hw_info):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    gpu = hw_info.get('gpu_name_smi', hw_info.get('gpu_name', 'GPU'))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name, res in all_results.items():
        Ns = [r['N'] for r in res]
        ax1.loglog(Ns, [r['total'] for r in res], 'o-', label=name, ms=5)
        ax2.loglog(Ns, [r['per_atom'] for r in res], 's-', label=name, ms=5)

    # Linear reference
    if all_results:
        first = list(all_results.values())[0]
        Ns = np.array([r['N'] for r in first])
        ref = first[-1]['total'] * Ns / first[-1]['N']
        ax1.loglog(Ns, ref, '--', color='gray', alpha=0.5, label='linear')

    ax1.set_xlabel('Number of atoms')
    ax1.set_ylabel('Total time (s)')
    ax1.set_title(f'Total time vs atom count\n({gpu})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Number of atoms')
    ax2.set_ylabel('Time per atom (s)')
    ax2.set_title(f'Per-atom efficiency vs batch size\n({gpu})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'atom_throughput.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

hw = get_hardware_info()
hw_str = format_hardware(hw)
print(hw_str)

atom_counts = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

all_results = {}
all_logs = [hw_str, ""]

res, log = run_scaling(*build_blue_mot(), "Blue MOT (F=0->1, dim=22)", atom_counts)
all_results['Blue MOT (dim=22)'] = res
all_logs.append(log)

res, log = run_scaling(*build_green_mot(), "Green MOT (F=2->3, dim=62)", atom_counts)
all_results['Green MOT (dim=62)'] = res
all_logs.append(log)

# Save log
log_path = os.path.join(OUT_DIR, 'results.txt')
with open(log_path, 'w') as f:
    f.write('\n'.join(all_logs) + '\n')
print(f"\nLog saved to {log_path}")

# Save JSON
json_path = os.path.join(OUT_DIR, 'results.json')
with open(json_path, 'w') as f:
    json.dump({'hardware': hw, 'results': all_results}, f, indent=2, default=str)
print(f"JSON saved to {json_path}")

make_plots(all_results, hw)
