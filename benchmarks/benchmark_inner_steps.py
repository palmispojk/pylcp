"""
Benchmark inner_max_steps effect on GPU round-trip overhead.

Tests multiple atom/transition configurations with varying state_dim to measure
the host-device sync overhead vs compute time tradeoff across problem sizes.

Configurations:
  - F=0 -> F'=1  (Rb87-like, state_dim ~ 16+6 = 22)
  - F=1 -> F'=2  (generic, state_dim ~ 36+6 = 42)
  - F=2 -> F'=3  (Sr88 green MOT, state_dim ~ 144+6 = 150)
"""
import os
if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.94'

import time
import numpy as np
import jax
import jax.numpy as jnp

import pylcp
from pylcp.integration_tools_gpu import optimal_batch_size


def build_problem(F_g, F_e, gF_g, gF_e, det, s, alpha, mass):
    """Build an OBE problem for a single F -> F' transition."""
    laserBeams = pylcp.conventional3DMOTBeams(
        k=1, s=s, delta=0., beam_type=pylcp.infinitePlaneWaveBeam
    )
    magField = pylcp.quadrupoleMagneticField(alpha)

    H_g, muq_g = pylcp.hamiltonians.singleF(F=F_g, gF=gF_g, muB=1)
    H_e, muq_e = pylcp.hamiltonians.singleF(F=F_e, gF=gF_e, muB=1)
    d_q = pylcp.hamiltonians.dqij_two_bare_hyperfine(F_g, F_e)
    n_e = 2 * F_e + 1
    hamiltonian = pylcp.hamiltonian(
        H_g, -det * np.eye(n_e) + H_e, muq_g, muq_e, d_q,
        mass=mass, muB=1, gamma=1, k=1
    )

    obe = pylcp.obe(laserBeams, magField, hamiltonian, transform_into_re_im=True)
    state_dim = hamiltonian.n**2 + 6
    return obe, hamiltonian, state_dim


PROBLEMS = {
    "F=0→F'=1 (Rb87-like)": dict(
        F_g=0, F_e=1, gF_g=0, gF_e=1, det=-4.0, s=1.5, alpha=4.3e-5, mass=805,
        tmax=1e4,
    ),
    "F=1→F'=2 (generic)": dict(
        F_g=1, F_e=2, gF_g=0, gF_e=0.5, det=-1.5, s=1.0, alpha=1.0, mass=200,
        tmax=1e4,
    ),
    "F=2→F'=3 (Sr88 green)": dict(
        F_g=2, F_e=3, gF_g=1.5, gF_e=1+1/3, det=-2.1, s=2, alpha=0.003, mass=1600,
        tmax=1e3,
    ),
}

# ---- Config ----
MAX_STEPS = 5000
NATOMS = 32
INNER_STEPS_SWEEP = [32, 64, 128, 256, 512]

print(f"Backend: {jax.default_backend()}")
print(f"Natoms: {NATOMS}, max_steps: {MAX_STEPS}")
print(f"Sweeping inner_max_steps: {INNER_STEPS_SWEEP}")
print("=" * 70)

all_results = {}

for name, params in PROBLEMS.items():
    print(f"\n{'=' * 70}")
    print(f"Problem: {name}")

    tmax = params.pop('tmax')
    obe, hamiltonian, state_dim = build_problem(**params)
    print(f"  n = {hamiltonian.n}, state_dim = {state_dim}, tmax = {tmax:.0e}")

    # Build initial conditions
    rng = np.random.default_rng(42)
    alpha_val = params['alpha']
    r0_all = (2.0 / alpha_val) * rng.standard_normal((NATOMS, 3))
    v0_all = 0.1 * rng.standard_normal((NATOMS, 3))

    rho0_all = []
    for i in range(NATOMS):
        obe.set_initial_position(r0_all[i])
        obe.set_initial_velocity(v0_all[i])
        obe.set_initial_rho_from_rateeq()
        rho0_all.append(obe.rho0)

    rho0_all = np.stack(rho0_all)
    y0_batch = jnp.array(np.concatenate([rho0_all, v0_all, r0_all], axis=1))
    keys_batch = jax.random.split(jax.random.PRNGKey(0), NATOMS)

    results = []
    for inner_steps in INNER_STEPS_SWEEP:
        print(f"  inner_max_steps={inner_steps} ... ", end="", flush=True)

        t0 = time.monotonic()
        sols = obe.evolve_motion(
            [0, tmax],
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            random_recoil=True,
            max_scatter_probability=0.5,
            max_step=tmax / MAX_STEPS,
            max_steps=MAX_STEPS,
            inner_max_steps=inner_steps,
        )
        elapsed = time.monotonic() - t0

        per_atom = elapsed / NATOMS
        results.append((inner_steps, elapsed, per_atom))
        print(f"{elapsed:.1f}s  |  {per_atom:.3f} s/atom")

    all_results[name] = results

# ---- Final summary ----
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for name, results in all_results.items():
    print(f"\n{name}:")
    print(f"  {'inner_max_steps':>16}  {'total (s)':>10}  {'s/atom':>10}")
    print(f"  {'-' * 46}")
    for inner_steps, elapsed, per_atom in results:
        print(f"  {inner_steps:>16}  {elapsed:>10.1f}  {per_atom:>10.3f}")