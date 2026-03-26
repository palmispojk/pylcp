"""
Green MOT simulation for Sr88 (F=2 -> F'=3) using GPU-batched OBE solver.

Rewritten from neutral_atoms_sim/MOT_sims/single_atom_sim.py to use the
new JAX/GPU-accelerated pylcp API with batched evolve_motion.
"""
import os

# Preallocate GPU memory before importing JAX.
if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.94'

import time
import numpy as np
import jax
import jax.numpy as jnp
import pickle

import pylcp
from pylcp.integration_tools_gpu import optimal_batch_size
import constants

# ---------------------------------------------------------------------------
# Build the trap
# ---------------------------------------------------------------------------
print(f"Starting to build the setup.")
trap_time = time.monotonic()

laserBeams = pylcp.conventional3DMOTBeams(
    k=constants.kmag, s=constants.s, delta=0., beam_type=pylcp.infinitePlaneWaveBeam
)
magField = pylcp.quadrupoleMagneticField(constants.alpha)

H_g, muq_g = pylcp.hamiltonians.singleF(F=2, gF=1.5, muB=constants.muB)
H_e, muq_e = pylcp.hamiltonians.singleF(F=3, gF=1 + 1 / 3, muB=constants.muB)
d_q = pylcp.hamiltonians.dqij_two_bare_hyperfine(2, 3)
hamiltonian = pylcp.hamiltonian(
    H_g, -constants.det * np.eye(7) + H_e, muq_g, muq_e, d_q,
    mass=constants.mass, muB=constants.muB, gamma=constants.gamma, k=constants.kmag
)

obe = pylcp.obe(laserBeams, magField, hamiltonian, transform_into_re_im=True)

# ---------------------------------------------------------------------------
# Build batched initial conditions
# ---------------------------------------------------------------------------
tmax = 1e5
# MAX_STEPS drives how far the outer while_loop runs.  Each outer step is
# scatter-rate limited to ~0.25–1.72 natural units depending on how close the
# atom is to resonance.  Worst case (fully captured, near resonance): dt≈0.25
# → need tmax/0.25 = 400 000 steps to reach tmax=1e5.
# SAVE_EVERY keeps output at 5000 points regardless of MAX_STEPS.
# INNER_MAX_STEPS: benchmark shows ~5 inner evals/outer step at max_step=0.25,
# so 64 is sufficient.
MAX_STEPS = 400_000
SAVE_EVERY = 80        # 400_000 // 80 = 5000 output points
INNER_MAX_STEPS = 64

state_dim = hamiltonian.n**2 + 6
optimal_n = optimal_batch_size(state_dim, MAX_STEPS, inner_max_steps=INNER_MAX_STEPS, safety=0.6, save_every=SAVE_EVERY)
Natoms = min(optimal_n, 1024) if optimal_n is not None else 96
print(f"State dim: {state_dim}, optimal batch size: {optimal_n}, using Natoms={Natoms}")

rng = np.random.default_rng()
r0_all = constants.rscale[None, :] * rng.standard_normal((Natoms, 3)) + constants.roffset[None, :]
v0_all = constants.vscale[None, :] * rng.standard_normal((Natoms, 3)) + constants.voffset[None, :]

# Compute equilibrium rho0 at each atom's starting position/velocity
# (matches original behaviour where rateeq was solved per atom)
rho0_all = []
for i in range(Natoms):
    obe.set_initial_position(r0_all[i])
    obe.set_initial_velocity(v0_all[i])
    obe.set_initial_rho_from_rateeq()
    rho0_all.append(obe.rho0)

rho0_all = np.stack(rho0_all)
y0_batch = jnp.array(np.concatenate([rho0_all, v0_all, r0_all], axis=1))
keys_batch = jax.random.split(jax.random.PRNGKey(rng.integers(0, 2**31)), Natoms)

trap_time_total = time.monotonic() - trap_time
m, s = divmod(int(trap_time_total), 60)
h, m = divmod(m, 60)
print(f"Trap build time: {h}h{m:02d}m{s:02d}s")
# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
print(f"Starting batched simulation of {Natoms} atoms on {jax.default_backend()}...")
t_total_start = time.monotonic()

sols = obe.evolve_motion(
    [0, tmax],
    y0_batch=y0_batch,
    keys_batch=keys_batch,
    random_recoil=True,
    max_scatter_probability=0.5,
    max_step=tmax / MAX_STEPS,
    max_steps=MAX_STEPS,
    inner_max_steps=INNER_MAX_STEPS,
    save_every=SAVE_EVERY,
    progress=True,
)

t_total = time.monotonic() - t_total_start
m, s = divmod(int(t_total), 60)
h, m = divmod(m, 60)
n_success = sum(1 for sol in sols if sol.success)
final_ts = np.array([float(sol._batched_state['t'][sol._index]) for sol in sols])
print(f"Simulation complete — {len(sols)} trajectories in {h}h{m:02d}m{s:02d}s")
print(f"  {t_total/Natoms:.2f} s/atom")
print(f"  Reached tmax: {n_success}/{Natoms} ({100*n_success/Natoms:.0f}%)")
print(f"  Final t: min={final_ts.min():.0f}  median={np.median(final_ts):.0f}  max={final_ts.max():.0f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = []
for sol in sols:
    results.append({
        't':        np.asarray(sol.t),
        'r':        np.asarray(sol.r),
        'v':        np.asarray(sol.v),
        'success':  sol.success,
        't_random': np.asarray(sol.t_random),
        'n_random': np.asarray(sol.n_random),
    })

with open('mot_simulation_data.pkl', 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Data saved to mot_simulation_data.pkl")