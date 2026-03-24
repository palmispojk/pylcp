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
tmax = 1e3
MAX_STEPS = 5000  # ~5000 output points, matching original t_eval resolution

state_dim = hamiltonian.n**2 + 6
optimal_n = optimal_batch_size(state_dim, MAX_STEPS, inner_max_steps=128, safety=0.6)
Natoms = optimal_n if optimal_n is not None else 96
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
)

t_total = time.monotonic() - t_total_start
m, s = divmod(int(t_total), 60)
h, m = divmod(m, 60)
print(f"Simulation complete — {len(sols)} trajectories in {h}h{m:02d}m{s:02d}s")
print(f"  {t_total/Natoms:.2f} s/atom")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = []
for sol in sols:
    results.append((np.asarray(sol.t), np.asarray(sol.r), np.asarray(sol.v)))

with open('mot_simulation_data.pkl', 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Data saved to mot_simulation_data.pkl")