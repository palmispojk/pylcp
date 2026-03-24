"""
Green MOT simulation for Sr88 (F=2 -> F'=3) using GPU-batched OBE solver.

Rewritten from neutral_atoms_sim/MOT_sims/single_atom_sim.py to use the
new JAX/GPU-accelerated pylcp API with batched evolve_motion.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import scipy.constants as const

import pylcp

# ---------------------------------------------------------------------------
# Physical constants (Sr88 green MOT, same as original)
# ---------------------------------------------------------------------------
frq_real = 603976506.6e6 * 2 * np.pi
gamma_real = 61.4e6
kmag_real = frq_real / const.c
muB_real = const.physical_constants["Bohr magneton"][0]
mass_real = const.value('atomic mass constant') * 88
alpha_real = 0.4  # T/m

# Natural units
gamma = 1
kmag = 1
muB = 1
mass = mass_real * gamma_real / const.hbar / kmag_real**2
alpha = alpha_real * muB_real / (gamma_real * kmag_real * const.hbar)
det = -2.1 * gamma
s = 2

# Initial condition sampling scales
rscale = np.array([2, 2, 2]) / alpha
roffset = np.array([0.0, 0.0, 0.0])
vscale = np.array([0.1, 0.1, 0.1])
voffset = np.array([0.0, 0.0, 0.0])

# ---------------------------------------------------------------------------
# Build the trap
# ---------------------------------------------------------------------------
laserBeams = pylcp.conventional3DMOTBeams(
    k=kmag, s=s, delta=0., beam_type=pylcp.infinitePlaneWaveBeam
)
magField = pylcp.quadrupoleMagneticField(alpha)

H_g, muq_g = pylcp.hamiltonians.singleF(F=2, gF=1.5, muB=muB)
H_e, muq_e = pylcp.hamiltonians.singleF(F=3, gF=1 + 1 / 3, muB=muB)
d_q = pylcp.hamiltonians.dqij_two_bare_hyperfine(2, 3)
hamiltonian = pylcp.hamiltonian(
    H_g, -det * np.eye(7) + H_e, muq_g, muq_e, d_q,
    mass=mass, muB=muB, gamma=gamma, k=kmag
)

obe = pylcp.obe(laserBeams, magField, hamiltonian, transform_into_re_im=True)

# Compute equilibrium density matrix at origin (used for all atoms)
obe.set_initial_position(np.array([0., 0., 0.]))
obe.set_initial_velocity(np.array([0., 0., 0.]))
obe.set_initial_rho_from_rateeq()
rho0 = obe.rho0  # flattened density matrix

# ---------------------------------------------------------------------------
# Build batched initial conditions
# ---------------------------------------------------------------------------
Natoms = 96
tmax = 1e5

rng = np.random.default_rng()
r0_all = rscale[None, :] * rng.standard_normal((Natoms, 3)) + roffset[None, :]
v0_all = vscale[None, :] * rng.standard_normal((Natoms, 3)) + voffset[None, :]

# y0 = [rho0, v0, r0] for each atom
rho0_tiled = np.tile(rho0, (Natoms, 1))
y0_batch = jnp.array(np.concatenate([rho0_tiled, v0_all, r0_all], axis=1))
keys_batch = jax.random.split(jax.random.PRNGKey(rng.integers(0, 2**31)), Natoms)

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
print(f"Starting batched simulation of {Natoms} atoms on {jax.default_backend()}...")

sols = obe.evolve_motion(
    [0, tmax],
    y0_batch=y0_batch,
    keys_batch=keys_batch,
    random_recoil=True,
    max_scatter_probability=0.5,
)

print(f"Simulation complete — {len(sols)} trajectories.")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = []
for sol in sols:
    results.append((np.asarray(sol.t), np.asarray(sol.r), np.asarray(sol.v)))

with open('mot_simulation_data.pkl', 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Data saved to mot_simulation_data.pkl")