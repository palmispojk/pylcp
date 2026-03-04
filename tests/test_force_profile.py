import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import pylcp

# Ground state: F=0 (1 state), Excited state: F'=1 (3 states)
H0_g, muq_g = pylcp.hamiltonians.singleF(0, gF=0, muB=1)
H0_e, muq_e = pylcp.hamiltonians.singleF(1, gF=1, muB=1)
dq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)  # d_q: g→e, shape (3, 1, 3)

hamiltonian = pylcp.hamiltonian()
hamiltonian.add_H_0_block('g', H0_g)
hamiltonian.add_H_0_block('e', H0_e)
hamiltonian.add_mu_q_block('g', muq_g)
hamiltonian.add_mu_q_block('e', muq_e)
hamiltonian.add_d_q_block('g', 'e', dq)

# Counter-propagating beams along z, detuned -1 linewidth, saturation s=0.1
laserBeams = pylcp.laserBeams([
    pylcp.infinitePlaneWaveBeam(kvec=np.array([0., 0., 1.]),
                                pol=np.array([1., 0., 0.]),
                                delta=-1.0, s=0.1),
    pylcp.infinitePlaneWaveBeam(kvec=np.array([0., 0., -1.]),
                                pol=np.array([1., 0., 0.]),
                                delta=-1.0, s=0.1),
], beam_type=pylcp.infinitePlaneWaveBeam)
magField = pylcp.constantMagneticField(np.zeros(3))

obe = pylcp.obe(laserBeams, magField, hamiltonian, transform_into_re_im=True)
obe.set_initial_position(np.zeros(3))
obe.set_initial_velocity(np.zeros(3))


# ── Test 1: find_equilibrium_force at v=0 → force should be ~0 by symmetry ──
obe.set_initial_velocity(np.zeros(3))
f = obe.find_equilibrium_force(deltat=200, itermax=50, Npts=1001)
print(f"Force at v=0: {np.array(f)}")
assert np.allclose(np.array(f), 0.0, atol=1e-6), "Force at v=0 should be zero"

# ── Test 2: find_equilibrium_force at +v and -v → antisymmetric ─────────────
v_test = 0.1
obe.set_initial_velocity(np.array([0., 0., v_test]))
f_pos = np.array(obe.find_equilibrium_force(deltat=200, itermax=50, Npts=1001))

obe.set_initial_velocity(np.array([0., 0., -v_test]))
f_neg = np.array(obe.find_equilibrium_force(deltat=200, itermax=50, Npts=1001))

print(f"Force at +v: {f_pos},  -v: {f_neg}")
assert np.allclose(f_pos, -f_neg, atol=1e-6), "Force must be antisymmetric in v"

# ── Test 3: generate_force_profile matches find_equilibrium_force ─────────────
# Small 1D grid along z-velocity
v_grid = np.array([0., 0.1, -0.1])
R = np.zeros((3, 1, 1, len(v_grid)))   # single spatial point, vary vz
V = np.zeros((3, 1, 1, len(v_grid)))
V[2] = v_grid  # vary vz

profile = obe.generate_force_profile(R, V, name='test')

# Compare each profile point against find_equilibrium_force
for ii, vz in enumerate(v_grid):
    obe.set_initial_position(np.zeros(3))
    obe.set_initial_velocity(np.array([0., 0., vz]))
    f_single = np.array(obe.find_equilibrium_force(deltat=200, itermax=50, Npts=1001))
    f_profile = np.array(profile.F[:, 0, 0, ii])
    print(f"vz={vz:.2f}: single={f_single}, profile={f_profile}")
    assert np.allclose(f_single, f_profile, atol=1e-5), \
        f"Mismatch at vz={vz}: {f_single} vs {f_profile}"

print("All tests passed.")
