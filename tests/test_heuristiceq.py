import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pylcp
import matplotlib.pyplot as plt

mass = 100.
# --- Setup: 1D optical molasses 
beams = pylcp.laserBeams([
    pylcp.laserBeam(kvec=np.array([0., 0.,  1.]), s=1., delta=-2.),
    pylcp.laserBeam(kvec=np.array([0., 0., -1.]), s=1., delta=-2.),
], beam_type=pylcp.laserBeam)

magField = pylcp.magField(np.array([0., 0., 0.]))

eq = pylcp.heuristiceq(beams, magField, gamma=1., k=1., mass=mass)

# --- Test 1: equilibrium force at v=0 should be ~0 by symmetry
F = eq.find_equilibrium_force()
print("F at v=0:", F)  # expect ~[0, 0, 0]

# --- Test 2: force profile over velocity range
V = np.zeros((3, 1, 1, 51))
V[2, 0, 0, :] = np.linspace(-5, 5, 51)
R = np.zeros_like(V)

prof = eq.generate_force_profile(R, V, name='molasses')
plt.plot(V[2, 0, 0, :], prof.F[2, 0, 0, :])
plt.xlabel('v_z'); plt.ylabel('F_z')
plt.title('Heuristic molasses force')
plt.show()

vz = V[2, 0, 0, :]
Fz = prof.F[2, 0, 0, :]


#  check slope at v=0 to get damping coefficient
dv = vz[1] - vz[0]
beta = -(Fz[25] - Fz[24]) / dv 
print(f"Damping coefficient beta = {beta:.4f}")
print(f"Cooling time = mass/beta = {mass/beta:.1f}") # the amount of time steps needed to cool to 0


# --- Test 3: evolve_motion for a single atom ---
y0 = jnp.array([[0., 0., 2., 0., 0., 0.]])  # [vx,vy,vz, x,y,z]
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 1)

sols = eq.evolve_motion([0., 50.], y0, keys, max_steps=5000)
sol = sols[0]
print("Final v_z:", sol.v[2, -1])  # should have decreased from the initial velocity however takes a lot of time steps to integrate to 0
print("Final z:  ", sol.r[2, -1])
