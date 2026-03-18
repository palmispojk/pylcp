"""
Benchmark: CPU vs GPU methods for force profile and evolve motion.

Force profile: generate_force_profile (CPU) vs generate_force_profile_gpu (GPU vmap).
Evolve motion: single-atom vs batched evolve_motion.

Uses the notebook 03 parameters (alpha=1e-4, det=-2.5, s=1.25).
"""
import time
import numpy as np
import pylcp
import jax
import jax.numpy as jnp

DET = -2.5
S = 1.25
ALPHA = 1e-4
NPTS = 4  # force profile realizations
N_ATOMS = 16  # evolve motion atoms
SEED = 42


def setup():
    Hg, Bgq = pylcp.hamiltonians.singleF(F=0, gF=0, muB=1)
    He, Beq = pylcp.hamiltonians.singleF(F=1, gF=1, muB=1)
    dijq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)
    hamiltonian = pylcp.hamiltonian(
        Hg, -DET * np.eye(3) + He, Bgq, Beq, dijq, mass=100
    )
    laserBeams = pylcp.conventional3DMOTBeams(
        s=S, delta=0., beam_type=pylcp.infinitePlaneWaveBeam
    )
    magField = pylcp.quadrupoleMagneticField(ALPHA)
    obe = pylcp.obe(laserBeams, magField, hamiltonian,
                    transform_into_re_im=True)
    return obe


def warmup(obe):
    """JIT warmup for both methods."""
    R0 = [np.array([0.]), np.array([0.]), np.array([0.])]
    V0 = [np.array([0.]), np.array([0.]), np.array([0.])]
    obe.generate_force_profile(R0, V0, deltat=1., itermax=1)
    obe.profile.clear()
    obe.generate_force_profile_gpu(R0, V0, deltat=1., itermax=1)
    obe.profile.clear()


def run_method(obe, method_name, method_func):
    z = np.arange(-5.01, 5.01, 0.25)
    R_base = [np.zeros(z.shape), np.zeros(z.shape), z / ALPHA]
    V_base = [np.zeros(z.shape), np.zeros(z.shape), np.zeros(z.shape)]

    kw = dict(
        deltat_tmax=2 * np.pi * 100,
        deltat_r=4 / ALPHA,
        itermax=1000,
        progress_bar=False,
        npts_conv_divisor=1,
    )

    np.random.seed(SEED)
    t0 = time.perf_counter()
    for ii in range(NPTS):
        offset = 2 * np.pi * (np.random.rand(3) - 0.5).reshape(3, 1)
        R_shifted = [R_base[j] + offset[j] for j in range(3)]
        method_func(R_shifted, V_base, name='z_%d' % ii, **kw)
    elapsed = time.perf_counter() - t0

    F_all = np.array([
        np.asarray(obe.profile[k].F)
        for k in sorted(obe.profile) if k.startswith('z_')
    ])
    iters_all = np.array([
        np.asarray(obe.profile[k].iterations)
        for k in sorted(obe.profile) if k.startswith('z_')
    ])

    avgF = np.mean(F_all[:, 2, :], axis=0)
    d2F = np.diff(avgF, n=2)

    print(f"\n  {method_name}:")
    print(f"    Time:       {elapsed:.1f}s")
    print(f"    Iters:      min={iters_all.min()}, max={iters_all.max()}, mean={iters_all.mean():.1f}")
    print(f"    max|d2F|:   {np.max(np.abs(d2F)):.4e}")
    print(f"    mean|d2F|:  {np.mean(np.abs(d2F)):.4e}")

    obe.profile.clear()
    return avgF, elapsed


def warmup_evolve(obe):
    """JIT warmup for evolve_motion (single and batched)."""
    rho0 = jnp.array(obe.rho0)
    v0 = jnp.zeros(3)
    r0 = jnp.zeros(3)
    y0 = jnp.concatenate([rho0, v0, r0])

    # Single atom
    obe.evolve_motion([0, 100], y0_batch=y0[jnp.newaxis, :],
                      freeze_axis=[True, True, False], max_steps=200)

    # Batch of 2
    y0_batch = jnp.stack([y0, y0])
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    obe.evolve_motion([0, 100], y0_batch=y0_batch, keys_batch=keys,
                      freeze_axis=[True, True, False], max_steps=200)


def run_evolve_motion(obe, label, n_atoms, batched=False):
    """Benchmark evolve_motion: serial (one at a time) vs batched."""
    np.random.seed(SEED)
    t_span = [0, 2 * np.pi * 500]

    rho0 = jnp.array(obe.rho0)
    # Random initial positions within MOT capture range
    r_init = np.random.uniform(-2 / ALPHA, 2 / ALPHA, size=(n_atoms, 3))
    r_init[:, :2] = 0.  # only z displacement
    v_init = np.zeros((n_atoms, 3))

    y0_list = []
    for i in range(n_atoms):
        y0 = jnp.concatenate([rho0, jnp.array(v_init[i]), jnp.array(r_init[i])])
        y0_list.append(y0)

    kw = dict(freeze_axis=[True, True, False], max_steps=10000)

    t0 = time.perf_counter()
    if batched:
        y0_batch = jnp.stack(y0_list)
        keys = jax.random.split(jax.random.PRNGKey(SEED), n_atoms)
        obe.evolve_motion(t_span, y0_batch=y0_batch, keys_batch=keys, **kw)
        final_z = np.array([sol.r[2, -1] for sol in obe.sols])
    else:
        final_z = np.zeros(n_atoms)
        for i in range(n_atoms):
            obe.evolve_motion(t_span, y0_batch=y0_list[i][jnp.newaxis, :], **kw)
            final_z[i] = float(obe.sol.r[2, -1])
    elapsed = time.perf_counter() - t0

    print(f"\n  {label} ({n_atoms} atoms):")
    print(f"    Time:       {elapsed:.1f}s")
    print(f"    Final z:    mean={np.mean(final_z):.2f}, std={np.std(final_z):.2f}")

    return final_z, elapsed


if __name__ == '__main__':
    print(f"Parameters: det={DET}, s={S}, alpha={ALPHA}")
    print(f"Realizations: {NPTS}, seed: {SEED}, atoms: {N_ATOMS}")

    obe = setup()

    # --- Force profile benchmark ---
    print("\n" + "=" * 50)
    print("  Force Profile Benchmark")
    print("=" * 50)

    print("\nWarming up JIT (force profile)...")
    warmup(obe)

    avgF_cpu, t_cpu = run_method(obe, "CPU (generate_force_profile)", obe.generate_force_profile)
    avgF_gpu, t_gpu = run_method(obe, "GPU (generate_force_profile_gpu)", obe.generate_force_profile_gpu)

    diff = avgF_cpu - avgF_gpu
    print(f"\n  Force Profile Comparison:")
    print(f"    Speedup:         {t_cpu/t_gpu:.2f}x")
    print(f"    Max |F diff|:    {np.max(np.abs(diff)):.4e}")
    print(f"    Mean |F diff|:   {np.mean(np.abs(diff)):.4e}")

    # --- Evolve motion benchmark ---
    print("\n" + "=" * 50)
    print("  Evolve Motion Benchmark")
    print("=" * 50)

    print("\nWarming up JIT (evolve motion)...")
    warmup_evolve(obe)

    z_serial, t_serial = run_evolve_motion(obe, "Serial", N_ATOMS, batched=False)
    z_batch, t_batch = run_evolve_motion(obe, "Batched", N_ATOMS, batched=True)

    zdiff = z_serial - z_batch
    print(f"\n  Evolve Motion Comparison:")
    print(f"    Speedup:         {t_serial/t_batch:.2f}x")
    print(f"    Max |z diff|:    {np.max(np.abs(zdiff)):.4e}")
    print(f"    Mean |z diff|:   {np.mean(np.abs(zdiff)):.4e}")
