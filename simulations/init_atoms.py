"""Initialize atoms for a MOT stage from the previous stage's pickle output.

Each stage saves {'r': (N, 3), 'v': (N, 3)} of its captured atoms in its own
natural units. `initialize_from_pickle` reads that file, rescales to the
destination stage's natural units if the transition differs (blue -> red),
feeds each atom through the governing-equation's built-in initializers
(`set_initial_position_and_velocity` + `set_initial_rho_from_rateeq`), and
returns `(y0_batch, keys_batch)` ready for `evolve_motion`.
"""
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def initialize_from_pickle(pickle_path, obe, dst_constants, src_constants=None,
                           n_atoms=None, rng=None, seed=None):
    """Load atoms from an upstream stage and build (y0_batch, keys_batch).

    Parameters
    ----------
    pickle_path : str or Path
        Upstream stage's final-state pickle: {'r': (N, 3), 'v': (N, 3)}.
    obe : pylcp.obe or pylcp.rateeq
        Governing-equation object for this stage. Its built-in initializers
        (`set_initial_position_and_velocity`, `set_initial_rho_from_rateeq`)
        are used to set up each atom — the loader does not touch the state
        vector directly.
    dst_constants : module
        This stage's `constants` module. Must expose `kmag_real`, `gamma_real`.
    src_constants : module, optional
        Upstream stage's `constants`. If None, no unit conversion is applied
        (valid when source and destination share a transition).
    n_atoms : int, optional
        Number of atoms to produce. If None, uses the captured count as-is.
        If larger, samples with replacement.
    rng : np.random.Generator, optional
        Used for resampling and seeding `keys_batch` if `seed` is None.
    seed : int, optional
        Explicit PRNG seed for `keys_batch`. Overrides `rng` for the keys.

    Returns
    -------
    y0_batch : jnp.ndarray, shape (n_atoms, state_dim)
        Stacked [rho0 || v0 || r0] in this stage's natural units.
    keys_batch : jnp.ndarray, shape (n_atoms, 2)
        Per-atom JAX PRNG keys for `evolve_motion(..., random_recoil=True)`.
    """
    path = Path(pickle_path)
    with path.open('rb') as f:
        state = pickle.load(f)

    r = np.asarray(state['r'], dtype=float)
    v = np.asarray(state['v'], dtype=float)
    if r.ndim != 2 or r.shape[1] != 3 or v.shape != r.shape:
        raise ValueError(
            f"Expected r, v with shape (N, 3); got {r.shape}, {v.shape}"
        )

    # Cross-transition unit conversion: natural units scale with (k, gamma).
    # r_nat = r_SI * k_real ;  v_nat = v_SI * k_real / gamma_real.
    if src_constants is not None:
        k_src, g_src = src_constants.kmag_real, src_constants.gamma_real
        k_dst, g_dst = dst_constants.kmag_real, dst_constants.gamma_real
        r = r * (k_dst / k_src)
        v = v * ((g_src * k_dst) / (k_src * g_dst))

    if rng is None:
        rng = np.random.default_rng()

    N_saved = r.shape[0]
    if n_atoms is None:
        n_atoms = N_saved
    if n_atoms != N_saved:
        idx = rng.choice(N_saved, size=n_atoms, replace=(n_atoms > N_saved))
        r, v = r[idx], v[idx]

    # Feed each atom through the governing-equation's built-in initializers.
    rho0_all = []
    for i in range(n_atoms):
        obe.set_initial_position_and_velocity(r[i], v[i])
        obe.set_initial_rho_from_rateeq()
        rho0_all.append(np.asarray(obe.rho0))

    rho0_all = np.stack(rho0_all)
    y0_batch = jnp.asarray(np.concatenate([rho0_all, v, r], axis=1))

    key_seed = seed if seed is not None else int(rng.integers(0, 2**31))
    keys_batch = jax.random.split(jax.random.PRNGKey(key_seed), n_atoms)

    return y0_batch, keys_batch
