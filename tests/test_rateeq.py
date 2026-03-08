"""
Tests for pylcp/rateeq.py
"""
import pytest
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import pylcp.hamiltonians as hamiltonians
from pylcp.hamiltonian import hamiltonian
from pylcp.fields import laserBeams, laserBeam, constantMagneticField
from pylcp.rateeq import rateeq, force_profile


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_ham(gamma=1.0, k=1.0, mass=1.0):
    """Minimal F=0 -> F'=1 Hamiltonian (1 ground + 3 excited states)."""
    H0_g, mu_g = hamiltonians.singleF(F=0, gF=0)
    H0_e, mu_e = hamiltonians.singleF(F=1, gF=1)
    d_q = hamiltonians.dqij_two_bare_hyperfine(0, 1)
    return hamiltonian(H0_g, H0_e, mu_g, mu_e, d_q,
                       mass=mass, gamma=gamma, k=k)


@pytest.fixture
def ham():
    return make_ham(gamma=1.0, k=1.0, mass=1.0)


@pytest.fixture
def zero_B():
    return constantMagneticField(jnp.array([0., 0., 0.]))


@pytest.fixture
def single_beam_beams():
    """One σ+ beam along +z, on resonance, weak saturation."""
    return laserBeams([{'kvec': [0., 0., 1.], 'pol': +1, 's': 0.1, 'delta': 0.}])


@pytest.fixture
def symmetric_beams():
    """Two counter-propagating σ+/σ- beams along z, equal detuning & intensity."""
    return laserBeams([
        {'kvec': [0., 0.,  1.], 'pol': +1, 's': 0.5, 'delta': -1.0},
        {'kvec': [0., 0., -1.], 'pol': -1, 's': 0.5, 'delta': -1.0},
    ])


@pytest.fixture
def req(single_beam_beams, zero_B, ham):
    return rateeq(single_beam_beams, zero_B, ham)


@pytest.fixture
def req_sym(symmetric_beams, zero_B, ham):
    return rateeq(symmetric_beams, zero_B, ham)


# ---------------------------------------------------------------------------
# TestForceProfile
# ---------------------------------------------------------------------------

class TestForceProfile:
    def test_construction_2d_grid(self, req):
        R = np.zeros((3, 5, 4))
        V = np.zeros((3, 5, 4))
        fp = force_profile(R, V, req.laserBeams, req.hamiltonian)
        assert fp.F.shape == (3, 5, 4)
        assert fp.R.shape == (3, 5, 4)

    def test_construction_1d_grid(self, req):
        R = np.zeros((3, 10))
        V = np.zeros((3, 10))
        fp = force_profile(R, V, req.laserBeams, req.hamiltonian)
        assert fp.F.shape == (3, 10)

    def test_Rijl_key_present(self, req):
        R = np.zeros((3, 3))
        V = np.zeros((3, 3))
        fp = force_profile(R, V, req.laserBeams, req.hamiltonian)
        assert 'g->e' in fp.Rijl

    def test_Neq_shape(self, req):
        R = np.zeros((3, 5))
        V = np.zeros((3, 5))
        fp = force_profile(R, V, req.laserBeams, req.hamiltonian)
        # 4 states (1 ground + 3 excited), grid length 5
        assert fp.Neq.shape == (5, req.hamiltonian.n)


# ---------------------------------------------------------------------------
# TestRateeqInit
# ---------------------------------------------------------------------------

class TestRateeqInit:
    def test_hamiltonian_stored(self, req, ham):
        assert req.hamiltonian.n == ham.n

    def test_include_mag_forces_default_true(self, req):
        assert req.include_mag_forces is True

    def test_include_mag_forces_can_disable(self, single_beam_beams, zero_B, ham):
        r = rateeq(single_beam_beams, zero_B, ham, include_mag_forces=False)
        assert r.include_mag_forces is False

    def test_svd_eps_default(self, req):
        assert req.svd_eps == pytest.approx(1e-10)

    def test_custom_svd_eps(self, single_beam_beams, zero_B, ham):
        r = rateeq(single_beam_beams, zero_B, ham, svd_eps=1e-8)
        assert r.svd_eps == pytest.approx(1e-8)

    def test_recoil_velocity_key_present(self, req):
        assert 'g->e' in req.recoil_velocity

    def test_recoil_velocity_positive(self, req):
        assert req.recoil_velocity['g->e'] > 0

    def test_profile_starts_empty(self, req):
        assert req.profile == {}

    def test_r0_defaults_to_origin(self, req):
        assert jnp.allclose(req.r0, jnp.zeros(3))

    def test_v0_defaults_to_zero(self, req):
        assert jnp.allclose(req.v0, jnp.zeros(3))

    def test_Rev_decay_precomputed_for_diagonal_ham(self, req, ham):
        # F=0->F=1 with singleF gives a diagonal hamiltonian
        assert np.all(ham.diagonal)
        assert hasattr(req, 'Rev_decay')
        assert req.Rev_decay.shape == (ham.n, ham.n)


# ---------------------------------------------------------------------------
# TestSetInitialPop
# ---------------------------------------------------------------------------

class TestSetInitialPop:
    def test_wrong_length_raises(self, req):
        with pytest.raises(ValueError):
            req.set_initial_pop(jnp.array([1., 0., 0.]))  # too short (3 vs 4)

    def test_nan_raises(self, req, ham):
        bad = jnp.array([float('nan')] + [0.] * (ham.n - 1))
        with pytest.raises(ValueError):
            req.set_initial_pop(bad)

    def test_inf_raises(self, req, ham):
        bad = jnp.array([float('inf')] + [0.] * (ham.n - 1))
        with pytest.raises(ValueError):
            req.set_initial_pop(bad)

    def test_valid_pop_stored(self, req, ham):
        N0 = jnp.zeros(ham.n).at[0].set(1.)
        req.set_initial_pop(N0)
        assert jnp.allclose(req.N0, N0)

    def test_set_from_equilibrium(self, req):
        req.set_initial_pop_from_equilibrium()
        assert hasattr(req, 'N0')
        assert jnp.allclose(jnp.sum(req.N0), 1.0, atol=1e-6)
        assert jnp.all(req.N0 >= 0)


# ---------------------------------------------------------------------------
# TestConstructEvolutionMatrix
# ---------------------------------------------------------------------------

class TestConstructEvolutionMatrix:
    def test_returns_matrix_and_rijl(self, req, ham):
        Rev, Rijl = req.construct_evolution_matrix(
            jnp.zeros(3), jnp.zeros(3), t=0.)
        assert Rev.shape == (ham.n, ham.n)
        assert 'g->e' in Rijl

    def test_column_sums_are_zero(self, req, ham):
        """Probability conservation: columns of Rev must sum to zero."""
        Rev, _ = req.construct_evolution_matrix(
            jnp.zeros(3), jnp.zeros(3), t=0.)
        col_sums = jnp.sum(Rev, axis=0)
        assert jnp.allclose(col_sums, jnp.zeros(ham.n), atol=1e-10)

    def test_diagonal_negative_or_zero(self, req, ham):
        """Diagonal entries of Rev must be ≤ 0 (decay/pump out)."""
        Rev, _ = req.construct_evolution_matrix(
            jnp.zeros(3), jnp.zeros(3), t=0.)
        assert jnp.all(jnp.diag(Rev) <= 0)

    def test_off_diagonal_non_negative(self, req, ham):
        """Off-diagonal entries of Rev must be ≥ 0 (rates in)."""
        Rev, _ = req.construct_evolution_matrix(
            jnp.zeros(3), jnp.zeros(3), t=0.)
        n = ham.n
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert float(Rev[i, j]) >= -1e-12, \
                        f"Off-diagonal Rev[{i},{j}] = {float(Rev[i,j])} < 0"

    def test_pumping_increases_with_intensity(self, zero_B, ham):
        """Higher intensity → larger off-diagonal pumping rates."""
        beams_lo = laserBeams([{'kvec': [0., 0., 1.], 'pol': +1,
                                 's': 0.01, 'delta': 0.}])
        beams_hi = laserBeams([{'kvec': [0., 0., 1.], 'pol': +1,
                                 's': 1.0,  'delta': 0.}])
        req_lo = rateeq(beams_lo, zero_B, make_ham())
        req_hi = rateeq(beams_hi, zero_B, make_ham())
        Rev_lo, _ = req_lo.construct_evolution_matrix(
            jnp.zeros(3), jnp.zeros(3))
        Rev_hi, _ = req_hi.construct_evolution_matrix(
            jnp.zeros(3), jnp.zeros(3))
        # σ+ beam drives mF=0 → mF=+1, which is index 3 in the excited manifold.
        # Pumping Rev[3,0] should be larger for higher intensity.
        assert float(Rev_hi[3, 0]) > float(Rev_lo[3, 0])


# ---------------------------------------------------------------------------
# TestEquilibriumPopulations
# ---------------------------------------------------------------------------

class TestEquilibriumPopulations:
    def test_populations_sum_to_one(self, req):
        Neq = req.equilibrium_populations(jnp.zeros(3), jnp.zeros(3), t=0.)
        assert float(jnp.sum(Neq)) == pytest.approx(1.0, abs=1e-6)

    def test_populations_non_negative(self, req):
        Neq = req.equilibrium_populations(jnp.zeros(3), jnp.zeros(3), t=0.)
        assert jnp.all(Neq >= -1e-10)

    def test_return_details_gives_tuple(self, req):
        result = req.equilibrium_populations(
            jnp.zeros(3), jnp.zeros(3), t=0., return_details=True)
        assert len(result) == 3  # (Neq, Rev, Rijl)

    def test_ground_state_has_population(self, req):
        """At weak saturation the ground state should hold most population."""
        Neq = req.equilibrium_populations(jnp.zeros(3), jnp.zeros(3), t=0.)
        # Ground state is index 0 (1 state), excited are 1-3 (3 states)
        assert float(Neq[0]) > 0.0

    def test_zero_intensity_gives_all_ground(self, zero_B, ham):
        """With no laser, all population stays in the ground state."""
        no_beams = laserBeams([{'kvec': [0., 0., 1.], 'pol': +1,
                                 's': 0., 'delta': 0.}])
        req_dark = rateeq(no_beams, zero_B, make_ham())
        Neq = req_dark.equilibrium_populations(
            jnp.zeros(3), jnp.zeros(3), t=0.)
        # Ground state (index 0) should have all population
        assert float(Neq[0]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestFindEquilibriumForce
# ---------------------------------------------------------------------------

class TestFindEquilibriumForce:
    def test_returns_shape_3(self, req):
        req.set_initial_pop_from_equilibrium()
        F = req.find_equilibrium_force()
        assert F.shape == (3,)

    def test_return_details_gives_tuple(self, req):
        req.set_initial_pop_from_equilibrium()
        result = req.find_equilibrium_force(return_details=True)
        assert len(result) == 5  # (F_eq, f_eq, N_eq, Rijl, f_mag)

    def test_symmetric_beams_zero_force_at_origin(self, req_sym):
        """Two equal counter-propagating beams → net z-force = 0 at r=0, v=0."""
        req_sym.set_initial_pop_from_equilibrium()
        F = req_sym.find_equilibrium_force()
        assert float(F[2]) == pytest.approx(0.0, abs=1e-10)

    def test_single_beam_gives_positive_z_force(self, req):
        """+z beam → net force in +z direction."""
        req.set_initial_pop_from_equilibrium()
        F = req.find_equilibrium_force()
        assert float(F[2]) > 0.0

    def test_force_scale_with_intensity(self, zero_B, ham):
        """Force should increase with laser intensity."""
        beams_lo = laserBeams([{'kvec': [0., 0., 1.], 'pol': +1,
                                 's': 0.1, 'delta': 0.}])
        beams_hi = laserBeams([{'kvec': [0., 0., 1.], 'pol': +1,
                                 's': 2.0, 'delta': 0.}])
        req_lo = rateeq(beams_lo, zero_B, make_ham())
        req_hi = rateeq(beams_hi, zero_B, make_ham())
        req_lo.set_initial_pop_from_equilibrium()
        req_hi.set_initial_pop_from_equilibrium()
        F_lo = req_lo.find_equilibrium_force()
        F_hi = req_hi.find_equilibrium_force()
        assert float(F_hi[2]) > float(F_lo[2])

    def test_no_time_dependence_flag_works(self, req):
        """Calling find_equilibrium_force with no tdepend set should succeed."""
        req.set_initial_pop_from_equilibrium()
        F = req.find_equilibrium_force()
        assert not jnp.any(jnp.isnan(F))


# ---------------------------------------------------------------------------
# Test1DMOTForceProfile – regression tests for magnetic field gradient
# ---------------------------------------------------------------------------

class Test1DMOTForceProfile:
    """1D MOT with linear B-field gradient: force profile must be restoring.

    This is the primary regression test for the Zeeman shift bug in the JAX
    force-profile path (_generate_force_profile_jax).
    """

    @pytest.fixture
    def mot_fp(self):
        """Build a rateeq 1D MOT and return its force profile.

        Uses alpha scaled so that the Zeeman shift matches the detuning
        at x = ±x_res, i.e.  alpha * x_res * mu_z_per_state = |delta|.
        """
        from pylcp.fields import magField
        ham = make_ham(gamma=1.0, k=1.0, mass=1.0)
        # mu_z per excited mF state (physical units)
        mu_val = 1399624.49171  # |diag(mu_e[1])[0]|
        delta = -4.0
        x_res = 5.0  # resonance position
        alpha = abs(delta) / (x_res * mu_val)
        beams = laserBeams([
            {'kvec': [1., 0., 0.], 'pol': -1, 's': 1.0, 'delta': delta},
            {'kvec': [-1., 0., 0.], 'pol': -1, 's': 1.0, 'delta': delta},
        ])
        B = magField(lambda R: -alpha * R)
        req = rateeq(beams, B, ham)
        x = np.linspace(-10, 10, 21) * x_res / 10.0
        R = np.array([x, np.zeros_like(x), np.zeros_like(x)])
        V = np.zeros_like(R)
        fp = req.generate_force_profile(R, V)
        return fp, x

    def test_force_profile_no_nan(self, mot_fp):
        """Force profile with a B-field gradient must not contain NaN."""
        fp, _ = mot_fp
        assert not np.any(np.isnan(fp.F))

    def test_force_at_origin_is_zero(self, mot_fp):
        """By symmetry the force at x=0 (where B=0) must vanish."""
        fp, x = mot_fp
        origin_idx = np.argmin(np.abs(x))
        assert float(fp.F[0, origin_idx]) == pytest.approx(0., abs=1e-10)

    def test_force_is_restoring(self, mot_fp):
        """For x>0 the force must point in -x (restoring), and vice versa."""
        fp, x = mot_fp
        pos_mask = x > 1.0
        neg_mask = x < -1.0
        assert np.all(fp.F[0, pos_mask] < 0.), "Force should be negative for x>0"
        assert np.all(fp.F[0, neg_mask] > 0.), "Force should be positive for x<0"

    def test_force_is_antisymmetric(self, mot_fp):
        """F(x) ≈ -F(-x) for the symmetric 1D MOT."""
        fp, _ = mot_fp
        F_x = np.array(fp.F[0])
        assert np.allclose(F_x, -F_x[::-1], atol=1e-10)

    def test_force_nonzero_away_from_origin(self, mot_fp):
        """Force must be non-zero near the resonance position."""
        fp, x = mot_fp
        # Pick a point away from origin but within the trapping region
        idx = np.argmin(np.abs(x - x[len(x)//2 + len(x)//4]))
        assert abs(float(fp.F[0, idx])) > 1e-6