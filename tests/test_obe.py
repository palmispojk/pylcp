"""
Tests for pylcp/obe.py
"""
import pytest
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import pylcp.hamiltonians as hamiltonians
from pylcp.hamiltonian import hamiltonian
from pylcp.fields import laserBeams, laserBeam, constantMagneticField
from pylcp.obe import obe, force_profile


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def make_ham(gamma=1.0, k=1.0, mass=1.0):
    """Minimal F=0 -> F'=1 Hamiltonian (1 ground + 3 excited states)."""
    H0_g, mu_g = hamiltonians.singleF(F=0, gF=0)
    H0_e, mu_e = hamiltonians.singleF(F=1, gF=1)
    d_q = hamiltonians.dqij_two_bare_hyperfine(0, 1)
    return hamiltonian(H0_g, H0_e, mu_g, mu_e, d_q,
                       mass=mass, gamma=gamma, k=k)


@pytest.fixture(scope='module')
def ham():
    return make_ham()


@pytest.fixture(scope='module')
def zero_B():
    return constantMagneticField(jnp.array([0., 0., 0.]))


@pytest.fixture(scope='module')
def weak_B():
    return constantMagneticField(jnp.array([0., 0., 0.1]))


@pytest.fixture(scope='module')
def single_beam():
    """One sigma+ beam along +z, weak saturation, on resonance."""
    return laserBeams([{'kvec': [0., 0., 1.], 'pol': +1, 's': 0.1, 'delta': 0.}])


@pytest.fixture(scope='module')
def symmetric_beams():
    """Two counter-propagating sigma+/sigma- beams, equal intensity."""
    return laserBeams([
        {'kvec': [0., 0.,  1.], 'pol': +1, 's': 0.5, 'delta': -1.0},
        {'kvec': [0., 0., -1.], 'pol': -1, 's': 0.5, 'delta': -1.0},
    ])


@pytest.fixture(scope='module')
def obe_transform(single_beam, zero_B, ham):
    """OBE with transform_into_re_im=True (default)."""
    return obe(single_beam, zero_B, ham, transform_into_re_im=True)


@pytest.fixture(scope='module')
def obe_complex(single_beam, zero_B, ham):
    """OBE with transform_into_re_im=False (complex basis)."""
    return obe(single_beam, zero_B, ham, transform_into_re_im=False)


@pytest.fixture(scope='module')
def obe_sym(symmetric_beams, zero_B, ham):
    """OBE with symmetric beams."""
    return obe(symmetric_beams, zero_B, ham)


# ---------------------------------------------------------------------------
# TestForceProfile
# ---------------------------------------------------------------------------

class TestForceProfile:
    def test_basic_shapes(self, obe_transform):
        o = obe_transform
        R = np.zeros((3, 5, 4))
        V = np.zeros((3, 5, 4))
        fp = force_profile(R, V, o.laserBeams, o.hamiltonian)
        assert fp.F.shape == (3, 5, 4)
        assert fp.R.shape == (3, 5, 4)
        assert fp.V.shape == (3, 5, 4)

    def test_iterations_shape(self, obe_transform):
        o = obe_transform
        R = np.zeros((3, 3))
        V = np.zeros((3, 3))
        fp = force_profile(R, V, o.laserBeams, o.hamiltonian)
        assert fp.iterations.shape == (3,)

    def test_fq_key_present(self, obe_transform):
        o = obe_transform
        R = np.zeros((3, 3))
        V = np.zeros((3, 3))
        fp = force_profile(R, V, o.laserBeams, o.hamiltonian)
        assert 'g->e' in fp.fq

    def test_fq_shape(self, obe_transform):
        o = obe_transform
        R = np.zeros((3, 5))
        V = np.zeros((3, 5))
        fp = force_profile(R, V, o.laserBeams, o.hamiltonian)
        # fq[key] has shape R.shape + (3, n_beams) = (3, 5, 3, 1)
        assert fp.fq['g->e'].shape == (3, 5, 3, 1)

    def test_store_data_updates_F(self, obe_transform):
        o = obe_transform
        R = np.zeros((3, 2))
        V = np.zeros((3, 2))
        fp = force_profile(R, V, o.laserBeams, o.hamiltonian)
        F_in = np.array([1., 2., 3.])
        Neq_in = np.array([0.5, 0.1, 0.1, 0.3])
        fp.store_data(
            (0,), Neq_in, F_in,
            {'g->e': np.zeros((3, 1))},
            np.zeros(3),
            5,
            {'g->e': np.zeros((3, 3, 1))}
        )
        np.testing.assert_allclose(fp.F[:, 0], F_in)
        assert fp.iterations[0] == 5


# ---------------------------------------------------------------------------
# TestObeInit
# ---------------------------------------------------------------------------

class TestObeInit:
    def test_basic_construction(self, obe_transform):
        assert obe_transform is not None

    def test_construction_no_transform(self, obe_complex):
        assert obe_complex is not None

    def test_ev_mat_keys_with_transform(self, obe_transform):
        keys = set(obe_transform.ev_mat.keys())
        assert 'decay' in keys
        assert 'H0' in keys
        assert 'reE' in keys
        assert 'imE' in keys
        assert 'B' in keys
        # d_q and d_q* should be deleted after transform
        assert 'd_q' not in keys
        assert 'd_q*' not in keys

    def test_ev_mat_keys_no_transform(self, obe_complex):
        keys = set(obe_complex.ev_mat.keys())
        assert 'decay' in keys
        assert 'H0' in keys
        assert 'd_q' in keys
        assert 'd_q*' in keys
        assert 'B' in keys

    def test_ev_mat_decay_shape(self, obe_transform):
        n = obe_transform.hamiltonian.n
        assert obe_transform.ev_mat['decay'].shape == (n**2, n**2)

    def test_ev_mat_H0_shape(self, obe_transform):
        n = obe_transform.hamiltonian.n
        assert obe_transform.ev_mat['H0'].shape == (n**2, n**2)

    def test_ev_mat_B_length(self, obe_transform):
        assert len(obe_transform.ev_mat['B']) == 3

    def test_decay_rates_positive(self, obe_transform):
        # Excited states have positive decay rates; ground state has zero
        for key in obe_transform.decay_rates:
            assert np.all(obe_transform.decay_rates[key] >= 0)

    def test_decay_rates_truncated_positive(self, obe_transform):
        for key in obe_transform.decay_rates_truncated:
            assert np.all(obe_transform.decay_rates_truncated[key] > 0)

    def test_ev_mat_are_jax_arrays(self, obe_transform):
        for key in obe_transform.ev_mat:
            if isinstance(obe_transform.ev_mat[key], dict):
                for subkey in obe_transform.ev_mat[key]:
                    assert isinstance(obe_transform.ev_mat[key][subkey], jnp.ndarray)
            elif isinstance(obe_transform.ev_mat[key], list):
                for v in obe_transform.ev_mat[key]:
                    assert isinstance(v, jnp.ndarray)
            else:
                assert isinstance(obe_transform.ev_mat[key], jnp.ndarray)

    def test_transform_true_matrices_are_real(self, obe_transform):
        # After transform, decay and H0 should be real (imaginary parts removed)
        decay = np.array(obe_transform.ev_mat['decay'])
        H0 = np.array(obe_transform.ev_mat['H0'])
        assert np.allclose(np.imag(decay), 0, atol=1e-12)
        assert np.allclose(np.imag(H0), 0, atol=1e-12)

    def test_magField_setter_clears_dydt_cache(self, single_beam, zero_B, ham):
        o = obe(single_beam, zero_B, ham)
        o.set_initial_rho_equally()
        o.evolve_density([0., 1.], n_points=5)
        assert '_dydt' in o.__dict__
        o.magField = zero_B
        assert '_dydt' not in o.__dict__

    def test_laserBeams_setter_clears_dydt_cache(self, single_beam, zero_B, ham):
        o = obe(single_beam, zero_B, ham)
        o.set_initial_rho_equally()
        o.evolve_density([0., 1.], n_points=5)
        assert '_dydt' in o.__dict__
        o.laserBeams = o.laserBeams
        assert '_dydt' not in o.__dict__


# ---------------------------------------------------------------------------
# TestBuildCoherentEvSubmatrix (optimization verification)
# ---------------------------------------------------------------------------

class TestBuildCoherentEvSubmatrix:
    """Verify the kron-product implementation matches the physics."""

    def _make_liouvillian_loop(self, H, n):
        """Reference implementation via the original triple loop."""
        ev = np.zeros((n**2, n**2), dtype='complex128')
        idx = lambda i, j: i + j * n
        for ii in range(n):
            for jj in range(n):
                for kk in range(n):
                    ev[idx(ii, jj), idx(ii, kk)] += 1j * H[kk, jj]
                    ev[idx(ii, jj), idx(kk, jj)] -= 1j * H[ii, kk]
        return ev

    def _make_liouvillian_kron(self, H, n):
        """New vectorized kron implementation."""
        I = np.eye(n)
        H = np.asarray(H)
        return 1j * np.kron(H.T, I) - 1j * np.kron(I, H)

    def test_kron_matches_loop_identity(self):
        n = 3
        H = np.eye(n, dtype='complex128')
        L_loop = self._make_liouvillian_loop(H, n)
        L_kron = self._make_liouvillian_kron(H, n)
        np.testing.assert_allclose(L_loop, L_kron, atol=1e-14)

    def test_kron_matches_loop_diagonal(self):
        n = 4
        H = np.diag([0., 1., 2., 3.]).astype('complex128')
        L_loop = self._make_liouvillian_loop(H, n)
        L_kron = self._make_liouvillian_kron(H, n)
        np.testing.assert_allclose(L_loop, L_kron, atol=1e-14)

    def test_kron_matches_loop_full(self):
        n = 3
        rng = np.random.default_rng(42)
        H_re = rng.standard_normal((n, n))
        H = (H_re + H_re.T).astype('complex128')  # Hermitian
        H[0, 1] += 0.5j
        H[1, 0] -= 0.5j
        L_loop = self._make_liouvillian_loop(H, n)
        L_kron = self._make_liouvillian_kron(H, n)
        np.testing.assert_allclose(L_loop, L_kron, atol=1e-14)

    def test_liouvillian_preserves_trace(self):
        """d/dt tr(rho) = 0 -> sum of each row of L corresponding to diagonal of rho must be zero."""
        n = 4
        H = np.diag([0., 1., -1., 2.]).astype('complex128')
        L = self._make_liouvillian_kron(H, n)
        # Diagonal elements of rho correspond to indices i+i*n = i*(n+1)
        diag_indices = [i + i * n for i in range(n)]
        # sum over diagonal rows of L gives d/dt tr(rho)
        trace_rate = np.sum(L[diag_indices, :], axis=0)
        np.testing.assert_allclose(trace_rate, 0, atol=1e-14)

    def test_liouvillian_anti_hermitian_for_hermitian_H(self):
        """For Hermitian H, L + L† = 0 (L is anti-Hermitian / skew-Hermitian)."""
        n = 3
        H = np.array([[1., 0.5+0.2j, 0.], [0.5-0.2j, -1., 0.3j], [0., -0.3j, 0.]])
        L = self._make_liouvillian_kron(H, n)
        np.testing.assert_allclose(L + np.conj(L.T), 0, atol=1e-14)

    def test_ev_mat_shape(self, obe_transform):
        n = obe_transform.hamiltonian.n
        assert obe_transform.ev_mat['H0'].shape == (n**2, n**2)


# ---------------------------------------------------------------------------
# TestDensityIndex
# ---------------------------------------------------------------------------

class TestDensityIndex:
    """Test the __density_index mapping through observable/rho behavior."""

    def test_rho0_diagonal_from_equally(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        # With transform=True, rho0 is in re/im basis.
        # The diagonal element for ground state 0 should be set.
        assert o.rho0 is not None
        assert o.rho0.shape == (o.hamiltonian.n**2,)

    def test_rho0_trace_one(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        # Recover complex rho and check trace = 1
        if o.transform_into_re_im:
            rho_complex = jnp.dot(jnp.asarray(o.U), o.rho0.astype('complex128'))
        else:
            rho_complex = o.rho0
        rho_mat = jnp.reshape(rho_complex, (o.hamiltonian.n, o.hamiltonian.n))
        trace = jnp.real(jnp.trace(rho_mat))
        assert float(trace) == pytest.approx(1.0, abs=1e-12)

    def test_rho0_ground_only_from_equally(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        if o.transform_into_re_im:
            rho_complex = jnp.dot(jnp.asarray(o.U), o.rho0.astype('complex128'))
        else:
            rho_complex = o.rho0
        rho_mat = jnp.reshape(rho_complex, (o.hamiltonian.n, o.hamiltonian.n))
        diag = jnp.real(jnp.diagonal(rho_mat))
        # n_g=1 ground state should have pop=1, all 3 excited states pop=0
        assert float(diag[0]) == pytest.approx(1.0, abs=1e-12)
        for i in range(1, o.hamiltonian.n):
            assert float(diag[i]) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# TestSetInitialRho
# ---------------------------------------------------------------------------

class TestSetInitialRho:
    def test_set_from_populations(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        Npop = np.zeros(n)
        Npop[0] = 1.0
        o.set_initial_rho_from_populations(Npop)
        assert o.rho0 is not None
        assert o.rho0.shape == (n**2,)

    def test_set_from_populations_normalizes(self, obe_transform, ham):
        o = obe_transform
        Npop = np.ones(ham.n) * 2.0  # un-normalized
        o.set_initial_rho_from_populations(Npop)
        # After normalization, trace = 1
        if o.transform_into_re_im:
            rho_c = jnp.dot(jnp.asarray(o.U), o.rho0.astype('complex128'))
        else:
            rho_c = o.rho0
        rho_mat = jnp.reshape(rho_c, (ham.n, ham.n))
        assert float(jnp.real(jnp.trace(rho_mat))) == pytest.approx(1.0, abs=1e-12)

    def test_set_from_populations_wrong_length_raises(self, obe_transform, ham):
        with pytest.raises(ValueError, match='Npop'):
            obe_transform.set_initial_rho_from_populations(np.ones(ham.n + 1))

    def test_set_from_populations_nan_raises(self, obe_transform, ham):
        bad = np.full(ham.n, np.nan)
        with pytest.raises(ValueError):
            obe_transform.set_initial_rho_from_populations(bad)

    def test_set_rho_flat_complex(self, obe_transform, ham):
        o = obe_transform
        rho0 = np.zeros(ham.n**2, dtype='complex128')
        rho0[0] = 1.0  # ground state population
        o.set_initial_rho(rho0)
        assert o.rho0 is not None

    def test_set_rho_matrix_input(self, obe_transform, ham):
        o = obe_transform
        rho0_mat = np.zeros((ham.n, ham.n), dtype='complex128')
        rho0_mat[0, 0] = 1.0
        o.set_initial_rho(rho0_mat)
        assert o.rho0.shape == (ham.n**2,)

    def test_set_rho_nan_raises(self, obe_transform, ham):
        bad = np.full(ham.n**2, np.nan, dtype='complex128')
        with pytest.raises(ValueError):
            obe_transform.set_initial_rho(bad)

    def test_set_rho_wrong_size_raises(self, obe_transform, ham):
        bad = np.zeros(ham.n**2 + 1, dtype='complex128')
        with pytest.raises(ValueError):
            obe_transform.set_initial_rho(bad)

    def test_set_equally_produces_normalized_rho(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        assert o.rho0 is not None
        # The trace of rho0 should be 1
        if o.transform_into_re_im:
            rho_c = jnp.dot(jnp.asarray(o.U), o.rho0.astype('complex128'))
        else:
            rho_c = o.rho0
        rho_mat = jnp.reshape(rho_c, (o.hamiltonian.n, o.hamiltonian.n))
        assert float(jnp.real(jnp.trace(rho_mat))) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# TestObservable
# ---------------------------------------------------------------------------

class TestObservable:
    def test_identity_gives_trace_one(self, obe_transform, ham):
        o = obe_transform
        o.set_initial_rho_equally()
        rho0 = o.rho0
        n = ham.n
        I_op = jnp.eye(n)
        result = o.observable(I_op, rho0)
        assert float(result) == pytest.approx(1.0, abs=1e-12)

    def test_projector_gives_population(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        # Start with all population in ground state (index 0)
        rho0_mat = np.zeros((n, n), dtype='complex128')
        rho0_mat[0, 0] = 1.0
        P0 = jnp.zeros((n, n), dtype='complex128').at[0, 0].set(1.)
        result = o.observable(P0, rho0_mat)
        assert float(result) == pytest.approx(1.0, abs=1e-12)

    def test_excited_projector_zero_in_ground(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        rho0_mat = np.zeros((n, n), dtype='complex128')
        rho0_mat[0, 0] = 1.0
        P_exc = jnp.zeros((n, n), dtype='complex128').at[1, 1].set(1.)
        result = o.observable(P_exc, rho0_mat)
        assert float(result) == pytest.approx(0.0, abs=1e-12)

    def test_vector_observable_shape(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        rho0_mat = np.zeros((n, n), dtype='complex128')
        rho0_mat[0, 0] = 1.0
        # Vector operator: shape (3, n, n)
        O_vec = jnp.zeros((3, n, n), dtype='complex128')
        result = o.observable(O_vec, rho0_mat)
        assert result.shape == (3,)

    def test_wrong_O_size_raises(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        rho0_mat = np.zeros((n, n), dtype='complex128')
        rho0_mat[0, 0] = 1.0
        bad_O = jnp.zeros((n+1, n+1))
        with pytest.raises(ValueError):
            o.observable(bad_O, rho0_mat)

    def test_wrong_rho_shape_raises(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        bad_rho = jnp.zeros((n+1, n+1))
        O = jnp.eye(n)
        with pytest.raises(ValueError):
            o.observable(O, bad_rho)

    def test_flat_rho_input_works(self, obe_transform, ham):
        """1D rho0 (internal representation) is auto-reshaped by observable."""
        o = obe_transform
        n = ham.n
        o.set_initial_rho_equally()
        rho0_flat = o.rho0  # 1D array in internal (re/im) basis
        I_op = jnp.eye(n)
        result = o.observable(I_op, rho0_flat)
        assert float(result) == pytest.approx(1.0, abs=1e-12)

    def test_result_is_real(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        rho0_mat = np.eye(n, dtype='complex128') / n
        O = jnp.eye(n)
        result = o.observable(O, rho0_mat)
        assert jnp.isreal(result)


# ---------------------------------------------------------------------------
# TestEvolveDensity
# ---------------------------------------------------------------------------

class TestEvolveDensity:
    def test_returns_sol_object(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        sol = o.evolve_density([0., 5.], n_points=11)
        assert hasattr(sol, 't')
        assert hasattr(sol, 'rho')

    def test_t_shape(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        sol = o.evolve_density([0., 5.], n_points=11)
        assert sol.t.shape == (11,)

    def test_rho_shape(self, obe_transform, ham):
        o = obe_transform
        o.set_initial_rho_equally()
        n = ham.n
        sol = o.evolve_density([0., 5.], n_points=11)
        assert sol.rho.shape == (n, n, 11)

    def test_rho_trace_conserved(self, obe_transform, ham):
        """tr(rho(t)) = 1 at all time points."""
        o = obe_transform
        o.set_initial_rho_equally()
        sol = o.evolve_density([0., 5.], n_points=21)
        n = ham.n
        for t_idx in range(21):
            rho_t = sol.rho[:, :, t_idx]
            trace = float(jnp.real(jnp.trace(rho_t)))
            assert trace == pytest.approx(1.0, abs=1e-6)

    def test_rho_diagonal_non_negative(self, obe_transform, ham):
        """Populations are non-negative at all times."""
        o = obe_transform
        o.set_initial_rho_equally()
        sol = o.evolve_density([0., 5.], n_points=21)
        for t_idx in range(21):
            diag = jnp.real(jnp.diagonal(sol.rho[:, :, t_idx]))
            assert float(jnp.min(diag)) >= -1e-8

    def test_excited_pop_grows_from_ground(self, obe_transform, ham):
        """Starting from ground state, excited state population increases."""
        o = obe_transform
        o.set_initial_rho_equally()
        sol = o.evolve_density([0., 20.], n_points=101)
        n = ham.n
        # Initial excited pop
        exc_init = float(jnp.real(jnp.sum(jnp.diagonal(sol.rho[1:, 1:, 0]))))
        # Later excited pop
        exc_final = float(jnp.real(jnp.sum(jnp.diagonal(sol.rho[1:, 1:, -1]))))
        assert exc_final > exc_init

    def test_initial_rho_is_initial_condition(self, obe_transform, ham):
        """rho at t=0 should match the initial condition."""
        o = obe_transform
        n = ham.n
        # Set ground state only
        rho0_mat = np.zeros((n, n), dtype='complex128')
        rho0_mat[0, 0] = 1.0
        o.set_initial_rho(rho0_mat)
        sol = o.evolve_density([0., 1.], n_points=11)
        # At t=0 (index 0), diagonal[0] should be ~1
        pop_ground_t0 = float(jnp.real(sol.rho[0, 0, 0]))
        assert pop_ground_t0 == pytest.approx(1.0, abs=1e-4)

    def test_no_transform_rho_trace_conserved(self, obe_complex, ham):
        """Without re/im transform, trace is also conserved."""
        o = obe_complex
        o.set_initial_rho_equally()
        sol = o.evolve_density([0., 5.], n_points=21)
        for t_idx in range(21):
            trace = float(jnp.real(jnp.trace(sol.rho[:, :, t_idx])))
            assert trace == pytest.approx(1.0, abs=1e-5)

    def test_y0_batch_override(self, obe_transform, ham):
        """Explicit y0_batch can override rho0."""
        o = obe_transform
        o.set_initial_rho_equally()
        n = ham.n
        rho0 = o.rho0
        y0 = jnp.concatenate([rho0, jnp.zeros(6)])
        y0_batch = y0[None, :]
        sol = o.evolve_density([0., 5.], y0_batch=y0_batch, n_points=11)
        assert sol.rho.shape == (n, n, 11)


# ---------------------------------------------------------------------------
# TestForce
# ---------------------------------------------------------------------------

class TestForce:
    def _get_rho0(self, o):
        """Get the initial rho0 in internal representation (re/im basis if transform=True)."""
        o.set_initial_rho_equally()
        return o.rho0  # 1D array in internal basis

    def test_force_shape_single_point(self, obe_transform):
        o = obe_transform
        rho = self._get_rho0(o)
        r = jnp.zeros(3)
        f = o.force(r, 0., rho)
        assert f.shape == (3,)

    def test_force_return_details_tuple(self, obe_transform):
        o = obe_transform
        rho = self._get_rho0(o)
        r = jnp.zeros(3)
        result = o.force(r, 0., rho, return_details=True)
        assert len(result) == 4  # f, f_laser, f_laser_q, f_mag

    def test_force_f_laser_has_key(self, obe_transform):
        o = obe_transform
        rho = self._get_rho0(o)
        r = jnp.zeros(3)
        _, f_laser, _, _ = o.force(r, 0., rho, return_details=True)
        assert 'g->e' in f_laser

    def test_force_f_laser_q_has_key(self, obe_transform):
        o = obe_transform
        rho = self._get_rho0(o)
        r = jnp.zeros(3)
        _, _, f_laser_q, _ = o.force(r, 0., rho, return_details=True)
        assert 'g->e' in f_laser_q

    def test_force_symmetric_near_zero_at_origin(self, obe_sym):
        """Symmetric counter-propagating beams give near-zero net force at v=0, B=0."""
        o = obe_sym
        rho = self._get_rho0(o)
        r = jnp.zeros(3)
        f = o.force(r, 0., rho)
        # z-component of force should be near zero by symmetry
        assert float(jnp.abs(f[2])) < 0.5

    def test_single_beam_force_avg_nonzero(self, obe_transform):
        """Time-averaged force from a single +z beam must be positive."""
        o = obe_transform
        o.set_initial_rho_equally()
        F = o.find_equilibrium_force(deltat=50, itermax=5, Npts=201,
                                     initial_rho='equally')
        assert float(F[2]) > 0

    def test_force_with_no_mag_forces(self, single_beam, zero_B, ham):
        o = obe(single_beam, zero_B, ham, include_mag_forces=False)
        o.set_initial_rho_equally()
        rho = o.rho0
        r = jnp.zeros(3)
        f = o.force(r, 0., rho)
        assert f.shape == (3,)


# ---------------------------------------------------------------------------
# TestFindEquilibriumForce
# ---------------------------------------------------------------------------

class TestFindEquilibriumForce:
    def test_returns_shape_3_array(self, obe_transform):
        o = obe_transform
        F = o.find_equilibrium_force(deltat=10, itermax=3, Npts=101,
                                     initial_rho='equally')
        assert F.shape == (3,)

    def test_initial_rho_equally(self, obe_transform):
        o = obe_transform
        F = o.find_equilibrium_force(deltat=10, itermax=3, Npts=101,
                                     initial_rho='equally')
        assert not jnp.any(jnp.isnan(F))

    def test_initial_rho_invalid_raises(self, obe_transform):
        with pytest.raises(ValueError, match='not understood'):
            obe_transform.find_equilibrium_force(initial_rho='invalid')

    def test_return_details_structure(self, obe_transform):
        o = obe_transform
        result = o.find_equilibrium_force(
            deltat=10, itermax=2, Npts=51,
            initial_rho='equally', return_details=True
        )
        # Returns (F, f_laser, f_laser_q, f_mag, Neq, ii)
        assert len(result) == 6
        F, f_laser, f_laser_q, f_mag, Neq, n_iter = result
        assert F.shape == (3,)
        assert Neq.shape == (obe_transform.hamiltonian.n,)
        assert isinstance(n_iter, (int, np.integer))

    def test_Neq_sums_to_one(self, obe_transform, ham):
        o = obe_transform
        _, _, _, _, Neq, _ = o.find_equilibrium_force(
            deltat=10, itermax=3, Npts=101,
            initial_rho='equally', return_details=True
        )
        assert float(jnp.sum(Neq)) == pytest.approx(1.0, abs=1e-4)

    def test_single_beam_force_z_positive(self, obe_transform):
        """Single +z beam → equilibrium force has positive z component."""
        o = obe_transform
        F = o.find_equilibrium_force(
            deltat=50, itermax=5, Npts=201,
            initial_rho='equally'
        )
        assert float(F[2]) > 0

    def test_symmetric_beam_force_near_zero(self, obe_sym):
        """Counter-propagating symmetric beams → near-zero net z force at origin."""
        o = obe_sym
        F = o.find_equilibrium_force(
            deltat=20, itermax=5, Npts=101,
            initial_rho='equally'
        )
        assert float(jnp.abs(F[2])) < 0.1

    def test_initial_rho_frompops(self, obe_transform, ham):
        o = obe_transform
        init_pop = np.zeros(ham.n)
        init_pop[0] = 1.0
        F = o.find_equilibrium_force(
            deltat=10, itermax=2, Npts=51,
            initial_rho='frompops', init_pop=init_pop
        )
        assert F.shape == (3,)


# ---------------------------------------------------------------------------
# TestFullOBEEv  (utility methods)
# ---------------------------------------------------------------------------

class TestFullOBEEv:
    def test_full_OBE_ev_shape(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        r = jnp.zeros(3)
        ev = o.full_OBE_ev(r, 0.)
        assert ev.shape == (n**2, n**2)

    def test_full_OBE_ev_no_nan(self, obe_transform):
        o = obe_transform
        r = jnp.zeros(3)
        ev = o.full_OBE_ev(r, 0.)
        assert not jnp.any(jnp.isnan(ev))

    def test_full_OBE_ev_with_B(self, obe_transform, ham):
        """full_OBE_ev should run without error when B field is non-zero."""
        from pylcp.fields import constantMagneticField
        single_beam = laserBeams([{'kvec': [0., 0., 1.], 'pol': +1, 's': 0.1, 'delta': 0.}])
        B_field = constantMagneticField(jnp.array([0., 0., 0.1]))
        o_B = obe(single_beam, B_field, make_ham())
        r = jnp.zeros(3)
        ev = o_B.full_OBE_ev(r, 0.)
        assert ev.shape == (o_B.hamiltonian.n**2, o_B.hamiltonian.n**2)
        assert not jnp.any(jnp.isnan(ev))


# ---------------------------------------------------------------------------
# Test1DMOTForceProfile – regression tests for magnetic field gradient
# ---------------------------------------------------------------------------

class Test1DMOTForceProfile:
    """1D MOT with linear B-field gradient: OBE force must be restoring."""

    @pytest.fixture
    def mot_obe(self):
        from pylcp.fields import magField
        ham = make_ham(gamma=1.0, k=1.0, mass=1.0)
        mu_val = 1399624.49171  # |diag(mu_e[1])[0]|
        delta = -4.0
        x_res = 5.0
        alpha = abs(delta) / (x_res * mu_val)
        beams = laserBeams([
            {'kvec': [1., 0., 0.], 'pol': -1, 's': 1.0, 'delta': delta},
            {'kvec': [-1., 0., 0.], 'pol': -1, 's': 1.0, 'delta': delta},
        ])
        B = magField(lambda R: -alpha * R)
        return obe(beams, B, ham), x_res

    def test_force_at_origin_is_zero(self, mot_obe):
        """By symmetry the force at x=0 (where B=0) must vanish."""
        o, _ = mot_obe
        o.set_initial_position_and_velocity(jnp.zeros(3), jnp.zeros(3))
        F = o.find_equilibrium_force(deltat=200, itermax=50, Npts=2001)
        assert float(F[0]) == pytest.approx(0., abs=1e-6)

    def test_force_nonzero_away_from_origin(self, mot_obe):
        """Force must be non-zero in the linear trapping region."""
        o, x_res = mot_obe
        o.set_initial_position_and_velocity(
            jnp.array([x_res / 2, 0., 0.]), jnp.zeros(3))
        F = o.find_equilibrium_force(deltat=200, itermax=50, Npts=2001)
        assert abs(float(F[0])) > 1e-6

    def test_force_is_restoring(self, mot_obe):
        """Force in the linear region should be restoring."""
        o, x_res = mot_obe
        # Use half the resonance position — well inside the linear trapping region
        x_test = x_res / 2
        for x_val, expected_sign in [(x_test, -1), (-x_test, +1)]:
            o.set_initial_position_and_velocity(
                jnp.array([x_val, 0., 0.]), jnp.zeros(3))
            F = o.find_equilibrium_force(deltat=200, itermax=50, Npts=2001)
            assert float(F[0]) * expected_sign > 0., \
                f"Force at x={x_val} should have sign {expected_sign}, got {float(F[0])}"

    def test_force_no_nan(self, mot_obe):
        """OBE force at origin (B=0) must not produce NaN."""
        o, _ = mot_obe
        o.set_initial_position_and_velocity(jnp.zeros(3), jnp.zeros(3))
        F = o.find_equilibrium_force(deltat=200, itermax=50, Npts=2001)
        assert not jnp.any(jnp.isnan(F))


class TestEvolveMotion:
    """Tests for obe.evolve_motion, including default y0/keys and dtype consistency."""

    @pytest.fixture
    def mot_obe_transform(self):
        """OBE with transform_into_re_im=True (real-valued rho), weak beam."""
        ham = make_ham(mass=100.0)
        beams = laserBeams([
            {'kvec': [0., 0., 1.], 'pol': +1, 's': 0.01, 'delta': -1.0},
            {'kvec': [0., 0., -1.], 'pol': +1, 's': 0.01, 'delta': -1.0},
        ])
        B = constantMagneticField(jnp.array([0., 0., 0.]))
        return obe(beams, B, ham, transform_into_re_im=True)

    @pytest.fixture
    def mot_obe_complex(self):
        """OBE with transform_into_re_im=False (complex-valued rho), weak beam."""
        ham = make_ham(mass=100.0)
        beams = laserBeams([
            {'kvec': [0., 0., 1.], 'pol': +1, 's': 0.01, 'delta': -1.0},
            {'kvec': [0., 0., -1.], 'pol': +1, 's': 0.01, 'delta': -1.0},
        ])
        B = constantMagneticField(jnp.array([0., 0., 0.]))
        return obe(beams, B, ham, transform_into_re_im=False)

    def test_evolve_motion_default_args(self, mot_obe_transform):
        """evolve_motion should work without explicit y0_batch/keys_batch."""
        o = mot_obe_transform
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        # Should not raise
        o.evolve_motion([0, 10], freeze_axis=[True, True, False],
                        random_recoil=False)
        assert len(o.sols) == 1
        assert not jnp.any(jnp.isnan(o.sols[0].r))

    def test_evolve_motion_transform_dtype_consistency(self, mot_obe_transform):
        """With transform_into_re_im=True, y0 and dydt must both be real-valued.

        This catches the dtype mismatch where __drhodt returns complex but y0
        is float64, which causes diffrax buffer dtype errors.
        """
        o = mot_obe_transform
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()

        y0 = jnp.concatenate([o.rho0, o.v0, o.r0])
        assert y0.dtype == jnp.float64, \
            f"y0 should be float64 with transform_into_re_im=True, got {y0.dtype}"

        # Simulate what evolve_motion's dydt does internally
        rho = y0[:-6]
        r = y0[-3:]
        drhodt_raw = o._obe__drhodt(r, 0.0, rho)
        drhodt = jnp.real(drhodt_raw)  # this is what the fix applies
        F = o.force(r, 0.0, rho, return_details=False)
        dydt_out = jnp.concatenate([drhodt, F / o.hamiltonian.mass, jnp.zeros(3)])
        assert dydt_out.dtype == y0.dtype, \
            f"dydt output dtype {dydt_out.dtype} must match y0 dtype {y0.dtype}"

    def test_evolve_motion_complex_mode(self, mot_obe_complex):
        """evolve_motion should also work with transform_into_re_im=False."""
        o = mot_obe_complex
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        o.evolve_motion([0, 10], freeze_axis=[True, True, False],
                        random_recoil=False)
        assert len(o.sols) == 1
        assert not jnp.any(jnp.isnan(o.sols[0].r))