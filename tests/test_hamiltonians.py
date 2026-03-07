"""
Tests for pylcp/hamiltonians/__init__.py and pylcp/hamiltonians/XFmolecules.py
"""
import pytest
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import pylcp.hamiltonians as ham
from pylcp.hamiltonians import XFmolecules


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def is_hermitian(M, atol=1e-10):
    M = np.array(M)
    return np.allclose(M, np.conj(M.T), atol=atol)


def is_spherical_rank1_mu(M, atol=1e-10):
    """Verify the rank-1 spherical tensor property:
      mu_q[1] (q=0) is Hermitian
      conj(mu_q[2].T) == -mu_q[0]  i.e. (mu_{+1})† = -mu_{-1}
    """
    M0 = np.array(M[0])
    M1 = np.array(M[1])
    M2 = np.array(M[2])
    return (np.allclose(M1, np.conj(M1.T), atol=atol) and
            np.allclose(np.conj(M2.T), -M0, atol=atol))


def is_unitary(U, atol=1e-10):
    U = np.array(U)
    return np.allclose(np.conj(U.T) @ U, np.eye(U.shape[0]), atol=atol)


# ---------------------------------------------------------------------------
# TestUncoupledIndex
# ---------------------------------------------------------------------------

class TestUncoupledIndex:
    def test_first_index_is_zero(self):
        # mJ=-1/2, mI=-1/2 → first basis state
        assert ham.uncoupled_index(0.5, 0.5, -0.5, -0.5) == 0

    def test_last_index(self):
        assert ham.uncoupled_index(0.5, 0.5, 0.5, 0.5) == 3

    def test_all_indices_distinct(self):
        J, I = 0.5, 0.5
        indices = set()
        for mJ in np.arange(-J, J + 1, 1):
            for mI in np.arange(-I, I + 1, 1):
                indices.add(ham.uncoupled_index(J, I, mJ, mI))
        assert len(indices) == 4

    def test_integer_spin(self):
        # J=1, I=0: 3 states
        indices = set()
        for mJ in np.arange(-1, 2, 1):
            indices.add(ham.uncoupled_index(1, 0, mJ, 0))
        assert len(indices) == 3


# ---------------------------------------------------------------------------
# TestCoupledIndex
# ---------------------------------------------------------------------------

class TestCoupledIndex:
    def test_F0_is_zero(self):
        assert ham.coupled_index(0, 0, 0) == 0

    def test_F1_mF0_with_Fmin0(self):
        # F=0 contributes 1 state, then F=1,mF=-1,0,+1 → mF=0 is index 2
        assert ham.coupled_index(1, 0, 0) == 2

    def test_invalid_mF_raises(self):
        with pytest.raises(ValueError):
            ham.coupled_index(1, 2, 0)  # |mF|=2 > F=1

    def test_all_F1_indices_distinct(self):
        indices = {ham.coupled_index(1, m, 1) for m in [-1, 0, 1]}
        assert len(indices) == 3

    def test_sequential_for_Fmin0(self):
        # F=0,1: coupled_index(0,0,0)=0, (1,-1,0)=1, (1,0,0)=2, (1,1,0)=3
        assert ham.coupled_index(0, 0, 0) == 0
        assert ham.coupled_index(1, -1, 0) == 1
        assert ham.coupled_index(1, 1, 0) == 3


# ---------------------------------------------------------------------------
# TestDqijNorm
# ---------------------------------------------------------------------------

class TestDqijNorm:
    def test_shape_preserved(self):
        rng = np.random.default_rng(0)
        d = rng.random((3, 4, 5))
        assert ham.dqij_norm(d).shape == d.shape

    def test_column_norms_are_one(self):
        rng = np.random.default_rng(1)
        d = rng.random((3, 3, 5))
        d_n = ham.dqij_norm(d)
        for jj in range(d.shape[2]):
            assert np.linalg.norm(d_n[:, :, jj]) == pytest.approx(1.0, abs=1e-12)

    def test_zero_column_stays_zero(self):
        d = np.zeros((3, 2, 3))
        d[:, :, 1] = 1.0
        d_n = ham.dqij_norm(d)
        assert np.allclose(d_n[:, :, 0], 0.)
        assert np.allclose(d_n[:, :, 2], 0.)

    def test_already_normalized_unchanged(self):
        d = np.zeros((3, 1, 1))
        d[1, 0, 0] = 1.0  # single unit entry
        d_n = ham.dqij_norm(d)
        assert d_n[1, 0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestSingleF
# ---------------------------------------------------------------------------

class TestSingleF:
    def test_H0_shape_F1(self):
        H0, _ = ham.singleF(F=1, gF=1)
        assert H0.shape == (3, 3)

    def test_mu_q_shape_F1(self):
        _, mu_q = ham.singleF(F=1, gF=1)
        assert mu_q.shape == (3, 3, 3)

    def test_H0_is_zero(self):
        H0, _ = ham.singleF(F=1, gF=1)
        assert jnp.allclose(H0, jnp.zeros((3, 3)))

    def test_H0_shape_F2(self):
        H0, _ = ham.singleF(F=2, gF=1)
        assert H0.shape == (5, 5)

    def test_mu_q_hermitian(self):
        _, mu_q = ham.singleF(F=2, gF=1)
        assert is_spherical_rank1_mu(mu_q)

    def test_F_half_shape(self):
        H0, mu_q = ham.singleF(F=0.5, gF=2)
        assert H0.shape == (2, 2)
        assert mu_q.shape == (3, 2, 2)

    def test_return_basis_shape(self):
        _, _, basis = ham.singleF(F=1, gF=1, return_basis=True)
        assert basis.shape == (3, 2)

    def test_return_basis_F_values(self):
        _, _, basis = ham.singleF(F=1, gF=1, return_basis=True)
        assert np.all(basis[:, 0] == 1.0)

    def test_gF_zero_gives_zero_mu(self):
        _, mu_q = ham.singleF(F=1, gF=0)
        assert jnp.allclose(mu_q, jnp.zeros((3, 3, 3)))

    def test_q0_component_is_diagonal(self):
        """mu_q[1] (q=0, pi transition) should be diagonal."""
        _, mu_q = ham.singleF(F=1, gF=1)
        off = np.array(mu_q[1]) - np.diag(np.diagonal(np.array(mu_q[1])))
        assert np.allclose(off, 0., atol=1e-12)

    def test_q0_diagonal_scales_with_gF(self):
        _, mu_q_1 = ham.singleF(F=1, gF=1)
        _, mu_q_2 = ham.singleF(F=1, gF=2)
        assert jnp.allclose(mu_q_2[1], 2 * mu_q_1[1], atol=1e-12)


# ---------------------------------------------------------------------------
# TestHyperfineUncoupled
# ---------------------------------------------------------------------------

class TestHyperfineUncoupled:
    def test_shape_J_half_I_half(self):
        H0, mu_q = ham.hyperfine_uncoupled(J=0.5, I=0.5, gJ=2., gI=0., Ahfs=1.)
        assert H0.shape == (4, 4)
        assert mu_q.shape == (3, 4, 4)

    def test_shape_J1_I1(self):
        H0, mu_q = ham.hyperfine_uncoupled(J=1., I=1., gJ=2., gI=0., Ahfs=1.)
        assert H0.shape == (9, 9)
        assert mu_q.shape == (3, 9, 9)

    def test_H0_hermitian(self):
        H0, _ = ham.hyperfine_uncoupled(J=0.5, I=0.5, gJ=2., gI=0., Ahfs=1.)
        assert is_hermitian(H0)

    def test_mu_q_hermitian(self):
        _, mu_q = ham.hyperfine_uncoupled(J=0.5, I=0.5, gJ=2., gI=0., Ahfs=1.)
        assert is_spherical_rank1_mu(mu_q)

    def test_return_basis_shape(self):
        _, _, basis = ham.hyperfine_uncoupled(
            J=0.5, I=0.5, gJ=2., gI=0., Ahfs=1., return_basis=True
        )
        assert basis.shape[1] == 4  # 4 states


# ---------------------------------------------------------------------------
# TestHyperfineCoupled
# ---------------------------------------------------------------------------

class TestHyperfineCoupled:
    def test_shape_J_half_I_half(self):
        H0, mu_q = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2., gI=0., Ahfs=1.)
        assert H0.shape == (4, 4)
        assert mu_q.shape == (3, 4, 4)

    def test_shape_J_half_I_3half(self):
        H0, _ = ham.hyperfine_coupled(J=0.5, I=1.5, gJ=2., gI=0., Ahfs=1.)
        assert H0.shape == (8, 8)

    def test_H0_hermitian(self):
        H0, _ = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2., gI=0., Ahfs=1.)
        assert is_hermitian(H0)

    def test_mu_q_hermitian(self):
        _, mu_q = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2., gI=0., Ahfs=1.)
        assert is_spherical_rank1_mu(mu_q)

    def test_diagonal_energies_J_half_I_half(self):
        """J=I=1/2, Ahfs=1: E(F=0)=-3/4 (×1), E(F=1)=+1/4 (×3)."""
        H0, _ = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2., gI=0., Ahfs=1.)
        diag = np.sort(np.real(np.diagonal(np.array(H0))))
        assert diag[0] == pytest.approx(-0.75, abs=1e-10)
        assert diag[1] == pytest.approx(0.25, abs=1e-10)
        assert diag[2] == pytest.approx(0.25, abs=1e-10)
        assert diag[3] == pytest.approx(0.25, abs=1e-10)

    def test_hyperfine_splitting_scales_with_Ahfs(self):
        H0_1, _ = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2., gI=0., Ahfs=1.)
        H0_2, _ = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2., gI=0., Ahfs=2.)
        d1 = np.sort(np.real(np.diagonal(np.array(H0_1))))
        d2 = np.sort(np.real(np.diagonal(np.array(H0_2))))
        split1 = d1[-1] - d1[0]
        split2 = d2[-1] - d2[0]
        assert split2 == pytest.approx(2 * split1, abs=1e-10)

    def test_negative_Ahfs_flips_levels(self):
        """Negative Ahfs: F=0 should be above F=1 (tests Bhfs != 0 fix)."""
        H0, _ = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2., gI=0., Ahfs=-1.)
        diag = np.sort(np.real(np.diagonal(np.array(H0))))
        assert diag[0] == pytest.approx(-0.25, abs=1e-10)
        assert diag[3] == pytest.approx(0.75, abs=1e-10)

    def test_return_basis(self):
        _, _, basis = ham.hyperfine_coupled(
            J=0.5, I=0.5, gJ=2., gI=0., Ahfs=1., return_basis=True
        )
        assert basis.shape[1] == 4

    def test_Bhfs_nonzero_changes_energies(self):
        """Non-zero Bhfs should change the eigenspectrum."""
        H0_no, _ = ham.hyperfine_coupled(J=1., I=1., gJ=2., gI=0., Ahfs=1., Bhfs=0.)
        H0_b, _ = ham.hyperfine_coupled(J=1., I=1., gJ=2., gI=0., Ahfs=1., Bhfs=0.1)
        ev_no = np.sort(np.linalg.eigvalsh(np.array(H0_no)))
        ev_b = np.sort(np.linalg.eigvalsh(np.array(H0_b)))
        assert not np.allclose(ev_no, ev_b, atol=1e-6)

    def test_Bhfs_negative_accepted(self):
        """Negative Bhfs should not be silently skipped (tests Bhfs != 0 fix)."""
        H0_pos, _ = ham.hyperfine_coupled(J=1., I=1., gJ=2., gI=0., Ahfs=1., Bhfs=0.1)
        H0_neg, _ = ham.hyperfine_coupled(J=1., I=1., gJ=2., gI=0., Ahfs=1., Bhfs=-0.1)
        # The matrices should differ (not both same as Bhfs=0)
        assert not np.allclose(np.array(H0_pos), np.array(H0_neg), atol=1e-10)


# ---------------------------------------------------------------------------
# TestFineStructureUncoupled
# ---------------------------------------------------------------------------

class TestFineStructureUncoupled:
    def test_shape_L1_S_half_I_half(self):
        H0, mu_q = ham.fine_structure_uncoupled(
            L=1, S=0.5, I=0.5, xi=1., a_c=0., a_orb=0., a_dip=0.,
            gL=1., gS=2., gI=0.
        )
        # (2*1+1)*(2*0.5+1)*(2*0.5+1) = 3*2*2 = 12 states
        assert H0.shape == (12, 12)
        assert mu_q.shape == (3, 12, 12)

    def test_H0_hermitian_L1(self):
        H0, _ = ham.fine_structure_uncoupled(
            L=1, S=0.5, I=0., xi=1., a_c=0.1, a_orb=0., a_dip=0.,
            gL=1., gS=2., gI=0.
        )
        assert is_hermitian(H0)

    def test_H0_hermitian_L2_with_a_dip(self):
        """L=2 triggers section-IV off-diagonals; Hermiticity verifies the bug fix."""
        H0, _ = ham.fine_structure_uncoupled(
            L=2, S=0.5, I=0., xi=1., a_c=0., a_orb=0.5, a_dip=1.,
            gL=1., gS=2., gI=0.
        )
        assert is_hermitian(H0)

    def test_mu_q_hermitian_L1(self):
        _, mu_q = ham.fine_structure_uncoupled(
            L=1, S=0.5, I=0., xi=0., a_c=0., a_orb=0., a_dip=0.,
            gL=1., gS=2., gI=0.
        )
        assert is_spherical_rank1_mu(mu_q)

    def test_all_zero_couplings_gives_zero_H0(self):
        H0, _ = ham.fine_structure_uncoupled(
            L=1, S=0.5, I=0., xi=0., a_c=0., a_orb=0., a_dip=0.,
            gL=1., gS=2., gI=0.
        )
        assert jnp.allclose(H0, jnp.zeros_like(H0))

    def test_return_basis_length(self):
        _, _, basis = ham.fine_structure_uncoupled(
            L=1, S=0.5, I=0., xi=1., a_c=0., a_orb=0., a_dip=0.,
            gL=1., gS=2., gI=0., return_basis=True
        )
        # 3 * 2 = 6 states (I=0)
        assert len(basis) == 6

    def test_L2_section_IV_hermitian_all_terms(self):
        """Exercise section IV for L=2 with both a_orb and a_dip nonzero."""
        H0, _ = ham.fine_structure_uncoupled(
            L=2, S=0.5, I=0.5, xi=0.5, a_c=0.1, a_orb=0.3, a_dip=0.2,
            gL=1., gS=2., gI=0.001
        )
        assert is_hermitian(H0)


# ---------------------------------------------------------------------------
# TestDqijTwoFineStructureManifoldsUncoupled
# ---------------------------------------------------------------------------

class TestDqijTwoFineStructureManifoldsUncoupled:
    def _s_to_p_bases(self):
        """S-state (L=0) → P-state (L=1) with S=1/2, I=0."""
        basis_g = [(0, 0.5, 0), (0, -0.5, 0)]
        basis_e = [(-1, 0.5, 0), (-1, -0.5, 0),
                   ( 0, 0.5, 0), ( 0, -0.5, 0),
                   ( 1, 0.5, 0), ( 1, -0.5, 0)]
        return basis_g, basis_e

    def test_shape(self):
        basis_g, basis_e = self._s_to_p_bases()
        d_q = ham.dqij_two_fine_stucture_manifolds_uncoupled(basis_g, basis_e)
        assert d_q.shape == (3, 2, 6)

    def test_no_nan(self):
        basis_g, basis_e = self._s_to_p_bases()
        d_q = ham.dqij_two_fine_stucture_manifolds_uncoupled(basis_g, basis_e)
        assert not jnp.any(jnp.isnan(d_q))

    def test_selection_rules(self):
        """d_q[kk, ii, jj] nonzero ⟺ mL_g - mL_e = q, mS_g = mS_e, mI_g = mI_e."""
        basis_g, basis_e = self._s_to_p_bases()
        d_q = np.array(
            ham.dqij_two_fine_stucture_manifolds_uncoupled(basis_g, basis_e)
        )
        for kk, q in enumerate([-1, 0, 1]):
            for ii, (mL, mS, mI) in enumerate(basis_g):
                for jj, (mLp, mSp, mIp) in enumerate(basis_e):
                    expect = (mL == mLp + q) and (mS == mSp) and (mI == mIp)
                    assert (abs(d_q[kk, ii, jj]) > 1e-12) == expect

    def test_same_basis_identity_coupling(self):
        """If both bases share the same L, mL=mLp=0, q=0 coupling is 1."""
        basis = [(0, 0.5, 0)]
        d_q = np.array(
            ham.dqij_two_fine_stucture_manifolds_uncoupled(basis, basis)
        )
        assert d_q[1, 0, 0] == 1.0  # q=0 (index 1)


# ---------------------------------------------------------------------------
# TestDqijTwoHyperfineManifolds
# ---------------------------------------------------------------------------

class TestDqijTwoHyperfineManifolds:
    def test_shape_J0_Jp1_I0(self):
        d_q = ham.dqij_two_hyperfine_manifolds(J=0, Jp=1, I=0)
        assert d_q.shape == (3, 1, 3)

    def test_shape_J_half_Jp_half_I_half(self):
        d_q = ham.dqij_two_hyperfine_manifolds(J=0.5, Jp=0.5, I=0.5)
        assert d_q.shape == (3, 4, 4)

    def test_no_nan(self):
        d_q = ham.dqij_two_hyperfine_manifolds(J=0, Jp=1, I=0)
        assert not jnp.any(jnp.isnan(d_q))

    def test_normalized_columns(self):
        d_q = np.array(ham.dqij_two_hyperfine_manifolds(J=0, Jp=1, I=0, normalize=True))
        for jj in range(d_q.shape[2]):
            assert np.linalg.norm(d_q[:, :, jj]) == pytest.approx(1.0, abs=1e-10)

    def test_unnormalized_not_unit(self):
        """Without normalization some columns may differ from unit norm."""
        d_q = np.array(
            ham.dqij_two_hyperfine_manifolds(J=0.5, Jp=0.5, I=0.5, normalize=False)
        )
        norms = [np.linalg.norm(d_q[:, :, jj]) for jj in range(d_q.shape[2])]
        # At least some columns should not be exactly 1.0 before normalization
        assert not all(n == pytest.approx(1.0, abs=1e-6) for n in norms)

    def test_return_basis(self):
        _, basis_g, basis_e = ham.dqij_two_hyperfine_manifolds(
            J=0.5, Jp=0.5, I=0.5, return_basis=True
        )
        assert basis_g.shape[0] == 4
        assert basis_e.shape[0] == 4


# ---------------------------------------------------------------------------
# TestDqijTwoBareHyperfine
# ---------------------------------------------------------------------------

class TestDqijTwoBareHyperfine:
    def test_shape_F0_Fp1(self):
        d_q = ham.dqij_two_bare_hyperfine(0, 1)
        assert d_q.shape == (3, 1, 3)

    def test_shape_F1_Fp2(self):
        d_q = ham.dqij_two_bare_hyperfine(1, 2)
        assert d_q.shape == (3, 3, 5)

    def test_no_nan(self):
        assert not jnp.any(jnp.isnan(ham.dqij_two_bare_hyperfine(1, 2)))

    def test_column_norms_unit_F0_Fp1(self):
        d_q = np.array(ham.dqij_two_bare_hyperfine(0, 1, normalize=True))
        for jj in range(3):
            assert np.linalg.norm(d_q[:, :, jj]) == pytest.approx(1.0, abs=1e-10)

    def test_selection_rules_F0_Fp1(self):
        """F=0→F'=1: each q-component has exactly one nonzero entry."""
        d_q = np.array(ham.dqij_two_bare_hyperfine(0, 1, normalize=False))
        for q_idx in range(3):
            assert np.count_nonzero(d_q[q_idx]) == 1

    def test_dtype_complex(self):
        d_q = ham.dqij_two_bare_hyperfine(0, 1)
        assert jnp.issubdtype(d_q.dtype, jnp.complexfloating)


# ---------------------------------------------------------------------------
# TestXstate (XFmolecules)
# ---------------------------------------------------------------------------

class TestXstate:
    def test_shape_N0_I_half(self):
        H0, mu_p, U = XFmolecules.Xstate(N=0, I=0.5)
        # N=0, S=1/2 → J=1/2 → F=0,1 → 4 states
        assert H0.shape == (4, 4)
        assert mu_p.shape == (3, 4, 4)
        assert U.shape == (4, 4)

    def test_shape_N1_I_half(self):
        H0, mu_p, U = XFmolecules.Xstate(N=1, I=0.5)
        # N=1, S=1/2 → J=1/2,3/2 → 4+8 = 12 states
        assert H0.shape == (12, 12)
        assert mu_p.shape == (3, 12, 12)

    def test_H0_hermitian(self):
        H0, _, _ = XFmolecules.Xstate(N=1, I=0.5, b=100., c=30.)
        assert is_hermitian(H0)

    def test_mu_p_hermitian(self):
        _, mu_p, _ = XFmolecules.Xstate(N=1, I=0.5)
        assert is_spherical_rank1_mu(mu_p)

    def test_U_unitary(self):
        _, _, U = XFmolecules.Xstate(N=1, I=0.5, b=100., c=30.)
        assert is_unitary(U)

    def test_no_nan(self):
        H0, mu_p, U = XFmolecules.Xstate(
            N=1, I=0.5, B=10000., b=100., c=30., gamma=40.
        )
        assert not jnp.any(jnp.isnan(H0))
        assert not jnp.any(jnp.isnan(mu_p))

    def test_return_basis_length(self):
        _, _, _, basis = XFmolecules.Xstate(N=1, I=0.5, return_basis=True)
        assert len(basis) == 12

    def test_diagonal_without_hyperfine(self):
        """All coupling constants zero → H0 should be all zeros (no splitting)."""
        H0, _, _ = XFmolecules.Xstate(N=0, I=0.5)
        assert jnp.allclose(jnp.real(H0), jnp.zeros_like(jnp.real(H0)), atol=1e-12)

    def test_multi_N(self):
        H0, mu_p, U = XFmolecules.Xstate(N=[0, 1], I=0.5, B=10000.)
        # N=0: 4 states, N=1: 12 states → 16 total
        assert H0.shape == (16, 16)


# ---------------------------------------------------------------------------
# TestAstate (XFmolecules)
# ---------------------------------------------------------------------------

class TestAstate:
    def test_shape_J_half_I_half_single_P(self):
        H0, mu_p = XFmolecules.Astate(J=0.5, I=0.5, P=+1)
        # J=1/2, I=1/2 → F=0,1 → 4 states
        assert H0.shape == (4, 4)
        assert mu_p.shape == (3, 4, 4)

    def test_H0_hermitian(self):
        H0, _ = XFmolecules.Astate(J=0.5, I=0.5, P=+1, b=5., c=2., a=3.)
        assert is_hermitian(H0)

    def test_mu_p_hermitian(self):
        _, mu_p = XFmolecules.Astate(J=0.5, I=0.5, P=+1)
        assert is_spherical_rank1_mu(mu_p)

    def test_no_nan(self):
        H0, mu_p = XFmolecules.Astate(J=0.5, I=0.5, P=+1, a=3., b=5., c=2.)
        assert not jnp.any(jnp.isnan(H0))
        assert not jnp.any(jnp.isnan(mu_p))

    def test_return_basis(self):
        _, _, basis = XFmolecules.Astate(J=0.5, I=0.5, P=+1, return_basis=True)
        assert len(basis) == 4

    def test_two_J_values(self):
        H0, mu_p = XFmolecules.Astate(J=[0.5, 1.5], I=0.5, P=+1)
        # J=1/2 → 4 states, J=3/2 → 8 states → 12 total
        assert H0.shape == (12, 12)

    def test_two_P_values(self):
        H0, mu_p = XFmolecules.Astate(J=0.5, I=0.5, P=[+1, -1])
        # Two parities × 4 states = 8 states
        assert H0.shape == (8, 8)


# ---------------------------------------------------------------------------
# TestDipoleXandAstates (XFmolecules)
# ---------------------------------------------------------------------------

class TestDipoleXandAstates:
    @pytest.fixture(scope='class')
    def bases(self):
        _, _, U_X, Xbasis = XFmolecules.Xstate(N=1, I=0.5, return_basis=True)
        _, _, Abasis = XFmolecules.Astate(J=0.5, I=0.5, P=+1, return_basis=True)
        return Xbasis, Abasis, np.array(U_X)

    def test_output_shape(self, bases):
        Xbasis, Abasis, U_X = bases
        dijq = XFmolecules.dipoleXandAstates(Xbasis, Abasis, UX=U_X)
        assert dijq.shape == (3, 12, 4)

    def test_no_nan(self, bases):
        Xbasis, Abasis, U_X = bases
        dijq = XFmolecules.dipoleXandAstates(Xbasis, Abasis, UX=U_X)
        assert not jnp.any(jnp.isnan(dijq))

    def test_return_intermediate_length(self, bases):
        Xbasis, Abasis, U_X = bases
        result = XFmolecules.dipoleXandAstates(
            Xbasis, Abasis, UX=U_X, return_intermediate=True
        )
        assert len(result) == 6  # dijq, T_ap, T_ba, intdijq, intbasis_ap, intbasis_ba

    def test_no_UX_uses_identity(self, bases):
        Xbasis, Abasis, _ = bases
        # Passing UX=[] (default identity) should not raise
        dijq = XFmolecules.dipoleXandAstates(Xbasis, Abasis)
        assert dijq.shape == (3, 12, 4)