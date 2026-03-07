import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from pylcp.hamiltonian import hamiltonian
import pylcp.hamiltonians as hamiltonians


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_two_level(n_g=1, n_e=1):
    """Minimal two-level Hamiltonian (scalar ground/excited, no structure)."""
    H0_g = jnp.zeros((n_g, n_g))
    H0_e = jnp.zeros((n_e, n_e))
    mu_g = jnp.zeros((3, n_g, n_g))
    mu_e = jnp.zeros((3, n_e, n_e))
    d_q = jnp.ones((3, n_g, n_e), dtype=jnp.complex128) / jnp.sqrt(3.)
    return H0_g, H0_e, mu_g, mu_e, d_q


def make_F0_to_F1():
    """F=0 -> F'=1 two-level system, the canonical MOT example."""
    H0_g, mu_g = hamiltonians.singleF(F=0, gF=0)
    H0_e, mu_e = hamiltonians.singleF(F=1, gF=1)
    d_q = hamiltonians.dqij_two_bare_hyperfine(0, 1)
    return H0_g, H0_e, mu_g, mu_e, d_q


# ---------------------------------------------------------------------------
# hamiltonian.block inner class
# ---------------------------------------------------------------------------

class TestBlock:
    def test_scalar_block(self):
        b = hamiltonian.block('test', jnp.eye(2))
        assert b.n == 2
        assert b.m == 2
        assert b.diagonal

    def test_non_diagonal_block(self):
        M = jnp.array([[0., 1.], [0., 0.]])
        b = hamiltonian.block('test', M)
        assert not b.diagonal

    def test_non_square_block(self):
        M = jnp.ones((2, 3))
        b = hamiltonian.block('test', M)
        assert b.n == 2
        assert b.m == 3
        assert not b.diagonal

    def test_return_block_in_place(self):
        M = jnp.eye(2)
        b = hamiltonian.block('test', M)
        placed = b.return_block_in_place(1, 1, 4)
        assert placed.shape == (4, 4)
        assert float(jnp.real(placed[1, 1])) == pytest.approx(1.0)
        assert float(jnp.real(placed[0, 0])) == pytest.approx(0.0)

    def test_repr(self):
        b = hamiltonian.block('test', jnp.eye(3))
        assert 'test' in repr(b)
        assert '3' in repr(b)


# ---------------------------------------------------------------------------
# hamiltonian.vector_block inner class
# ---------------------------------------------------------------------------

class TestVectorBlock:
    def test_shape(self):
        M = jnp.zeros((3, 2, 2))
        vb = hamiltonian.vector_block('mu', M)
        assert vb.n == 2
        assert vb.m == 2

    def test_diagonal_detection(self):
        M = jnp.zeros((3, 2, 2))
        M = M.at[1].set(jnp.eye(2))
        vb = hamiltonian.vector_block('mu', M)
        assert vb.diagonal

    def test_return_block_in_place(self):
        M = jnp.zeros((3, 2, 2))
        M = M.at[0].set(jnp.eye(2))
        vb = hamiltonian.vector_block('mu', M)
        placed = vb.return_block_in_place(0, 0, 4)
        assert placed.shape == (3, 4, 4)
        assert float(jnp.abs(placed[0, 0, 0])) == pytest.approx(1.0)
        assert float(jnp.abs(placed[0, 2, 2])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# add_H_0_block
# ---------------------------------------------------------------------------

class TestAddH0Block:
    def test_single_block(self):
        ham = hamiltonian()
        ham.add_H_0_block('g', jnp.zeros((2, 2)))
        assert ham.n == 2
        assert 'g' in ham.state_labels

    def test_two_blocks(self):
        ham = hamiltonian()
        ham.add_H_0_block('g', jnp.zeros((2, 2)))
        ham.add_H_0_block('e', jnp.zeros((3, 3)))
        assert ham.n == 5
        assert ham.blocks.shape == (2, 2)

    def test_non_square_raises(self):
        ham = hamiltonian()
        with pytest.raises(ValueError):
            ham.add_H_0_block('g', jnp.zeros((2, 3)))

    def test_duplicate_raises(self):
        ham = hamiltonian()
        ham.add_H_0_block('g', jnp.zeros((2, 2)))
        with pytest.raises(ValueError):
            ham.add_H_0_block('g', jnp.zeros((2, 2)))

    def test_set_mass(self):
        ham = hamiltonian()
        ham.set_mass(87.)
        assert ham.mass == pytest.approx(87.)


# ---------------------------------------------------------------------------
# add_mu_q_block
# ---------------------------------------------------------------------------

class TestAddMuQBlock:
    def test_mu_before_H0(self):
        ham = hamiltonian()
        mu = jnp.zeros((3, 2, 2))
        ham.add_mu_q_block('g', mu)
        assert ham.n == 2

    def test_mu_after_H0(self):
        ham = hamiltonian()
        ham.add_H_0_block('g', jnp.zeros((2, 2)))
        ham.add_mu_q_block('g', jnp.zeros((3, 2, 2)))
        assert ham.n == 2

    def test_wrong_shape_raises(self):
        ham = hamiltonian()
        with pytest.raises(ValueError):
            ham.add_mu_q_block('g', jnp.zeros((2, 2, 2)))  # first dim must be 3

    def test_duplicate_mu_raises(self):
        ham = hamiltonian()
        ham.add_mu_q_block('g', jnp.zeros((3, 2, 2)))
        with pytest.raises(ValueError):
            ham.add_mu_q_block('g', jnp.zeros((3, 2, 2)))


# ---------------------------------------------------------------------------
# add_d_q_block
# ---------------------------------------------------------------------------

class TestAddDQBlock:
    def setup_method(self):
        self.ham = hamiltonian()
        self.ham.add_H_0_block('g', jnp.zeros((1, 1)))
        self.ham.add_H_0_block('e', jnp.zeros((3, 3)))

    def test_add_d_q(self):
        d_q = hamiltonians.dqij_two_bare_hyperfine(0, 1)
        self.ham.add_d_q_block('g', 'e', d_q)
        assert 'g->e' in self.ham.laser_keys

    def test_wrong_size_raises(self):
        # d_q must be 3 x n_g x n_e = 3 x 1 x 3
        bad_d_q = jnp.zeros((3, 2, 3), dtype=jnp.complex128)
        with pytest.raises(ValueError):
            self.ham.add_d_q_block('g', 'e', bad_d_q)

    def test_unknown_label1_raises(self):
        d_q = jnp.zeros((3, 1, 3), dtype=jnp.complex128)
        with pytest.raises(ValueError, match='not found'):
            self.ham.add_d_q_block('x', 'e', d_q)

    def test_unknown_label2_raises(self):
        d_q = jnp.zeros((3, 1, 3), dtype=jnp.complex128)
        with pytest.raises(ValueError, match='not found'):
            self.ham.add_d_q_block('g', 'x', d_q)

    def test_error_message_cites_label2_not_label1(self):
        # Regression test for copy-paste bug: error for unknown label2 must say label2
        d_q = jnp.zeros((3, 1, 3), dtype=jnp.complex128)
        with pytest.raises(ValueError, match='x'):
            self.ham.add_d_q_block('g', 'x', d_q)

    def test_dagger_stored(self):
        d_q = hamiltonians.dqij_two_bare_hyperfine(0, 1)
        self.ham.add_d_q_block('g', 'e', d_q)
        # Both g->e and e->g blocks should be populated
        ind = self.ham.laser_keys['g->e']
        assert self.ham.blocks[ind] is not None
        assert self.ham.blocks[ind[::-1]] is not None


# ---------------------------------------------------------------------------
# Five-argument constructor
# ---------------------------------------------------------------------------

class TestFiveArgConstructor:
    def test_two_level_construction(self):
        H0_g, H0_e, mu_g, mu_e, d_q = make_two_level()
        ham = hamiltonian(H0_g, H0_e, mu_g, mu_e, d_q)
        assert ham.n == 2
        assert 'g' in ham.state_labels
        assert 'e' in ham.state_labels

    def test_F0_to_F1(self):
        H0_g, H0_e, mu_g, mu_e, d_q = make_F0_to_F1()
        ham = hamiltonian(H0_g, H0_e, mu_g, mu_e, d_q)
        assert ham.n == 4   # 1 ground + 3 excited states
        assert 'g->e' in ham.laser_keys

    def test_wrong_num_args_raises(self):
        with pytest.raises((ValueError, NotImplementedError)):
            hamiltonian(jnp.zeros((1, 1)))


# ---------------------------------------------------------------------------
# make_full_matrices
# ---------------------------------------------------------------------------

class TestMakeFullMatrices:
    def setup_method(self):
        H0_g, H0_e, mu_g, mu_e, d_q = make_F0_to_F1()
        self.ham = hamiltonian(H0_g, H0_e, mu_g, mu_e, d_q)

    def test_returns_four_items(self):
        result = self.ham.make_full_matrices()
        assert len(result) == 4

    def test_H0_shape(self):
        self.ham.make_full_matrices()
        assert self.ham.H_0.shape == (4, 4)

    def test_mu_q_shape(self):
        self.ham.make_full_matrices()
        assert self.ham.mu_q.shape == (3, 4, 4)

    def test_d_q_bare_keys(self):
        self.ham.make_full_matrices()
        assert 'g->e' in self.ham.d_q_bare

    def test_d_q_shape(self):
        self.ham.make_full_matrices()
        assert self.ham.d_q.shape == (3, 4, 4)

    def test_mu_cartesian(self):
        self.ham.make_full_matrices()
        assert self.ham.mu.shape == (3, 4, 4)

    def test_d_cartesian(self):
        self.ham.make_full_matrices()
        assert self.ham.d.shape == (3, 4, 4)

    def test_d_q_hermitian_pair(self):
        # d_q_star[key] should be the conjugate transpose of d_q_bare[key]
        self.ham.make_full_matrices()
        key = 'g->e'
        for kk in range(3):
            dagger = jnp.conjugate(self.ham.d_q_bare[key][kk].T)
            assert jnp.allclose(dagger, self.ham.d_q_star[key][kk], atol=1e-10)

    def test_H0_zero_for_degenerate(self):
        # F=0 and F=1 both have zero field-independent energy
        self.ham.make_full_matrices()
        assert jnp.allclose(self.ham.H_0, jnp.zeros((4, 4)))


# ---------------------------------------------------------------------------
# return_full_H
# ---------------------------------------------------------------------------

class TestReturnFullH:
    def setup_method(self):
        H0_g, H0_e, mu_g, mu_e, d_q = make_F0_to_F1()
        self.ham = hamiltonian(H0_g, H0_e, mu_g, mu_e, d_q)
        self.ham.make_full_matrices()

    def test_zero_fields_gives_H0(self):
        Eq = jnp.zeros(3, dtype=jnp.complex128)
        Bq = jnp.zeros(3, dtype=jnp.complex128)
        H = self.ham.return_full_H(Eq, Bq)
        assert H.shape == (4, 4)
        assert jnp.allclose(H, self.ham.H_0, atol=1e-12)

    def test_H_is_hermitian_at_zero_field(self):
        Eq = jnp.zeros(3, dtype=jnp.complex128)
        Bq = jnp.zeros(3, dtype=jnp.complex128)
        H = self.ham.return_full_H(Eq, Bq)
        assert jnp.allclose(H, jnp.conj(H.T), atol=1e-10)

    def test_H_dict_eq(self):
        # Passing Eq as a dict should give same result as array
        Eq_arr = jnp.array([1., 0., 0.], dtype=jnp.complex128)
        Bq = jnp.zeros(3, dtype=jnp.complex128)
        H_arr = self.ham.return_full_H(Eq_arr, Bq)
        H_dict = self.ham.return_full_H({'g->e': Eq_arr}, Bq)
        assert jnp.allclose(H_arr, H_dict)

    def test_nonzero_B_breaks_degeneracy(self):
        Eq = jnp.zeros(3, dtype=jnp.complex128)
        Bq = jnp.array([0., 1., 0.], dtype=jnp.complex128)  # B along y (spherical pi)
        H = self.ham.return_full_H(Eq, Bq)
        # Diagonal should now differ from H_0 for the excited states
        assert not jnp.allclose(jnp.diag(H), jnp.diag(self.ham.H_0), atol=1e-10)

    def test_make_full_matrices_called_automatically(self):
        # If make_full_matrices was never called, return_full_H calls it
        H0_g, H0_e, mu_g, mu_e, d_q = make_F0_to_F1()
        ham_fresh = hamiltonian(H0_g, H0_e, mu_g, mu_e, d_q)
        Eq = jnp.zeros(3, dtype=jnp.complex128)
        Bq = jnp.zeros(3, dtype=jnp.complex128)
        H = ham_fresh.return_full_H(Eq, Bq)
        assert H.shape == (4, 4)


# ---------------------------------------------------------------------------
# diag_static_field
# ---------------------------------------------------------------------------

class TestDiagStaticField:
    def setup_method(self):
        H0_g, mu_g = hamiltonians.singleF(F=1, gF=1)
        H0_e, mu_e = hamiltonians.singleF(F=2, gF=1)
        d_q = hamiltonians.dqij_two_bare_hyperfine(1, 2)
        self.ham = hamiltonian(H0_g, H0_e, mu_g, mu_e, d_q)

    def test_returns_hamiltonian(self):
        rot = self.ham.diag_static_field(1.0)
        assert isinstance(rot, hamiltonian)

    def test_zero_field(self):
        rot = self.ham.diag_static_field(0.0)
        assert isinstance(rot, hamiltonian)

    def test_rotated_H0_is_diagonal(self):
        rot = self.ham.diag_static_field(1.0)
        rot.make_full_matrices()
        H0 = rot.H_0
        # Off-diagonal elements should be ~zero after diagonalization
        off_diag = H0 - jnp.diag(jnp.diagonal(H0))
        assert jnp.allclose(off_diag, jnp.zeros_like(off_diag), atol=1e-10)

    def test_non_float_raises(self):
        with pytest.raises(ValueError):
            self.ham.diag_static_field(jnp.array([0., 0., 1.]))

    def test_U_matrices_are_unitary(self):
        self.ham.diag_static_field(1.0)
        for U in self.ham.U:
            # U should be unitary: U† U = I
            prod = jnp.conj(U.T) @ U
            assert jnp.allclose(prod, jnp.eye(U.shape[0]), atol=1e-10)

    def test_eigenvalues_real(self):
        rot = self.ham.diag_static_field(2.0)
        rot.make_full_matrices()
        assert jnp.allclose(jnp.imag(jnp.diagonal(rot.H_0)),
                             jnp.zeros(self.ham.n), atol=1e-10)


# ---------------------------------------------------------------------------
# print_structure (smoke test)
# ---------------------------------------------------------------------------

class TestPrintStructure:
    def test_print_runs(self, capsys):
        ham = hamiltonian()
        ham.add_H_0_block('g', jnp.zeros((2, 2)))
        ham.print_structure()
        out = capsys.readouterr().out
        assert len(out) > 0


# ---------------------------------------------------------------------------
# hamiltonians module helpers
# ---------------------------------------------------------------------------

class TestHamiltoniansModule:
    def test_singleF_shapes(self):
        H0, mu_q = hamiltonians.singleF(F=1, gF=1)
        assert H0.shape == (3, 3)
        assert mu_q.shape == (3, 3, 3)

    def test_singleF_H0_zero(self):
        H0, _ = hamiltonians.singleF(F=1, gF=1)
        assert jnp.allclose(H0, jnp.zeros((3, 3)))

    def test_singleF_return_basis(self):
        H0, mu_q, basis = hamiltonians.singleF(F=1, gF=1, return_basis=True)
        assert basis.shape == (3, 2)  # 3 states, (F, mF)

    def test_dqij_two_bare_hyperfine_F0_F1(self):
        d_q = hamiltonians.dqij_two_bare_hyperfine(0, 1)
        assert d_q.shape == (3, 1, 3)

    def test_dqij_two_bare_hyperfine_shape(self):
        d_q = hamiltonians.dqij_two_bare_hyperfine(1, 2)
        assert d_q.shape == (3, 3, 5)

    def test_dqij_selection_rules(self):
        # For F=0->F'=1, only q=-1,0,+1 connect m_F=0 to m_F'=-1,0,+1
        d_q = hamiltonians.dqij_two_bare_hyperfine(0, 1)
        # Each column (excited state) should have exactly one nonzero per q
        for q_idx in range(3):
            # Exactly one nonzero in each row of d_q[q_idx]
            assert int(jnp.count_nonzero(d_q[q_idx])) == 1

    def test_hyperfine_coupled_shapes(self):
        # J=1/2, I=3/2 -> 8 states
        H0, mu_q = hamiltonians.hyperfine_coupled(
            J=0.5, I=1.5, gJ=2.0, gI=0.001, Ahfs=1.0
        )
        assert H0.shape == (8, 8)
        assert mu_q.shape == (3, 8, 8)

    def test_coupled_index(self):
        # F=1, mF=0 at Fmin=0 -> index 2
        assert hamiltonians.coupled_index(1, 0, 0) == 2

    def test_dqij_two_hyperfine_manifolds(self):
        d_q = hamiltonians.dqij_two_hyperfine_manifolds(J=0, Jp=1, I=0)
        assert d_q.shape == (3, 1, 3)
