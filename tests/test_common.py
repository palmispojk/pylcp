"""
Tests for pylcp/common.py
"""
import time
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from pylcp.common import (
    progressBar,
    cart2spherical,
    spherical2cart,
    spherical_dot,
    base_force_profile,
    random_vector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_hamiltonian(n):
    class _H:
        pass
    h = _H()
    h.n = n
    return h


def _dummy_laserBeams(keys_and_sizes):
    """Return a dict whose values have .beam_vector lists of given lengths."""
    class _Beams:
        def __init__(self, size):
            self.beam_vector = [None] * size
    return {k: _Beams(n) for k, n in keys_and_sizes.items()}


# ---------------------------------------------------------------------------
# cart2spherical / spherical2cart
# ---------------------------------------------------------------------------

class TestCart2Spherical:
    def test_z_axis_is_pure_pi_zero(self):
        # ẑ maps to m=0 component only
        s = cart2spherical(jnp.array([0., 0., 1.]))
        assert float(jnp.abs(s[1])) == pytest.approx(1.0, abs=1e-6)
        assert float(jnp.abs(s[0])) == pytest.approx(0.0, abs=1e-6)
        assert float(jnp.abs(s[2])) == pytest.approx(0.0, abs=1e-6)

    def test_x_axis_equal_pm1(self):
        # x̂ has equal |m|=1 components and zero m=0
        s = cart2spherical(jnp.array([1., 0., 0.]))
        assert float(jnp.abs(s[0])) == pytest.approx(1/np.sqrt(2), abs=1e-6)
        assert float(jnp.abs(s[2])) == pytest.approx(1/np.sqrt(2), abs=1e-6)
        assert float(jnp.abs(s[1])) == pytest.approx(0.0, abs=1e-6)

    def test_output_shape(self):
        s = cart2spherical(jnp.array([1., 2., 3.]))
        assert s.shape == (3,)

    def test_roundtrip(self):
        A = jnp.array([1., -2., 3.])
        result = spherical2cart(cart2spherical(A))
        assert jnp.allclose(jnp.real(result), A, atol=1e-6)

    def test_norm_preserved(self):
        A = jnp.array([3., 4., 0.])
        s = cart2spherical(A)
        # |A|^2 = sum(|s_q|^2) -- norm preserved in the spherical basis
        norm_cart = float(jnp.linalg.norm(A))
        norm_sph = float(jnp.sqrt(jnp.sum(jnp.abs(s)**2)))
        assert norm_sph == pytest.approx(norm_cart, rel=1e-5)


class TestSpherical2Cart:
    def test_m0_is_z(self):
        # pure m=0 → ẑ
        A = jnp.array([0., 1., 0.], dtype=jnp.complex64)
        c = spherical2cart(A)
        assert float(jnp.abs(c[2])) == pytest.approx(1.0, abs=1e-6)
        assert float(jnp.abs(c[0])) == pytest.approx(0.0, abs=1e-6)
        assert float(jnp.abs(c[1])) == pytest.approx(0.0, abs=1e-6)

    def test_roundtrip(self):
        A = jnp.array([1., 0., -1.], dtype=jnp.complex64) / jnp.sqrt(2)
        result = cart2spherical(spherical2cart(A))
        assert jnp.allclose(result, A, atol=1e-6)

    def test_output_shape(self):
        A = jnp.array([1., 0., 0.], dtype=jnp.complex64)
        assert spherical2cart(A).shape == (3,)


# ---------------------------------------------------------------------------
# spherical_dot
# ---------------------------------------------------------------------------

class TestSphericalDot:
    def test_same_as_cartesian_dot_for_real(self):
        # For real vectors, spherical_dot(cart2spherical(A), cart2spherical(B))
        # should equal the Cartesian dot product A·B.
        A = jnp.array([1., 2., 3.])
        B = jnp.array([4., -1., 2.])
        dot_cart = float(jnp.dot(A, B))
        dot_sph = float(jnp.real(spherical_dot(cart2spherical(A), cart2spherical(B))))
        assert dot_sph == pytest.approx(dot_cart, rel=1e-5)

    def test_orthogonal_vectors(self):
        A = jnp.array([1., 0., 0.])
        B = jnp.array([0., 1., 0.])
        result = float(jnp.real(spherical_dot(cart2spherical(A), cart2spherical(B))))
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_self_dot_equals_norm_squared(self):
        A = jnp.array([3., 4., 0.])
        sA = cart2spherical(A)
        result = float(jnp.real(spherical_dot(sA, sA)))
        assert result == pytest.approx(25.0, rel=1e-5)


# ---------------------------------------------------------------------------
# progressBar
# ---------------------------------------------------------------------------

class TestProgressBar:
    def test_format_time_seconds(self):
        pb = progressBar()
        assert 's' in pb.format_time(5.0)

    def test_format_time_minutes(self):
        pb = progressBar()
        t = pb.format_time(90.0)
        assert ':' in t

    def test_format_time_hours(self):
        pb = progressBar()
        t = pb.format_time(7200.0)
        assert t.startswith('2:')

    def test_update_at_completion_prints_once(self, capsys):
        pb = progressBar()
        pb.update(1.0)
        out1 = capsys.readouterr().out
        pb.update(1.0)
        out2 = capsys.readouterr().out
        assert 'Completed' in out1
        assert out2 == ''  # second call is a no-op

    def test_update_zero_does_nothing(self, capsys):
        pb = progressBar()
        pb.update(0.0)
        out = capsys.readouterr().out
        assert out == ''

    def test_print_string_pads_to_max_width(self, capsys):
        pb = progressBar()
        pb.print_string('hello world')
        pb.print_string('hi')
        out = capsys.readouterr().out
        # The second write should be padded to at least the first width
        lines = out.split('\r')
        assert len(lines[1]) >= len(lines[0]) - len('\r')


# ---------------------------------------------------------------------------
# base_force_profile
# ---------------------------------------------------------------------------

class TestBaseForceProfile:
    def setup_method(self):
        self.R = jnp.zeros((3, 4))  # 4 spatial points
        self.V = jnp.zeros((3, 4))
        self.ham = _dummy_hamiltonian(3)
        self.beams = _dummy_laserBeams({'g->e': 2, 'g->e2': 1})

    def test_Neq_shape(self):
        fp = base_force_profile(self.R, self.V, self.beams, self.ham)
        assert fp.Neq.shape == (4, 3)

    def test_F_shape(self):
        fp = base_force_profile(self.R, self.V, self.beams, self.ham)
        assert fp.F.shape == (3, 4)

    def test_f_keys(self):
        fp = base_force_profile(self.R, self.V, self.beams, self.ham)
        assert 'g->e' in fp.f
        assert 'g->e2' in fp.f

    def test_f_shape_per_key(self):
        fp = base_force_profile(self.R, self.V, self.beams, self.ham)
        assert fp.f['g->e'].shape == (3, 4, 2)
        assert fp.f['g->e2'].shape == (3, 4, 1)

    def test_no_hamiltonian_neq_is_none(self):
        fp = base_force_profile(self.R, self.V, self.beams, None)
        assert fp.Neq is None

    def test_numpy_arrays_converted(self):
        R_np = np.zeros((3, 2))
        V_np = np.zeros((3, 2))
        fp = base_force_profile(R_np, V_np, self.beams, self.ham)
        assert isinstance(fp.R, jnp.ndarray)
        assert isinstance(fp.V, jnp.ndarray)

    def test_invalid_R_shape_raises(self):
        with pytest.raises(TypeError):
            base_force_profile(jnp.zeros((4, 4)), self.V, self.beams, self.ham)

    def test_store_data(self):
        fp = base_force_profile(self.R, self.V, self.beams, self.ham)
        Neq = jnp.ones(3)
        F = jnp.array([1., 2., 3.])
        F_laser = {'g->e': jnp.array([0.1, 0.2, 0.3]),
                   'g->e2': jnp.array([0.4, 0.5, 0.6])}
        F_mag = jnp.array([0., 0., 0.])
        ind = (0,)
        fp.store_data(ind, Neq, F, F_laser, F_mag)
        assert float(fp.F[0, 0]) == pytest.approx(1.0)
        assert float(fp.F[1, 0]) == pytest.approx(2.0)
        assert float(fp.F[2, 0]) == pytest.approx(3.0)

    def test_store_data_none_neq(self):
        fp = base_force_profile(self.R, self.V, self.beams, None)
        F = jnp.zeros(3)
        F_laser = {'g->e': jnp.zeros(3), 'g->e2': jnp.zeros(3)}
        F_mag = jnp.zeros(3)
        # should not raise
        fp.store_data((0,), None, F, F_laser, F_mag)


# ---------------------------------------------------------------------------
# random_vector
# ---------------------------------------------------------------------------

class TestRandomVector:
    def test_3d_returns_shape_3(self):
        key = jax.random.PRNGKey(0)
        v = random_vector(key)
        assert v.shape == (3,)

    def test_3d_unit_length(self):
        key = jax.random.PRNGKey(1)
        v = random_vector(key)
        assert float(jnp.linalg.norm(v)) == pytest.approx(1.0, abs=1e-6)

    def test_1d_unit_length(self):
        key = jax.random.PRNGKey(2)
        v = random_vector(key, [True, False, False])
        assert float(jnp.linalg.norm(v)) == pytest.approx(1.0, abs=1e-6)

    def test_1d_nonzero_only_on_free_axis(self):
        key = jax.random.PRNGKey(3)
        v = random_vector(key, [False, True, False])
        assert float(jnp.abs(v[0])) == pytest.approx(0.0, abs=1e-6)
        assert float(jnp.abs(v[2])) == pytest.approx(0.0, abs=1e-6)
        assert float(jnp.abs(v[1])) == pytest.approx(1.0, abs=1e-6)

    def test_1d_returns_plus_or_minus_1(self):
        key = jax.random.PRNGKey(4)
        v = random_vector(key, [True, False, False])
        assert abs(float(v[0])) == pytest.approx(1.0, abs=1e-6)

    def test_2d_unit_length(self):
        key = jax.random.PRNGKey(5)
        v = random_vector(key, [True, True, False])
        assert float(jnp.linalg.norm(v)) == pytest.approx(1.0, abs=1e-6)

    def test_2d_zero_on_locked_axis(self):
        key = jax.random.PRNGKey(6)
        v = random_vector(key, [True, True, False])
        assert float(jnp.abs(v[2])) == pytest.approx(0.0, abs=1e-6)

    def test_dtype_consistent_across_branches(self):
        key = jax.random.PRNGKey(7)
        v1 = random_vector(key, [True, False, False])
        v2 = random_vector(key, [True, True, False])
        v3 = random_vector(key, [True, True, True])
        assert v1.dtype == v2.dtype == v3.dtype

    def test_vmap_over_keys(self):
        keys = jax.random.split(jax.random.PRNGKey(42), 100)
        vmap_fn = jax.vmap(random_vector, in_axes=(0, None))
        vectors = vmap_fn(keys, [True, True, True])
        assert vectors.shape == (100, 3)
        norms = jnp.linalg.norm(vectors, axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_3d_distributes_on_sphere(self):
        # Rough isotropy check: mean of many random vectors ≈ 0
        keys = jax.random.split(jax.random.PRNGKey(99), 2000)
        vmap_fn = jax.vmap(random_vector, in_axes=(0, None))
        vectors = vmap_fn(keys, [True, True, True])
        mean = jnp.mean(vectors, axis=0)
        assert float(jnp.linalg.norm(mean)) < 0.1
