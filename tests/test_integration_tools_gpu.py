"""
Tests for pylcp/integration_tools_gpu.py

Covers RandomOdeResult, solve_ivp_dense, and solve_ivp_random.
All tests run on CPU (JAX default); no GPU required.
"""
import math
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pylcp.integration_tools_gpu import (
    RandomOdeResult,
    solve_ivp_dense,
    solve_ivp_random,
)


# ---------------------------------------------------------------------------
# Shared ODE definitions
# ---------------------------------------------------------------------------

def exp_decay(t, y, args):
    """dy/dt = -y  ->  y(t) = y0 * exp(-t)."""
    return -y


def harmonic(t, y, args):
    """Simple harmonic oscillator: d/dt [x, v] = [v, -x]."""
    return jnp.array([-y[1], y[0]])


def dummy_random(t, y, dt, key):
    """No-op stochastic function: never scatters."""
    return y, jnp.int32(0), jnp.float64(dt), key


def always_scatter(t, y, dt, key):
    """Always records one scatter event per step."""
    key, _ = jax.random.split(key)
    return y, jnp.int32(1), jnp.float64(dt), key


# ---------------------------------------------------------------------------
# TestRandomOdeResult
# ---------------------------------------------------------------------------

class TestRandomOdeResult:
    def _make(self):
        t = jnp.linspace(0., 1., 10)
        y = jnp.ones((2, 10))
        return RandomOdeResult(
            t=t, y=y,
            t_random=jnp.array([0.5]),
            n_random=jnp.array([1]),
            inds_random=jnp.zeros(10, dtype=bool).at[5].set(True),
            success=True,
            status=0,
            message="Success",
            nfev=42,
        )

    def test_has_t(self):
        assert hasattr(self._make(), 't')

    def test_has_y(self):
        assert hasattr(self._make(), 'y')

    def test_has_t_random(self):
        assert hasattr(self._make(), 't_random')

    def test_has_n_random(self):
        assert hasattr(self._make(), 'n_random')

    def test_has_inds_random(self):
        assert hasattr(self._make(), 'inds_random')

    def test_success_true(self):
        assert self._make().success is True

    def test_status_zero(self):
        assert self._make().status == 0

    def test_message_stored(self):
        assert self._make().message == "Success"

    def test_nfev_stored(self):
        assert self._make().nfev == 42


# ---------------------------------------------------------------------------
# TestSolveIvpDense
# ---------------------------------------------------------------------------

class TestSolveIvpDense:
    """GPU-batched deterministic ODE solver (solve_ivp_dense).

    Validates against analytical solutions: exponential decay dy/dt = −y
    → y(t) = y₀·e⁻ᵗ, and harmonic oscillator d/dt[x,v] = [v,−x] with
    energy conservation E = ½(x² + v²) = const.  Tests batching (multiple
    y₀ solved in parallel), solver backends (Dopri5, Bosh3, Kvaerno5),
    tolerance effects, and the dt₀ scaling fix for long time spans."""

    # --- shape / structure ---

    def test_ts_shape(self):
        y0 = jnp.array([[1.0]])
        ts, _ = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=50)
        assert ts.shape == (50,)

    def test_ys_shape_single(self):
        y0 = jnp.array([[1.0]])
        _, ys = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=50)
        assert ys.shape == (1, 50, 1)

    def test_ys_shape_batch(self):
        y0 = jnp.array([[1.0], [2.0], [3.0]])
        _, ys = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=21)
        assert ys.shape == (3, 21, 1)

    def test_ys_shape_2d_state(self):
        y0 = jnp.array([[0.0, 1.0]])
        _, ys = solve_ivp_dense(harmonic, [0., 1.], y0, n_points=11)
        assert ys.shape == (1, 11, 2)

    def test_ts_starts_at_t0(self):
        y0 = jnp.array([[1.0]])
        ts, _ = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=11)
        assert float(ts[0]) == pytest.approx(0.0)

    def test_ts_ends_at_tf(self):
        y0 = jnp.array([[1.0]])
        ts, _ = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=11)
        assert float(ts[-1]) == pytest.approx(1.0)

    def test_ts_monotonically_increasing(self):
        y0 = jnp.array([[1.0]])
        ts, _ = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=20)
        assert jnp.all(jnp.diff(ts) > 0)

    # --- numerical correctness ---

    def test_exponential_decay_accuracy(self):
        y0 = jnp.array([[2.0]])
        ts, ys = solve_ivp_dense(exp_decay, [0., 2.], y0, n_points=101)
        expected = 2.0 * jnp.exp(-ts)
        assert jnp.allclose(ys[0, :, 0], expected, atol=1e-4)

    def test_batch_scales_linearly(self):
        """y0 doubled → solution doubled at all times."""
        y0 = jnp.array([[1.0], [2.0]])
        ts, ys = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=51)
        ratio = ys[1, :, 0] / ys[0, :, 0]
        assert jnp.allclose(ratio, jnp.full_like(ratio, 2.0), atol=1e-5)

    def test_harmonic_half_period(self):
        """x(π) ≈ 0 for harmonic oscillator starting at x=0, v=1."""
        y0 = jnp.array([[0.0, 1.0]])
        ts, ys = solve_ivp_dense(harmonic, [0., jnp.pi], y0, n_points=101)
        assert float(ys[0, -1, 0]) == pytest.approx(0.0, abs=1e-3)

    def test_energy_conservation_harmonic(self):
        """E = 0.5*(x² + v²) must remain constant."""
        y0 = jnp.array([[0.0, 1.0]])
        ts, ys = solve_ivp_dense(harmonic, [0., 4 * jnp.pi], y0,
                                 n_points=201, rtol=1e-8, atol=1e-8)
        energy = 0.5 * (ys[0, :, 0] ** 2 + ys[0, :, 1] ** 2)
        assert jnp.allclose(energy, jnp.full_like(energy, 0.5), atol=1e-5)

    def test_no_nan_in_output(self):
        y0 = jnp.array([[1.0]])
        _, ys = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=50)
        assert not jnp.any(jnp.isnan(ys))

    # --- solver types ---

    def test_solver_dopri5(self):
        y0 = jnp.array([[1.0]])
        ts, ys = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=51,
                                 solver_type='Dopri5')
        assert jnp.allclose(ys[0, :, 0], jnp.exp(-ts), atol=1e-4)

    def test_solver_bosh3(self):
        y0 = jnp.array([[1.0]])
        ts, ys = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=51,
                                 solver_type='Bosh3')
        assert jnp.allclose(ys[0, :, 0], jnp.exp(-ts), atol=1e-3)

    def test_solver_kvaerno5(self):
        """Kvaerno5 is a stiff solver; verify it runs without error."""
        y0 = jnp.array([[1.0]])
        ts, ys = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=51,
                                 solver_type='Kvaerno5')
        assert not jnp.any(jnp.isnan(ys))
        assert float(ys[0, -1, 0]) == pytest.approx(math.exp(-1.), abs=1e-3)

    def test_invalid_solver_raises(self):
        y0 = jnp.array([[1.0]])
        with pytest.raises(Exception):
            solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=10,
                            solver_type='NotASolver')

    # --- tolerance effect ---

    def test_tighter_tolerance_more_accurate(self):
        y0 = jnp.array([[1.0]])
        _, ys_lo = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=11,
                                   rtol=1e-3, atol=1e-3)
        _, ys_hi = solve_ivp_dense(exp_decay, [0., 1.], y0, n_points=11,
                                   rtol=1e-9, atol=1e-9)
        exact = math.exp(-1.0)
        err_lo = abs(float(ys_lo[0, -1, 0]) - exact)
        err_hi = abs(float(ys_hi[0, -1, 0]) - exact)
        assert err_hi <= err_lo + 1e-12

    # --- long span dt0 scaling ---

    def test_long_span_no_nan(self):
        """dt0 scaling fix: a span of [0, 100] should not produce NaN."""
        y0 = jnp.array([[1.0]])
        _, ys = solve_ivp_dense(exp_decay, [0., 100.], y0, n_points=51)
        assert not jnp.any(jnp.isnan(ys))


# ---------------------------------------------------------------------------
# TestSolveIvpRandom
# ---------------------------------------------------------------------------

class TestSolveIvpRandom:
    """Stochastic ODE solver interleaving integration with random scattering.

    Each result is a RandomOdeResult carrying y(t) plus scattered-event
    times t_random, event counts n_random, and boolean indices inds_random.
    Without scattering (dummy_random) the solution matches y₀·e⁻ᵗ.  With
    always_scatter every step records an event.  Tests batch independence,
    early termination (max_steps too small → success=False), and the dt₀
    scaling fix for long spans."""

    def _key(self, seed=42):
        return jax.random.PRNGKey(seed)

    def _keys(self, n, seed=42):
        return jax.random.split(self._key(seed), n)

    # --- return type / structure ---

    def test_returns_list(self):
        y0 = jnp.array([[1.0]])
        result = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                                  y0, self._keys(1))
        assert isinstance(result, list)

    def test_list_length_matches_batch(self):
        N = 4
        y0 = jnp.ones((N, 1))
        results = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                                   y0, self._keys(N))
        assert len(results) == N

    def test_result_has_t(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert hasattr(sol, 't')

    def test_result_has_y(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert hasattr(sol, 'y')

    def test_result_has_t_random(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert hasattr(sol, 't_random')

    def test_result_has_n_random(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert hasattr(sol, 'n_random')

    def test_result_has_inds_random(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert hasattr(sol, 'inds_random')

    def test_result_has_success(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert hasattr(sol, 'success')

    # --- time array properties ---

    def test_t_starts_at_t0(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert float(sol.t[0]) == pytest.approx(0.0, abs=1e-10)

    def test_t_ends_at_tf(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1), max_steps=10000)[0]
        assert float(sol.t[-1]) == pytest.approx(1.0, abs=1e-6)

    def test_t_monotonically_increasing(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert jnp.all(jnp.diff(sol.t) >= 0.)

    # --- y array properties ---

    def test_y_t_same_length(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert sol.y.shape[1] == sol.t.shape[0]

    def test_y_state_dim_correct(self):
        y0 = jnp.array([[0.0, 1.0]])  # 2-element state
        sol = solve_ivp_random(harmonic, dummy_random, [0., 1.],
                               y0, self._keys(1), max_steps=5000)[0]
        assert sol.y.shape[0] == 2

    def test_no_nan_in_y(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert not jnp.any(jnp.isnan(sol.y))

    # --- numerical correctness ---

    def test_exponential_decay_accuracy(self):
        """Without scattering, result matches analytical exp(-t)."""
        y0 = jnp.array([[2.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1),
                               max_steps=10000, rtol=1e-6, atol=1e-6)[0]
        y_final = float(sol.y[0, -1])
        expected = 2.0 * math.exp(-1.0)
        assert y_final == pytest.approx(expected, rel=1e-3)

    # --- success / status ---

    def test_success_flag_true(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1), max_steps=10000)[0]
        assert sol.success is True

    def test_status_zero_on_success(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1), max_steps=10000)[0]
        assert sol.status == 0

    def test_terminated_early_when_max_steps_too_small(self):
        """max_steps=2 forces early termination."""
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 100.],
                               y0, self._keys(1), max_steps=2)[0]
        assert sol.success is False
        assert sol.status == -1

    # --- stochastic events ---

    def test_no_scatter_gives_empty_t_random(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert len(sol.t_random) == 0

    def test_always_scatter_populates_t_random(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, always_scatter, [0., 0.5],
                               y0, self._keys(1), max_steps=2000)[0]
        assert len(sol.t_random) > 0

    def test_always_scatter_n_random_positive(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, always_scatter, [0., 0.5],
                               y0, self._keys(1), max_steps=2000)[0]
        assert jnp.all(sol.n_random > 0)

    def test_inds_random_dtype_bool(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1))[0]
        assert sol.inds_random.dtype == jnp.bool_

    # --- batch independence ---

    def test_multi_atom_all_succeed(self):
        N = 3
        y0 = jnp.ones((N, 1))
        results = solve_ivp_random(exp_decay, dummy_random, [0., 0.5],
                                   y0, self._keys(N), max_steps=5000)
        assert all(sol.success for sol in results)

    def test_different_y0_batch_independent(self):
        """Two atoms with different y0 should give proportionally scaled y."""
        y0 = jnp.array([[1.0], [2.0]])
        results = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                                   y0, self._keys(2),
                                   max_steps=10000, rtol=1e-7, atol=1e-7)
        y1_final = float(results[0].y[0, -1])
        y2_final = float(results[1].y[0, -1])
        assert y2_final == pytest.approx(2.0 * y1_final, rel=1e-4)

    # --- solver types ---

    def test_solver_bosh3(self):
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                               y0, self._keys(1),
                               solver_type='Bosh3', max_steps=10000)[0]
        assert sol.success

    # --- max_step controls step count ---

    def test_small_max_step_gives_more_points(self):
        """A tighter max_step produces more time points."""
        y0 = jnp.array([[1.0]])
        sol_free = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                                    y0, self._keys(1, seed=0),
                                    max_step=float('inf'), max_steps=5000)[0]
        sol_tight = solve_ivp_random(exp_decay, dummy_random, [0., 1.],
                                     y0, self._keys(1, seed=0),
                                     max_step=0.05, max_steps=5000)[0]
        assert len(sol_tight.t) >= len(sol_free.t)

    # --- dt0 scaling fix ---

    def test_long_span_no_nan(self):
        """dt0 scaling fix: span [0, 500] should not produce NaN or hang."""
        y0 = jnp.array([[1.0]])
        sol = solve_ivp_random(exp_decay, dummy_random, [0., 500.],
                               y0, self._keys(1), max_steps=50000)[0]
        assert not jnp.any(jnp.isnan(sol.y))