import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.bsde_solver import solve_deterministic_bsde, BSDEResult
from src.labor_solver import compute_alpha_sigma, compute_rho_sigma


def make_simple_params():
    """Standard test parameters."""
    alphas = np.array([0.3, 0.4])
    rhos = np.array([0.5, 0.8])
    epsilon_i = np.array([1.0, 1.5])
    lcal_i = np.array([0.2, 0.3])
    alpha_Sigma = compute_alpha_sigma(alphas)
    rho_Sigma = compute_rho_sigma(alphas, rhos)
    return alphas, rhos, epsilon_i, lcal_i, alpha_Sigma, rho_Sigma


def test_deterministic_bsde_terminal_conditions():
    """a(T)=0 and Y_i(T)=epsilon_i-lcal_i must hold at terminal time."""
    alphas, rhos, epsilon_i, lcal_i, alpha_Sigma, rho_Sigma = make_simple_params()
    T, N_steps = 1.0, 100
    C, kappa = 1.2, 0.3
    result = solve_deterministic_bsde(T, N_steps, C, kappa, alphas, rhos,
                                      epsilon_i, lcal_i, alpha_Sigma, rho_Sigma)
    assert abs(result.a[-1]) < 1e-6, f"a(T) should be 0, got {result.a[-1]}"
    c_vec = epsilon_i - lcal_i
    for i in range(2):
        err = abs(result.Y[i, -1] - c_vec[i])
        assert err < 1e-6, f"Y_{i}(T) error {err} too large"


def test_deterministic_bsde_z_values_zero():
    """In deterministic mode, Z_a and Z_i should all be zero."""
    alphas, rhos, epsilon_i, lcal_i, alpha_Sigma, rho_Sigma = make_simple_params()
    result = solve_deterministic_bsde(1.0, 50, 1.2, 0.3, alphas, rhos,
                                      epsilon_i, lcal_i, alpha_Sigma, rho_Sigma)
    assert np.allclose(result.Z_a, 0.0)
    assert np.allclose(result.Z, 0.0)


def test_deterministic_bsde_a_positive():
    """A = exp(a) > 0 is guaranteed; here verify a values are finite."""
    alphas, rhos, epsilon_i, lcal_i, alpha_Sigma, rho_Sigma = make_simple_params()
    result = solve_deterministic_bsde(1.0, 100, 1.2, 0.3, alphas, rhos,
                                      epsilon_i, lcal_i, alpha_Sigma, rho_Sigma)
    A = np.exp(result.a)
    assert np.all(A > 0)
    assert np.all(np.isfinite(result.a))


def test_deterministic_bsde_returns_correct_shape():
    """BSDEResult arrays have correct shapes."""
    alphas, rhos, epsilon_i, lcal_i, alpha_Sigma, rho_Sigma = make_simple_params()
    N = 80
    result = solve_deterministic_bsde(2.0, N, 1.0, 0.2, alphas, rhos,
                                      epsilon_i, lcal_i, alpha_Sigma, rho_Sigma)
    assert result.t_grid.shape == (N + 1,)
    assert result.a.shape == (N + 1,)
    assert result.Y.shape == (2, N + 1)
    assert result.Z_a.shape == (N + 1,)
    assert result.Z.shape == (2, N + 1)


def test_deterministic_bsde_time_monotone():
    """t_grid must be monotonically increasing from 0 to T."""
    alphas, rhos, epsilon_i, lcal_i, alpha_Sigma, rho_Sigma = make_simple_params()
    result = solve_deterministic_bsde(1.0, 50, 1.0, 0.2, alphas, rhos,
                                      epsilon_i, lcal_i, alpha_Sigma, rho_Sigma)
    assert result.t_grid[0] == pytest.approx(0.0)
    assert result.t_grid[-1] == pytest.approx(1.0)
    assert np.all(np.diff(result.t_grid) > 0)


def test_deterministic_bsde_different_T():
    """Test with T=0.5 and T=5.0."""
    alphas, rhos, epsilon_i, lcal_i, alpha_Sigma, rho_Sigma = make_simple_params()
    for T in [0.5, 5.0]:
        result = solve_deterministic_bsde(T, 100, 1.0, 0.2, alphas, rhos,
                                          epsilon_i, lcal_i, alpha_Sigma, rho_Sigma)
        assert abs(result.a[-1]) < 1e-4
        assert result.mode == 'deterministic'


def test_deterministic_bsde_kappa_zero():
    """With kappa=0, the quadratic term vanishes."""
    alphas, rhos, epsilon_i, lcal_i, alpha_Sigma, rho_Sigma = make_simple_params()
    result = solve_deterministic_bsde(1.0, 100, 1.0, 0.0, alphas, rhos,
                                      epsilon_i, lcal_i, alpha_Sigma, rho_Sigma)
    assert abs(result.a[-1]) < 1e-6
