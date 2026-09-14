import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.welfare import (compute_annuity_factor, compute_certainty_equivalent,
                          compute_aggregate_welfare_formula)
from src.labor_solver import compute_alpha_sigma


def test_annuity_rho_zero():
    """K(T) = T + 1 when rho=0."""
    assert compute_annuity_factor(0.0, 1.0) == pytest.approx(2.0)
    assert compute_annuity_factor(0.0, 2.0) == pytest.approx(3.0)
    assert compute_annuity_factor(0.0, 0.5) == pytest.approx(1.5)


def test_annuity_positive():
    """K(T) > 0 for any rho >= 0 and T > 0."""
    for rho in [0.0, 0.5, 1.0, 2.0]:
        for T in [0.1, 1.0, 5.0]:
            K = compute_annuity_factor(rho, T)
            assert K > 0, f"K must be positive for rho={rho}, T={T}"


def test_annuity_formula_positive_rho():
    """K(T) = (1-exp(-rho*T))/rho + exp(-rho*T) for rho>0."""
    rho, T = 0.5, 2.0
    expected = (1 - np.exp(-rho * T)) / rho + np.exp(-rho * T)
    assert compute_annuity_factor(rho, T) == pytest.approx(expected)


def test_annuity_limit_rho_zero():
    """As rho->0, K(T) -> T+1."""
    K_small = compute_annuity_factor(1e-8, 1.0)
    K_zero = compute_annuity_factor(0.0, 1.0)  # = 2.0
    assert abs(K_small - K_zero) < 1e-4


def test_certainty_equivalent_structure():
    """CE_i = (1/alpha)*log_u(L) + (1/alpha)*log_K + X0/A0 + Y0."""
    alpha, beta, gamma = 0.3, -0.1, -0.5
    L, rho, T = 0.5, 0.5, 1.0
    X_i0_over_A0, Y_i0 = 1.0, 0.5
    log_u = beta * np.log(L) + gamma * np.log(1 - L)
    K = compute_annuity_factor(rho, T)
    expected = (1 / alpha) * log_u + (1 / alpha) * np.log(K) + X_i0_over_A0 + Y_i0
    result = compute_certainty_equivalent(alpha, beta, gamma, L, rho, T, X_i0_over_A0, Y_i0)
    assert abs(result - expected) < 1e-12


def test_certainty_equivalent_finite():
    """CE_i must be finite for valid L in (0,1)."""
    CE = compute_certainty_equivalent(0.3, -0.1, -0.5, 0.5, 0.5, 1.0, 1.0, 0.5)
    assert np.isfinite(CE)


def test_aggregate_welfare_formula_structure():
    """Verify formula: 1 + D0 + eps0 - a0/alpha_Sigma + sum(1/alpha_i * log K_i)."""
    alphas = np.array([0.3, 0.4])
    betas = np.array([-0.1, -0.15])
    gammas = np.array([-0.5, -0.7])
    rhos = np.array([0.5, 0.8])
    labors = np.array([0.5, 0.4])
    T, D0, eps0, a0 = 1.0, 1.0, 2.0, 0.1
    alpha_Sigma = compute_alpha_sigma(alphas)
    result = compute_aggregate_welfare_formula(alphas, betas, gammas, labors, rhos,
                                               T, D0, eps0, a0, alpha_Sigma)
    assert np.isfinite(result)
    # Manual check
    sum_log_K = sum(
        (1 / alphas[i]) * np.log(compute_annuity_factor(rhos[i], T))
        for i in range(2)
    )
    expected = 1.0 + D0 + eps0 - a0 / alpha_Sigma + sum_log_K
    assert abs(result - expected) < 1e-12
