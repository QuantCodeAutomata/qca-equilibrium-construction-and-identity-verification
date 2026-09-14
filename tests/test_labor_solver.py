import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.labor_solver import (psi, psi_prime, psi_second, solve_labor,
                               compute_effective_lambda, compute_alpha_sigma,
                               compute_rho_sigma, L_EPS)

# Standard test parameters
BETA = -0.1
GAMMA = -0.5


def test_psi_formula_basic():
    """Psi(l) = beta/l - gamma/(1-l) at l=0.5 with beta=-0.1, gamma=-0.5."""
    l = 0.5
    expected = BETA / l - GAMMA / (1 - l)  # -0.2 + 1.0 = 0.8
    assert abs(psi(l, BETA, GAMMA) - expected) < 1e-12


def test_psi_at_midpoint():
    """Numeric check of Psi(0.5)."""
    result = psi(0.5, -0.1, -0.5)
    assert abs(result - 0.8) < 1e-12


def test_psi_prime_positive():
    """Psi'(l) > 0 for beta<0, gamma<0 at multiple l values."""
    for l in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert psi_prime(l, BETA, GAMMA) > 0, f"Psi'({l}) must be positive"


def test_psi_strictly_increasing():
    """Psi must be strictly increasing: psi(l1) < psi(l2) for l1 < l2."""
    ls = np.linspace(0.05, 0.95, 50)
    psi_vals = [psi(l, BETA, GAMMA) for l in ls]
    diffs = np.diff(psi_vals)
    assert np.all(diffs > 0), "Psi must be strictly increasing"


def test_psi_second_formula():
    """Psi''(l) = 2*beta/l^3 - 2*gamma/(1-l)^3."""
    l, beta, gamma = 0.4, -0.2, -0.8
    expected = 2 * beta / l**3 - 2 * gamma / (1 - l)**3
    assert abs(psi_second(l, beta, gamma) - expected) < 1e-12


def test_solve_labor_foc_satisfied():
    """Solved L_i must satisfy Psi(L_i) = alpha*lambda*w."""
    alpha, beta, gamma = 0.3, -0.1, -0.5
    lambda_eff, w = 0.9, 2.0
    L = solve_labor(alpha, beta, gamma, lambda_eff, w)
    rhs = alpha * lambda_eff * w
    residual = abs(psi(L, beta, gamma) - rhs)
    assert residual < 1e-9, f"FOC residual {residual} too large"


def test_solve_labor_in_open_interval():
    """Solved L must be strictly in (0,1)."""
    alpha, beta, gamma = 0.2, -0.02, -0.8
    lambda_eff, w = 0.85, 3.0
    L = solve_labor(alpha, beta, gamma, lambda_eff, w)
    assert L_EPS <= L <= 1 - L_EPS, f"L={L} must be in (L_EPS, 1-L_EPS)"


def test_solve_labor_monotone_in_lambda():
    """Larger lambda_eff => larger L_i (Psi is increasing)."""
    alpha, beta, gamma, w = 0.3, -0.1, -0.5, 2.0
    L1 = solve_labor(alpha, beta, gamma, 0.5, w)
    L2 = solve_labor(alpha, beta, gamma, 0.9, w)
    assert L2 > L1, "L must increase with lambda_eff"


def test_solve_labor_monotone_in_wage():
    """Larger wage => larger L_i (for positive lambda_eff)."""
    alpha, beta, gamma, lambda_eff = 0.3, -0.1, -0.5, 0.9
    L1 = solve_labor(alpha, beta, gamma, lambda_eff, 1.0)
    L2 = solve_labor(alpha, beta, gamma, lambda_eff, 3.0)
    assert L2 > L1, "L must increase with wage for positive lambda"


def test_effective_lambda_formula():
    """lambda = lambda_keep + (lambda_ubi/I)*(1 + delta*(I-1))."""
    I, lambda_keep, lambda_ubi, delta = 2, 0.7, 0.2, 3.0
    expected = 0.7 + (0.2 / 2) * (1 + 3.0 * 1)  # = 0.7 + 0.1*4 = 1.1
    result = compute_effective_lambda(I, lambda_keep, lambda_ubi, delta)
    assert abs(result - expected) < 1e-12


def test_effective_lambda_delta_one_invariant():
    """When delta=1, lambda = lambda_keep + lambda_ubi = lambda_total (independent of I)."""
    for I in [2, 3, 5]:
        result = compute_effective_lambda(I, 0.7, 0.2, 1.0)
        assert abs(result - 0.9) < 1e-12, f"I={I}: expected 0.9 got {result}"


def test_alpha_sigma_formula():
    """alpha_Sigma = (1/alpha_1 + 1/alpha_2)^{-1}."""
    alphas = np.array([0.3, 0.4])
    expected = 1.0 / (1 / 0.3 + 1 / 0.4)
    result = compute_alpha_sigma(alphas)
    assert abs(result - expected) < 1e-12


def test_alpha_sigma_less_than_min():
    """alpha_Sigma < min(alpha_i)."""
    alphas = np.array([0.2, 0.3, 0.5])
    assert compute_alpha_sigma(alphas) < min(alphas)


def test_rho_sigma_formula():
    """rho_Sigma = alpha_Sigma * sum(rho_i/alpha_i)."""
    alphas = np.array([0.3, 0.4])
    rhos = np.array([0.5, 0.8])
    alpha_sig = compute_alpha_sigma(alphas)
    expected = alpha_sig * (0.5 / 0.3 + 0.8 / 0.4)
    result = compute_rho_sigma(alphas, rhos)
    assert abs(result - expected) < 1e-12


def test_psi_prime_numerical_gradient():
    """Check Psi'(l) against finite differences."""
    l, beta, gamma = 0.4, -0.2, -0.7
    eps = 1e-7
    numerical = (psi(l + eps, beta, gamma) - psi(l - eps, beta, gamma)) / (2 * eps)
    analytical = psi_prime(l, beta, gamma)
    assert abs(numerical - analytical) < 1e-6


def test_solve_labor_different_params():
    """Test with Figure 2 beta values (close to zero)."""
    # beta_1=-0.02 from Figure 2
    L = solve_labor(0.2, -0.02, -0.8, 0.85, 3.0)
    residual = abs(psi(L, -0.02, -0.8) - 0.2 * 0.85 * 3.0)
    assert residual < 1e-9
