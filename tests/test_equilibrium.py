import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.labor_solver import (solve_labor, compute_effective_lambda,
                               compute_alpha_sigma, compute_rho_sigma)
from src.equilibrium import (
    log_utility, compute_lcal_i, compute_epsilon_i,
    compute_epsilon_sigma, compute_response_function,
    compute_sde_coefficients, compute_aggregate_coefficients
)


def test_log_utility_formula():
    """log u(l) = beta*log(l) + gamma*log(1-l)."""
    l, beta, gamma = 0.5, -0.1, -0.5
    expected = beta * np.log(l) + gamma * np.log(1 - l)
    assert abs(log_utility(l, beta, gamma) - expected) < 1e-12


def test_lcal_i_formula():
    """Lcal_i = (1/alpha)*log_utility(L)."""
    alpha, beta, gamma, L = 0.3, -0.1, -0.5, 0.6
    expected = (1 / alpha) * (beta * np.log(L) + gamma * np.log(1 - L))
    result = compute_lcal_i(alpha, beta, gamma, L)
    assert abs(result - expected) < 1e-12


def test_epsilon_i_on_equilibrium():
    """epsilon_i(L_i) = lambda_keep*w_i*L_i + (lambda_ubi/I)*sum_j w_j*L_j."""
    wages = np.array([2.0, 3.0])
    labors = np.array([0.4, 0.5])
    lambda_keep, lambda_ubi = 0.7, 0.2
    I = 2
    agg = np.sum(wages * labors)  # = 2.0*0.4 + 3.0*0.5 = 0.8+1.5 = 2.3
    expected_0 = lambda_keep * wages[0] * labors[0] + (lambda_ubi / I) * agg
    result_0 = compute_epsilon_i(0, lambda_keep, lambda_ubi, wages, labors)
    assert abs(result_0 - expected_0) < 1e-12


def test_epsilon_sigma_formula():
    """epsilon_Sigma = (lambda_keep+lambda_ubi)*sum_i w_i*L_i."""
    wages = np.array([2.0, 3.0])
    labors = np.array([0.4, 0.5])
    lambda_keep, lambda_ubi = 0.7, 0.2
    expected = 0.9 * (2.0 * 0.4 + 3.0 * 0.5)  # = 0.9 * 2.3 = 2.07
    result = compute_epsilon_sigma(lambda_keep, lambda_ubi, wages, labors)
    assert abs(result - expected) < 1e-12


def test_income_identity_sum_eps_equals_eps_sigma():
    """sum_i epsilon_i(L_i) == epsilon_Sigma for I=2."""
    wages = np.array([2.0, 3.0])
    labors = np.array([0.4, 0.5])
    lambda_keep, lambda_ubi = 0.7, 0.2
    eps0 = compute_epsilon_i(0, lambda_keep, lambda_ubi, wages, labors)
    eps1 = compute_epsilon_i(1, lambda_keep, lambda_ubi, wages, labors)
    eps_sigma = compute_epsilon_sigma(lambda_keep, lambda_ubi, wages, labors)
    assert abs(eps0 + eps1 - eps_sigma) < 1e-12


def test_response_function_at_equilibrium():
    """Lambda_j^i(L_i) = delta*(w_i/w_j)*(L_i-L_i) + L_j = L_j."""
    L_i, L_j, w_i, w_j, delta = 0.4, 0.5, 2.0, 3.0, 2.0
    result = compute_response_function(0, 1, L_i, L_i, L_j, w_i, w_j, delta)
    assert abs(result - L_j) < 1e-12, "Response function at equilibrium must equal L_j"


def test_response_function_zero_wage():
    """If w_j=0, response should return L_j."""
    result = compute_response_function(0, 1, 0.4, 0.4, 0.5, 2.0, 0.0, 2.0)
    assert abs(result - 0.5) < 1e-12


def test_sde_coefficients_zero_sigma_w():
    """With sigma_w=0, sigma_L=0 and sigma_Lcal=0 and sigma_wL=0."""
    alpha, beta, gamma = 0.3, -0.1, -0.5
    lambda_eff, L, mu_w, sigma_w = 0.9, 0.5, 0.1, 0.0
    coeffs = compute_sde_coefficients(alpha, beta, gamma, lambda_eff, L, mu_w, sigma_w)
    assert abs(coeffs['sigma_L']) < 1e-12
    assert abs(coeffs['sigma_Lcal']) < 1e-12
    assert abs(coeffs['sigma_wL']) < 1e-12


def test_sde_coefficients_sigma_wL_formula():
    """sigma_wL = sigma_w*(L + Psi(L)/Psi'(L))."""
    from src.labor_solver import psi, psi_prime
    alpha, beta, gamma = 0.3, -0.1, -0.5
    lambda_eff, L, mu_w, sigma_w = 0.9, 0.5, 0.1, 0.1
    coeffs = compute_sde_coefficients(alpha, beta, gamma, lambda_eff, L, mu_w, sigma_w)
    Psi = psi(L, beta, gamma)
    Psi_p = psi_prime(L, beta, gamma)
    expected = sigma_w * (L + Psi / Psi_p)
    assert abs(coeffs['sigma_wL'] - expected) < 1e-12


def test_aggregate_kappa_formula():
    """kappa = alpha_Sigma*(sigma_D + sigma_epsilon - sigma_Lcal)."""
    # Use deterministic case: constant wages, zero sigma_w => sigma_epsilon=0, sigma_Lcal=0
    alphas = np.array([0.3, 0.4])
    rhos = np.array([0.5, 0.8])
    betas = np.array([-0.1, -0.15])
    gammas = np.array([-0.5, -0.7])
    lambda_keep, lambda_ubi, delta = 0.7, 0.2, 2.0
    wages = np.array([2.0, 3.0])
    lambda_eff = compute_effective_lambda(2, lambda_keep, lambda_ubi, delta)
    labors = np.array([
        solve_labor(alphas[i], betas[i], gammas[i], lambda_eff, wages[i])
        for i in range(2)
    ])
    alpha_Sigma = compute_alpha_sigma(alphas)
    mu_D, sigma_D = 0.1, 0.2
    # With sigma_w=0: sigma_epsilon=0, sigma_Lcal=0
    agents = [
        dict(alpha=alphas[i], beta=betas[i], gamma=gammas[i],
             lambda_eff=lambda_eff, L=labors[i], mu_w=0.0, sigma_w=0.0, rho=rhos[i])
        for i in range(2)
    ]
    agg = compute_aggregate_coefficients(agents, lambda_keep, lambda_ubi, mu_D, sigma_D)
    expected_kappa = alpha_Sigma * (sigma_D + 0.0 - 0.0)  # = alpha_Sigma * sigma_D
    assert abs(agg['kappa'] - expected_kappa) < 1e-10
