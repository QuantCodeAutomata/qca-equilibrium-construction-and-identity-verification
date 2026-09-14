import numpy as np
from typing import List, Tuple, Dict, Optional
from src.labor_solver import (psi, psi_prime, psi_second, solve_labor,
                               compute_effective_lambda, compute_alpha_sigma,
                               compute_rho_sigma, L_EPS)


def log_utility(l: float, beta: float, gamma: float) -> float:
    """Log of labor utility: log u_i(l) = beta*log(l) + gamma*log(1-l)."""
    return beta * np.log(l) + gamma * np.log(1 - l)


def compute_lcal_i(alpha: float, beta: float, gamma: float, L: float) -> float:
    """Compute Lcal_i = (1/alpha_i) * log u_i(L_i)."""
    return (1.0 / alpha) * log_utility(L, beta, gamma)


def compute_epsilon_i(i: int, lambda_keep: float, lambda_ubi: float,
                       wages: np.ndarray, labors: np.ndarray) -> float:
    """Compute perceived income epsilon_i on equilibrium labor choices.
    epsilon_i(L_i) = lambda_keep * w_i * L_i + (lambda_ubi/I) * sum_j w_j * L_j.
    """
    I = len(wages)
    return lambda_keep * wages[i] * labors[i] + (lambda_ubi / I) * np.sum(wages * labors)


def compute_epsilon_sigma(lambda_keep: float, lambda_ubi: float,
                           wages: np.ndarray, labors: np.ndarray) -> float:
    """Aggregate equilibrium perceived income.
    epsilon_Sigma = (lambda_keep + lambda_ubi) * sum_i w_i * L_i.
    """
    return (lambda_keep + lambda_ubi) * np.sum(wages * labors)


def compute_response_function(i: int, j: int, l_i: float,
                               L_i: float, L_j: float,
                               w_i: float, w_j: float, delta: float) -> float:
    """Compute perceived response of agent j to agent i's deviation l around L_i.
    Lambda_{j,t}^i(l) = delta * (w_i/w_j) * (l - L_i) + L_j.
    If w_j == 0, return L_j.
    """
    if abs(w_j) < 1e-15:
        return L_j
    return delta * (w_i / w_j) * (l_i - L_i) + L_j


def compute_sde_coefficients(alpha: float, beta: float, gamma: float,
                              lambda_eff: float, L: float,
                              mu_w: float, sigma_w: float
                              ) -> Dict[str, float]:
    """Compute SDE drift and diffusion coefficients for dL_i, dLcal_i, d(w_i*L_i).

    From the paper's Ito's lemma formulas:

    dL_i coefficients:
      sigma_L = alpha * lambda * sigma_w / Psi'(L)
      mu_L = alpha * lambda * mu_w / Psi'(L) - (alpha^2 * lambda^2 * sigma_w^2 / 2) * Psi''(L) / Psi'(L)^3

    dLcal_i coefficients:
      sigma_Lcal = lambda * sigma_w * Psi(L) / Psi'(L)
      mu_Lcal = lambda * mu_w * Psi(L) / Psi'(L)
              - (alpha * lambda^2 * sigma_w^2 / 2) * Psi''(L) * Psi(L) / Psi'(L)^3
              + alpha * lambda^2 * sigma_w^2 / (2 * Psi'(L))

    d(w_i * L_i) coefficients:
      sigma_wL = sigma_w * L + sigma_w * Psi(L) / Psi'(L)
      mu_wL = mu_w * L + (alpha * lambda * sigma_w^2 + mu_w * Psi(L)) / Psi'(L)
            - (alpha * lambda * sigma_w^2 / 2) * Psi(L) * Psi''(L) / Psi'(L)^3

    Returns dict with keys: sigma_L, mu_L, sigma_Lcal, mu_Lcal, sigma_wL, mu_wL
    """
    Psi = psi(L, beta, gamma)
    Psi_p = psi_prime(L, beta, gamma)
    Psi_pp = psi_second(L, beta, gamma)

    lam2_sw2 = lambda_eff**2 * sigma_w**2

    # dL_i
    sigma_L = alpha * lambda_eff * sigma_w / Psi_p
    mu_L = (alpha * lambda_eff * mu_w / Psi_p
            - (alpha**2 * lam2_sw2 / 2.0) * Psi_pp / Psi_p**3)

    # dLcal_i
    sigma_Lcal = lambda_eff * sigma_w * Psi / Psi_p
    mu_Lcal = (lambda_eff * mu_w * Psi / Psi_p
               - (alpha * lam2_sw2 / 2.0) * Psi_pp * Psi / Psi_p**3
               + alpha * lam2_sw2 / (2.0 * Psi_p))

    # d(w_i * L_i)
    sigma_wL = sigma_w * L + sigma_w * Psi / Psi_p
    mu_wL = (mu_w * L
             + (alpha * lambda_eff * sigma_w**2 + mu_w * Psi) / Psi_p
             - (alpha * lambda_eff * sigma_w**2 / 2.0) * Psi * Psi_pp / Psi_p**3)

    return {
        'sigma_L': sigma_L, 'mu_L': mu_L,
        'sigma_Lcal': sigma_Lcal, 'mu_Lcal': mu_Lcal,
        'sigma_wL': sigma_wL, 'mu_wL': mu_wL,
        'Psi': Psi, 'Psi_p': Psi_p, 'Psi_pp': Psi_pp
    }


def compute_aggregate_coefficients(
        agents: List[Dict], lambda_keep: float, lambda_ubi: float,
        mu_D: float, sigma_D: float
) -> Dict[str, float]:
    """Compute aggregate drift/diffusion for epsilon_Sigma and Lcal_Sigma,
    then compute kappa (market price of risk) and r (interest-rate-type coefficient).

    agents: list of dicts, each with keys:
        alpha, beta, gamma, lambda_eff (agent-level lambda),
        L (solved labor), mu_w, sigma_w, rho
    lambda_keep, lambda_ubi: policy parameters
    mu_D, sigma_D: dividend process coefficients

    Returns dict with kappa, r, sigma_epsilon, mu_epsilon, sigma_Lcal, mu_Lcal,
    alpha_Sigma, rho_Sigma.
    """
    I = len(agents)
    alphas = np.array([a['alpha'] for a in agents])
    rhos = np.array([a['rho'] for a in agents])
    alpha_Sigma = compute_alpha_sigma(alphas)
    rho_Sigma = compute_rho_sigma(alphas, rhos)

    sigma_Lcal_sum = 0.0
    mu_Lcal_sum = 0.0
    sigma_wL_sum = 0.0
    mu_wL_sum = 0.0

    for ag in agents:
        coeffs = compute_sde_coefficients(
            ag['alpha'], ag['beta'], ag['gamma'],
            ag['lambda_eff'], ag['L'], ag['mu_w'], ag['sigma_w']
        )
        sigma_Lcal_sum += coeffs['sigma_Lcal']
        mu_Lcal_sum += coeffs['mu_Lcal']
        sigma_wL_sum += coeffs['sigma_wL']
        mu_wL_sum += coeffs['mu_wL']

    lambda_total = lambda_keep + lambda_ubi
    sigma_epsilon = lambda_total * sigma_wL_sum
    mu_epsilon = lambda_total * mu_wL_sum
    sigma_Lcal = sigma_Lcal_sum
    mu_Lcal = mu_Lcal_sum

    # Market price of risk: kappa = alpha_Sigma * (sigma_D + sigma_epsilon - sigma_Lcal)
    kappa = alpha_Sigma * (sigma_D + sigma_epsilon - sigma_Lcal)

    # Interest-rate-type coefficient (statewise, evaluated without BSDE Z terms)
    # r = rho_Sigma + alpha_Sigma*(mu_D + mu_epsilon - mu_Lcal) - 0.5*kappa^2
    r = rho_Sigma + alpha_Sigma * (mu_D + mu_epsilon - mu_Lcal) - 0.5 * kappa**2

    return {
        'kappa': kappa, 'r': r,
        'sigma_epsilon': sigma_epsilon, 'mu_epsilon': mu_epsilon,
        'sigma_Lcal': sigma_Lcal, 'mu_Lcal': mu_Lcal,
        'alpha_Sigma': alpha_Sigma, 'rho_Sigma': rho_Sigma
    }
