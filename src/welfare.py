import numpy as np
from typing import Optional


def compute_annuity_factor(rho: float, T: float) -> float:
    """Compute K_i(T) = int_0^T exp(-rho*t) dt + exp(-rho*T).

    For rho > 0: K_i(T) = (1 - exp(-rho*T))/rho + exp(-rho*T)
    For rho == 0: K_i(T) = T + 1

    Parameters
    ----------
    rho : float, discount rate rho_i >= 0
    T : float, time horizon

    Returns
    -------
    float : K_i(T)
    """
    if rho == 0.0:
        return T + 1.0
    return (1.0 - np.exp(-rho * T)) / rho + np.exp(-rho * T)


def compute_certainty_equivalent(
        alpha: float, beta: float, gamma: float,
        L: float, rho: float, T: float,
        X_i0_over_A0: float, Y_i0: float
) -> float:
    """Compute certainty equivalent CE_i for agent i in deterministic setting.

    CE_i = (1/alpha_i) * log u_i(L_i) + (1/alpha_i) * log K_i(T)
           + X_{i,0}/A_0 + Y_{i,0}

    where log u_i(L_i) = beta_i*log(L_i) + gamma_i*log(1-L_i).

    Parameters
    ----------
    alpha : float, risk-aversion alpha_i > 0
    beta, gamma : float, utility parameters
    L : float, equilibrium labor L_i in (0,1)
    rho : float, discount rate rho_i >= 0
    T : float, time horizon
    X_i0_over_A0 : float, X_{i,0}/A_0 (initial wealth relative to annuity)
    Y_i0 : float, Y_i(0) from BSDE

    Returns
    -------
    float : certainty equivalent CE_i
    """
    log_u = beta * np.log(L) + gamma * np.log(1.0 - L)
    K = compute_annuity_factor(rho, T)
    return (1.0 / alpha) * log_u + (1.0 / alpha) * np.log(K) + X_i0_over_A0 + Y_i0


def compute_aggregate_welfare_formula(
        alphas: np.ndarray, betas: np.ndarray, gammas: np.ndarray,
        labors: np.ndarray, rhos: np.ndarray, T: float,
        D0: float, epsilon_sigma_0: float,
        a0: float, alpha_Sigma: float
) -> float:
    """Compute aggregate welfare using the paper's formula.

    sum_i CE_i = 1 + D_0 + epsilon_Sigma_0 - a_0/alpha_Sigma
                + sum_i (1/alpha_i) * log K_i(T)

    Note: This formula is derived assuming the wealth market clearing
    and aggregate consistency conditions.

    Parameters
    ----------
    D0 : float, initial dividend value
    epsilon_sigma_0 : float, initial aggregate perceived income
    a0 : float, a(0) from BSDE
    alpha_Sigma : float, aggregate risk-tolerance

    Returns
    -------
    float : aggregate certainty equivalent
    """
    I = len(alphas)
    sum_log_K = sum(
        (1.0 / alphas[i]) * np.log(compute_annuity_factor(rhos[i], T))
        for i in range(I)
    )
    return 1.0 + D0 + epsilon_sigma_0 - a0 / alpha_Sigma + sum_log_K
