import numpy as np
from scipy.optimize import brentq
from typing import Union

L_EPS = 1e-10  # boundary guard for l in (0,1)
ROOT_TOL = 1e-12  # brentq tolerance


def psi(l: float, beta: float, gamma: float) -> float:
    """Log-derivative of labor utility u(l)=l^beta*(1-l)^gamma.
    Psi(l) = beta/l - gamma/(1-l).
    Strictly increasing on (0,1) when beta<0 and gamma<0."""
    return beta / l - gamma / (1 - l)


def psi_prime(l: float, beta: float, gamma: float) -> float:
    """First derivative of Psi: Psi'(l) = -beta/l^2 - gamma/(1-l)^2."""
    return -beta / l**2 - gamma / (1 - l)**2


def psi_second(l: float, beta: float, gamma: float) -> float:
    """Second derivative of Psi: Psi''(l) = 2*beta/l^3 - 2*gamma/(1-l)^3."""
    return 2 * beta / l**3 - 2 * gamma / (1 - l)**3


def solve_labor(alpha: float, beta: float, gamma: float,
                lambda_eff: float, w: float) -> float:
    """Solve Psi(L) = alpha * lambda_eff * w for unique L in (0,1).
    Uses brentq on [L_EPS, 1-L_EPS].

    Parameters
    ----------
    alpha : float, agent's risk-aversion (alpha_i > 0)
    beta : float, labor utility parameter (beta_i < 0)
    gamma : float, labor utility parameter (gamma_i < 0)
    lambda_eff : float, effective labor incentive coefficient
    w : float, wage rate (must be nonzero; positive wages assumed for Section 4)

    Returns
    -------
    float : optimal labor L_i in (0, 1)
    """
    if lambda_eff == 0.0:
        # Psi(L) = 0; find root of beta/l - gamma/(1-l) = 0
        # Solution: l* = -beta/(gamma-beta) but check validity
        rhs = 0.0
    else:
        rhs = alpha * lambda_eff * w

    def equation(l: float) -> float:
        return psi(l, beta, gamma) - rhs

    # Check bracket - Psi is increasing so bracket is guaranteed if
    # equation(L_EPS) < 0 < equation(1-L_EPS) or vice versa
    f_lo = equation(L_EPS)
    f_hi = equation(1.0 - L_EPS)

    if f_lo * f_hi > 0:
        # Both same sign - solution may be outside (0,1) range or rhs too extreme
        # Return boundary closest to zero
        if abs(f_lo) < abs(f_hi):
            return float(L_EPS)
        else:
            return float(1.0 - L_EPS)

    return brentq(equation, L_EPS, 1.0 - L_EPS, xtol=ROOT_TOL, rtol=ROOT_TOL)


def solve_labor_vectorized(alpha: float, beta: float, gamma: float,
                            lambda_eff: float, w_array: np.ndarray) -> np.ndarray:
    """Vectorized version of solve_labor over an array of wage values."""
    return np.array([solve_labor(alpha, beta, gamma, lambda_eff, w) for w in w_array])


def compute_effective_lambda(I: int, lambda_keep: float,
                              lambda_ubi: float, delta: float) -> float:
    """Compute effective labor-incentive coefficient.
    lambda = lambda_keep + (lambda_ubi / I) * (1 + delta*(I-1)).

    Parameters
    ----------
    I : int, number of agents
    lambda_keep : float, fraction of own income kept
    lambda_ubi : float, fraction redistributed as UBI
    delta : float, social influence parameter

    Returns
    -------
    float : effective lambda
    """
    return lambda_keep + (lambda_ubi / I) * (1 + delta * (I - 1))


def compute_alpha_sigma(alphas: np.ndarray) -> float:
    """Compute aggregate risk-tolerance: alpha_Sigma = (sum_i 1/alpha_i)^(-1)."""
    return 1.0 / np.sum(1.0 / alphas)


def compute_rho_sigma(alphas: np.ndarray, rhos: np.ndarray) -> float:
    """Compute aggregate rho_Sigma = alpha_Sigma * sum_i (rho_i / alpha_i)."""
    alpha_sigma = compute_alpha_sigma(alphas)
    return alpha_sigma * np.sum(rhos / alphas)


def validate_params(alphas, rhos, betas, gammas, lambda_keep, lambda_ubi,
                    pi0_sum=None, theta0_sum=None):
    """Validate that all parameters satisfy the paper's admissibility conditions.
    Raises ValueError if any condition is violated.
    """
    assert np.all(alphas > 0), "alpha_i must be > 0"
    assert np.all(rhos >= 0), "rho_i must be >= 0"
    assert np.all(betas < 0), "beta_i must be < 0"
    assert np.all(gammas < 0), "gamma_i must be < 0"
    assert 0 <= lambda_keep <= 1, "lambda_keep must be in [0,1]"
    assert 0 <= lambda_ubi <= 1 - lambda_keep, "lambda_ubi must be in [0, 1-lambda_keep]"
    if pi0_sum is not None:
        assert abs(pi0_sum - 1.0) < 1e-9, "sum of pi_{i,0-} must equal 1"
    if theta0_sum is not None:
        assert abs(theta0_sum - 1.0) < 1e-9, "sum of theta_{i,0-} must equal 1"
