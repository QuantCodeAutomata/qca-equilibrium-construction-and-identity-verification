import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class BSDEResult:
    """Container for BSDE solution."""
    t_grid: np.ndarray        # Time grid [t_0, ..., t_N=T] shape (N+1,)
    a: np.ndarray             # a(t) values, shape (N+1,) or (N+1, M) for M paths
    Z_a: np.ndarray           # Z_a(t) values, same shape
    Y: np.ndarray             # Y_i(t) values, shape (I, N+1) or (I, N+1, M)
    Z: np.ndarray             # Z_i(t) values, same shape as Y
    residuals: Dict[str, float] = field(default_factory=dict)
    mode: str = 'deterministic'  # 'deterministic' or 'stochastic'


def solve_deterministic_bsde(
        T: float,
        N_steps: int,
        C: float,             # C = rho_Sigma + alpha_Sigma*(mu_D+mu_eps-mu_Lcal)
        kappa: float,         # kappa = alpha_Sigma*(sigma_D+sigma_eps-sigma_Lcal)
        alphas: np.ndarray,   # shape (I,)
        rhos: np.ndarray,     # shape (I,)
        epsilon_i: np.ndarray,# shape (I,), constant perceived incomes
        lcal_i: np.ndarray,   # shape (I,), constant Lcal_i
        alpha_Sigma: float,
        rho_Sigma: float,
) -> BSDEResult:
    """Solve the deterministic BSDE system (Z_a=0, Z_i=0).

    BSDE for a (reversed-time s = T - t):
      da_hat/ds = exp(-a_hat) - K, a_hat(0) = 0
    where K = C - 0.5*kappa^2 and C = rho_Sigma + alpha_Sigma*(mu_D+mu_eps-mu_Lcal).

    BSDE for Y_i (reversed time):
      dY_hat_i/ds = exp(-a_hat)*(c_i - Y_hat_i) + (1/alpha_i)*(rho_i - (1+a_hat)*exp(-a_hat) + 0.5*kappa^2)
      Y_hat_i(0) = c_i where c_i = epsilon_i - lcal_i

    Parameters
    ----------
    C : float = rho_Sigma + alpha_Sigma*(mu_D + mu_epsilon - mu_Lcal)
    kappa : float = alpha_Sigma*(sigma_D + sigma_epsilon - sigma_Lcal)
    """
    I = len(alphas)
    c_vec = epsilon_i - lcal_i  # terminal conditions Y_i(T) = epsilon_i - lcal_i
    K = C - 0.5 * kappa**2

    # System in reversed time: state = [a_hat, Y_hat_1, ..., Y_hat_I]
    def rhs_reversed(s: float, state: np.ndarray) -> np.ndarray:
        a_hat = state[0]
        Y_hat = state[1:]
        exp_neg_a = np.exp(-a_hat)

        da_hat = exp_neg_a - K
        dY_hat = np.zeros(I)
        for i in range(I):
            dY_hat[i] = (exp_neg_a * (c_vec[i] - Y_hat[i])
                         + (1.0 / alphas[i]) * (rhos[i] - (1 + a_hat) * exp_neg_a
                                                  + 0.5 * kappa**2))
        return np.concatenate([[da_hat], dY_hat])

    # Initial conditions in reversed time
    y0_reversed = np.concatenate([[0.0], c_vec])  # a_hat(0)=0, Y_hat_i(0)=c_i

    # Solve over [0, T] in reversed time
    s_eval = np.linspace(0, T, N_steps + 1)
    sol = solve_ivp(rhs_reversed, [0, T], y0_reversed, t_eval=s_eval,
                    method='Radau', rtol=1e-10, atol=1e-12)

    # Map back: a(t) = a_hat(T-t), Y_i(t) = Y_hat_i(T-t)
    t_grid = T - sol.t[::-1]  # forward time
    a_vals = sol.y[0][::-1]
    Y_vals = sol.y[1:][:, ::-1]  # shape (I, N+1)

    # Z values are zero in deterministic case
    Z_a_vals = np.zeros_like(a_vals)
    Z_vals = np.zeros_like(Y_vals)

    # Check terminal: a(T) ≈ 0, Y_i(T) ≈ c_i
    residuals = {
        'a_terminal': abs(a_vals[-1]),
        'Y_terminal_max': np.max(np.abs(Y_vals[:, -1] - c_vec))
    }

    return BSDEResult(
        t_grid=t_grid,
        a=a_vals,
        Z_a=Z_a_vals,
        Y=Y_vals,
        Z=Z_vals,
        residuals=residuals,
        mode='deterministic'
    )


def solve_stochastic_bsde(
        T: float,
        N_steps: int,
        M_paths: int,
        C_paths: np.ndarray,    # shape (N+1, M), or scalar if constant
        kappa_paths: np.ndarray,# shape (N+1, M), or scalar
        alphas: np.ndarray,     # shape (I,)
        rhos: np.ndarray,       # shape (I,)
        epsilon_i_paths: np.ndarray,  # shape (I, N+1, M)
        lcal_i_paths: np.ndarray,     # shape (I, N+1, M)
        alpha_Sigma: float,
        rho_Sigma: float,
        dB: np.ndarray,         # Brownian increments, shape (N, M)
) -> BSDEResult:
    """Solve the stochastic BSDE system using backward Euler regression.

    This implements a least-squares Monte Carlo regression approach
    for the one-factor quadratic BSDE:
      da = Z_a dB + [C_t - 0.5*(kappa_t - Z_a)^2 - exp(-a)] dt
    with a_T = 0.

    Then conditional on a, Z_a, solves each Y_i BSDE.

    Uses polynomial regression basis for conditional expectations.
    Custom implementation - Context7 found no library equivalent for this
    specific quadratic BSDE form.
    """
    I = len(alphas)
    h = T / N_steps
    t_grid = np.linspace(0, T, N_steps + 1)

    # Initialize
    a_paths = np.zeros((N_steps + 1, M_paths))
    Z_a_paths = np.zeros((N_steps + 1, M_paths))
    Y_paths = np.zeros((I, N_steps + 1, M_paths))
    Z_paths = np.zeros((I, N_steps + 1, M_paths))

    # Terminal conditions
    a_paths[N_steps] = 0.0
    for i in range(I):
        Y_paths[i, N_steps] = epsilon_i_paths[i, N_steps] - lcal_i_paths[i, N_steps]

    def poly_regression_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
        """Build polynomial basis matrix for regression."""
        cols = [np.ones(len(x))]
        for d in range(1, degree + 1):
            cols.append(x**d)
        return np.column_stack(cols)

    def regress_conditional_expectation(Y_next: np.ndarray,
                                         X_state: np.ndarray) -> np.ndarray:
        """Estimate E[Y_next | X_state] using polynomial regression."""
        basis = poly_regression_basis(X_state)
        coeffs, _, _, _ = np.linalg.lstsq(basis, Y_next, rcond=None)
        return basis @ coeffs

    # Backward pass for a
    for n in range(N_steps - 1, -1, -1):
        n_next = n + 1
        a_n1 = a_paths[n_next]

        if C_paths.ndim == 0:
            C_n = float(C_paths)
        else:
            C_n = C_paths[n]
        if kappa_paths.ndim == 0:
            kappa_n = float(kappa_paths)
        else:
            kappa_n = kappa_paths[n]

        dB_n = dB[n]  # shape (M,)

        # Z_a estimate: Z_{a,n} ≈ a_{n+1} * dB_n / h
        Z_a_n = a_n1 * dB_n / h

        # E[a_{n+1} | F_n] via regression on dB_n
        E_a_n1 = regress_conditional_expectation(a_n1, dB_n)

        # Backward Euler update for a with Picard iterations (implicit in exp(-a_n))
        a_n_est = E_a_n1.copy()
        for _ in range(5):
            driver = C_n - 0.5 * (kappa_n - Z_a_n)**2 - np.exp(-a_n_est)
            a_n_est = E_a_n1 - h * driver

        a_paths[n] = a_n_est
        Z_a_paths[n] = Z_a_n

    # Backward pass for each Y_i
    for i in range(I):
        for n in range(N_steps - 1, -1, -1):
            n_next = n + 1
            Y_i_n1 = Y_paths[i, n_next]
            a_n = a_paths[n]
            Z_a_n = Z_a_paths[n]

            if C_paths.ndim == 0:
                kappa_n = float(kappa_paths)
            else:
                kappa_n = kappa_paths[n]

            dB_n = dB[n]
            eps_i_n = epsilon_i_paths[i, n]
            lcal_i_n = lcal_i_paths[i, n]

            # Z_i estimate
            Z_i_n = Y_i_n1 * dB_n / h

            # E[Y_i_{n+1}] via regression
            E_Y_n1 = regress_conditional_expectation(Y_i_n1, dB_n)

            # Driver for Y_i BSDE; Picard iterations (implicit in Y_n_est)
            Y_n_est = E_Y_n1.copy()
            kappa_minus_Za = kappa_n - Z_a_n
            for _ in range(5):
                driver_Yi = (1.0 / alphas[i]) * (
                    -rhos[i]
                    + (1 + a_n + alphas[i] * (Y_n_est - eps_i_n + lcal_i_n)) * np.exp(-a_n)
                    - 0.5 * kappa_minus_Za**2
                    + alphas[i] * Z_i_n * kappa_minus_Za
                )
                Y_n_est = E_Y_n1 - h * driver_Yi

            Y_paths[i, n] = Y_n_est
            Z_paths[i, n] = Z_i_n

    return BSDEResult(
        t_grid=t_grid,
        a=a_paths,
        Z_a=Z_a_paths,
        Y=Y_paths,
        Z=Z_paths,
        residuals={},
        mode='stochastic'
    )
