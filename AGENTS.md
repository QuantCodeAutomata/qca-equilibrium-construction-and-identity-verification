# Project Knowledge — AGENTS.md

## Project Overview
Equilibrium model with labor supply, UBI policy, BSDE-based asset pricing, and welfare analysis.

## Source Module Interfaces

### `src/labor_solver.py`
- `psi(l, beta, gamma)` — log-derivative of utility, monotone on (0,1) for beta<0, gamma<0
- `solve_labor(alpha, beta, gamma, lambda_eff, w)` — brentq in (L_EPS, 1-L_EPS); returns boundary if bracket fails
- `compute_effective_lambda(I, lambda_keep, lambda_ubi, delta)` = `lambda_keep + (lambda_ubi/I)*(1+delta*(I-1))`
- `validate_params(alphas, rhos, betas, gammas, lambda_keep, lambda_ubi, pi0_sum=None, theta0_sum=None)` — uses `assert`, raises AssertionError; constraint: `lambda_ubi <= 1 - lambda_keep`

### `src/equilibrium.py`
- `compute_aggregate_coefficients(agents, lambda_keep, lambda_ubi, mu_D, sigma_D)` — agents is list of dicts with keys: alpha, beta, gamma, lambda_eff, L, mu_w, sigma_w, rho. Returns: kappa, r, sigma_epsilon, mu_epsilon, sigma_Lcal, mu_Lcal, alpha_Sigma, rho_Sigma
- `r = rho_Sigma + alpha_Sigma*(mu_D+mu_eps-mu_Lcal) - 0.5*kappa^2` (defined in source)
- BSDE's C parameter = `r + 0.5*kappa^2 = rho_Sigma + alpha_Sigma*(mu_D+mu_epsilon-mu_Lcal)`

### `src/bsde_solver.py`
- `solve_deterministic_bsde(T, N_steps, C, kappa, alphas, rhos, epsilon_i, lcal_i, alpha_Sigma, rho_Sigma)` → BSDEResult; `a` shape (N+1,), `Y` shape (I, N+1); `residuals` dict with `a_terminal` and `Y_terminal_max`
- `solve_stochastic_bsde(T, N_steps, M_paths, C_paths, kappa_paths, alphas, rhos, epsilon_i_paths, lcal_i_paths, alpha_Sigma, rho_Sigma, dB)` → BSDEResult; `a` shape (N+1,M), `Y` shape (I,N+1,M); `residuals` is empty dict `{}`
- **Known issue**: stochastic BSDE Picard iterations can overflow/diverge, producing NaN in `a_paths`. Guard with `np.nanmean(np.exp(np.clip(...)))` when computing exp(a).

### `src/welfare.py`
- `compute_aggregate_welfare_formula(alphas, betas, gammas, labors, rhos, T, D0, epsilon_sigma_0, a0, alpha_Sigma)` — `betas`, `gammas`, `labors` args are accepted but **not used** in the formula; formula assumes market-clearing initial wealth
- `compute_certainty_equivalent(alpha, beta, gamma, L, rho, T, X_i0_over_A0, Y_i0)` — uses `Y_i(0)` from BSDE

### `src/comparative_statics.py`
- `labor_sweep(I, alphas, betas, gammas, wages, lambda_total, lambda_ubi_grid, delta)` → DataFrame with columns: lambda_ubi, lambda_keep, lambda_eff, lambda_positive, L_1..L_I, agg_labor_income, epsilon_Sigma
- `extreme_policy_comparison(I, alphas, betas, gammas, wages, delta_grid)` → DataFrame; PC uses lambda_eff=1.0 (constant), SM uses lambda_eff=(1/I)+delta*(I-1)/I

## Key Mathematical Facts
- `lambda_eff_SM > lambda_eff_PC = 1` iff `delta > 1` (for I=2: lambda_eff_SM = 0.5+0.5*delta)
- `d(agg_income)/d(lambda_ubi) > 0` iff `delta > 1` (because `d(lambda_eff)/d(lambda_ubi) = (delta-1)/2` for I=2)
- BSDE C parameter: `C = rho_Sigma + alpha_Sigma*(mu_D + mu_epsilon - mu_Lcal)`
- Terminal conditions: `a(T) = 0`, `Y_i(T) = epsilon_i - Lcal_i`
- Income identity: `sum_i epsilon_i = epsilon_Sigma`
- Response consistency: `Lambda_j^i(L_i, L_i, L_j, w_i, w_j, delta)` = `delta*(w_i/w_j)*(l_i - L_i) + L_j`; at equilibrium (l_i = L_i) returns `L_j`, so residual = 0

## Experiment Files (`exp/`)
| File | Purpose | Outputs |
|------|---------|---------|
| exp1_equilibrium.py | 3 regimes, BSDE solve, identity verification | exp1_residuals.csv, exp1_equilibrium.png |
| exp2_comparative_statics.py | labor_sweep, monotonicity in delta | exp2_data.csv, exp2_comparative_statics.png |
| exp3_figure2.py | kappa/r vs wage grid, Figure 2 replication | exp3_data.csv, exp3_figure2.png |
| exp4_welfare.py | welfare sweep vs lambda_ubi, CE formula | exp4_data.csv, exp4_welfare.png |
| exp5_extreme_policy.py | PC vs SM, heterogeneous+homogeneous wages | exp5_data.csv, exp5_extreme_policy.png |

## Common Patterns in Exp Files
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib
matplotlib.use('Agg')  # must be BEFORE pyplot import
import matplotlib.pyplot as plt
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
```
