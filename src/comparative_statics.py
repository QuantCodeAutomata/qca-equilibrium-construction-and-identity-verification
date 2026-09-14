import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from src.labor_solver import (solve_labor, compute_effective_lambda,
                               psi, psi_prime, psi_second, L_EPS)
from src.equilibrium import compute_epsilon_i, compute_epsilon_sigma


def labor_sweep(
        I: int, alphas: np.ndarray, betas: np.ndarray, gammas: np.ndarray,
        wages: np.ndarray, lambda_total: float,
        lambda_ubi_grid: np.ndarray, delta: float
) -> pd.DataFrame:
    """Sweep labor choices over a grid of lambda_ubi values at fixed lambda_total.

    For each lambda_ubi on the grid:
    - lambda_keep = lambda_total - lambda_ubi
    - lambda_eff = effective_lambda(I, lambda_keep, lambda_ubi, delta)
    - Solve L_i for each agent
    - Compute aggregate labor income, epsilon_Sigma
    - Record monotonicity sign

    Returns DataFrame with columns:
    lambda_ubi, lambda_keep, lambda_eff, lambda_positive,
    L_1, ..., L_I, agg_labor_income, epsilon_Sigma
    """
    records = []
    for lubi in lambda_ubi_grid:
        lkeep = lambda_total - lubi
        leff = compute_effective_lambda(I, lkeep, lubi, delta)
        lambda_positive = leff > 0

        labors = np.array([
            solve_labor(alphas[i], betas[i], gammas[i], leff, wages[i])
            for i in range(I)
        ])
        agg_income = np.sum(wages * labors)
        eps_sigma = compute_epsilon_sigma(lkeep, lubi, wages, labors)

        row = {
            'lambda_ubi': lubi,
            'lambda_keep': lkeep,
            'lambda_eff': leff,
            'lambda_positive': lambda_positive,
            'agg_labor_income': agg_income,
            'epsilon_Sigma': eps_sigma,
        }
        for i in range(I):
            row[f'L_{i+1}'] = labors[i]
        records.append(row)

    return pd.DataFrame(records)


def extreme_policy_comparison(
        I: int, alphas: np.ndarray, betas: np.ndarray, gammas: np.ndarray,
        wages: np.ndarray, delta_grid: np.ndarray
) -> pd.DataFrame:
    """Compare pure capitalism (lambda_keep=1, lambda_ubi=0) vs
    social market economy (lambda_keep=0, lambda_ubi=1) across delta values.

    Returns DataFrame with per-delta comparison of L_i and aggregate labor income.
    """
    # Pure capitalism: lambda_eff_PC = 1 (independent of delta)
    lambda_eff_PC = 1.0
    labors_PC = np.array([
        solve_labor(alphas[i], betas[i], gammas[i], lambda_eff_PC, wages[i])
        for i in range(I)
    ])
    agg_income_PC = np.sum(wages * labors_PC)
    epsilon_sigma_PC = agg_income_PC  # (lambda_keep+lambda_ubi)=1

    records = []
    for delta in delta_grid:
        # Social market economy: lambda_eff_SM = (1/I) + delta*(I-1)/I
        lambda_eff_SM = (1.0 / I) + delta * (I - 1) / I
        labors_SM = np.array([
            solve_labor(alphas[i], betas[i], gammas[i], lambda_eff_SM, wages[i])
            for i in range(I)
        ])
        agg_income_SM = np.sum(wages * labors_SM)
        epsilon_sigma_SM = agg_income_SM  # (lambda_keep+lambda_ubi)=1

        delta_agg = agg_income_SM - agg_income_PC
        row = {
            'delta': delta,
            'lambda_eff_PC': lambda_eff_PC,
            'lambda_eff_SM': lambda_eff_SM,
            'agg_income_PC': agg_income_PC,
            'agg_income_SM': agg_income_SM,
            'delta_agg_income': delta_agg,
            'SM_dominates': delta_agg > 0,
            'epsilon_sigma_PC': epsilon_sigma_PC,
            'epsilon_sigma_SM': epsilon_sigma_SM,
        }
        for i in range(I):
            row[f'L_{i+1}_PC'] = labors_PC[i]
            row[f'L_{i+1}_SM'] = labors_SM[i]
            row[f'delta_L_{i+1}'] = labors_SM[i] - labors_PC[i]
        records.append(row)

    return pd.DataFrame(records)
