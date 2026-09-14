import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import pandas as pd
from src.comparative_statics import labor_sweep, extreme_policy_comparison
from src.labor_solver import compute_effective_lambda

# Standard test parameters
I = 2
ALPHAS = np.array([0.3, 0.4])
BETAS = np.array([-0.1, -0.15])
GAMMAS = np.array([-0.5, -0.7])
WAGES = np.array([2.0, 3.0])
LAMBDA_TOTAL = 0.9


def test_labor_sweep_returns_dataframe():
    """labor_sweep must return a pandas DataFrame."""
    grid = np.linspace(0, LAMBDA_TOTAL, 10)
    df = labor_sweep(I, ALPHAS, BETAS, GAMMAS, WAGES, LAMBDA_TOTAL, grid, delta=1.0)
    assert isinstance(df, pd.DataFrame)


def test_labor_sweep_correct_columns():
    """DataFrame must contain expected columns."""
    grid = np.linspace(0, LAMBDA_TOTAL, 5)
    df = labor_sweep(I, ALPHAS, BETAS, GAMMAS, WAGES, LAMBDA_TOTAL, grid, delta=1.0)
    assert 'lambda_ubi' in df.columns
    assert 'lambda_eff' in df.columns
    assert 'agg_labor_income' in df.columns
    assert 'L_1' in df.columns
    assert 'L_2' in df.columns


def test_labor_sweep_delta_one_invariant():
    """When delta=1, lambda_eff=lambda_total regardless of lambda_ubi."""
    grid = np.linspace(0, LAMBDA_TOTAL, 10)
    df = labor_sweep(I, ALPHAS, BETAS, GAMMAS, WAGES, LAMBDA_TOTAL, grid, delta=1.0)
    # All lambda_eff should equal lambda_total
    assert np.allclose(df['lambda_eff'], LAMBDA_TOTAL, atol=1e-10)


def test_labor_sweep_delta_one_constant_labor():
    """When delta=1, L_i should be constant across lambda_ubi values."""
    grid = np.linspace(0, LAMBDA_TOTAL, 20)
    df = labor_sweep(I, ALPHAS, BETAS, GAMMAS, WAGES, LAMBDA_TOTAL, grid, delta=1.0)
    assert df['L_1'].std() < 1e-10, "L_1 must be constant when delta=1"
    assert df['L_2'].std() < 1e-10, "L_2 must be constant when delta=1"


def test_labor_sweep_delta_pos_increasing():
    """When delta>1, L_i should increase with lambda_ubi."""
    grid = np.linspace(0.01, LAMBDA_TOTAL - 0.01, 20)
    df = labor_sweep(I, ALPHAS, BETAS, GAMMAS, WAGES, LAMBDA_TOTAL, grid, delta=3.0)
    # Filter where lambda_eff > 0
    df_pos = df[df['lambda_positive'] == True]
    diffs = np.diff(df_pos['agg_labor_income'].values)
    assert np.all(diffs > -1e-10), "Aggregate income should be non-decreasing for delta>1"


def test_labor_sweep_delta_neg_decreasing():
    """When delta<1 and lambda_eff>0, L_i should decrease with lambda_ubi."""
    grid = np.linspace(0.01, LAMBDA_TOTAL - 0.01, 20)
    df = labor_sweep(I, ALPHAS, BETAS, GAMMAS, WAGES, LAMBDA_TOTAL, grid, delta=-0.5)
    df_pos = df[df['lambda_positive'] == True]
    if len(df_pos) > 1:
        diffs = np.diff(df_pos['agg_labor_income'].values)
        assert np.all(diffs < 1e-10), "Aggregate income should be non-increasing for delta<1"


def test_extreme_policy_delta_one_equal():
    """When delta=1, SM and PC should have equal aggregate incomes."""
    df = extreme_policy_comparison(I, ALPHAS, BETAS, GAMMAS, WAGES, np.array([1.0]))
    row = df[df['delta'] == 1.0].iloc[0]
    assert abs(row['delta_agg_income']) < 1e-10


def test_extreme_policy_delta_pos_SM_dominates():
    """When delta>1, SM aggregate income > PC aggregate income."""
    df = extreme_policy_comparison(I, ALPHAS, BETAS, GAMMAS, WAGES, np.array([2.0, 3.0]))
    for _, row in df.iterrows():
        assert row['delta_agg_income'] > 0, f"SM should dominate at delta={row['delta']}"


def test_extreme_policy_delta_neg_PC_dominates():
    """When delta<1, PC aggregate income > SM aggregate income."""
    df = extreme_policy_comparison(I, ALPHAS, BETAS, GAMMAS, WAGES, np.array([-1.0, -0.5]))
    for _, row in df.iterrows():
        assert row['delta_agg_income'] < 0, f"PC should dominate at delta={row['delta']}"


def test_extreme_policy_lambda_eff_SM_formula():
    """lambda_eff_SM = (1/I) + delta*(I-1)/I for I=2: 0.5 + 0.5*delta."""
    deltas = np.array([0.0, 1.0, 2.0])
    df = extreme_policy_comparison(I, ALPHAS, BETAS, GAMMAS, WAGES, deltas)
    for _, row in df.iterrows():
        d = row['delta']
        expected_sm = 0.5 + 0.5 * d
        assert abs(row['lambda_eff_SM'] - expected_sm) < 1e-12


def test_extreme_policy_PC_constant_across_delta():
    """PC results must be the same for all delta values."""
    deltas = np.array([-1.0, 0.0, 1.0, 2.0])
    df = extreme_policy_comparison(I, ALPHAS, BETAS, GAMMAS, WAGES, deltas)
    # agg_income_PC should be constant
    assert df['agg_income_PC'].std() < 1e-12, "PC income must be independent of delta"
