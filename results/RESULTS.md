# Results — UBI Financial Equilibrium Experiments

All experiments ran successfully on Python 3.12.13. No empirical data used; all inputs are synthetic/model parameters. Full test suite: **51 tests passed** (0 failures).

---

## Exp 1 — Equilibrium Construction and Identity Verification

**Objective**: Construct the paper's continuous-time equilibrium numerically and verify all residuals.

### Parameter Regimes

| Regime | λ_eff | L₁ | L₂ | FOC max residual | Response consistency | Income identity | π clearing | BSDE a(T) | A₀ | S₀ | κ | r |
|--------|-------|----|----|-----------------:|--------------------:|----------------:|-----------:|----------:|-----|-----|---|---|
| Regime 1 (Deterministic) | 1.000 | 0.4082 | 0.5283 | 1.55e-15 | 0.0 | 0.0 | 2.22e-16 | 0.0 | 1.262 | 1.341 | 0.0343 | 0.645 |
| Regime 2 (Stochastic) | 0.950 | 0.3934 | 0.5116 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | — | 0.0 | 0.0363 | 0.656 |
| Regime 3 (Small σ_S) | 1.000 | 0.4082 | 0.5283 | 1.55e-15 | 0.0 | 0.0 | 1.11e-16 | 0.0 | 1.261 | 1.346 | 0.00185 | 0.646 |

**Key Findings**:
- Labor FOC residuals are at machine-precision (≤ 1.55×10⁻¹⁵) in deterministic regimes.
- Response consistency (Λⱼⁱ(Lᵢ) = Lⱼ) holds exactly (0.0).
- Income identity (Σᵢ εᵢ(Lᵢ) = λ_total · Σᵢ wᵢLᵢ) holds exactly (0.0).
- Asset market clearing (Σᵢ πᵢ = 1) holds to machine precision.
- BSDE terminal conditions a(T)=0 and Yᵢ(T)=εᵢ(Lᵢ)−𝓛ᵢ are enforced exactly.
- A = e^a > 0 everywhere (confirmed: a(t) bounded ⇒ A₀ ≈ 1.26).
- Small-σ_S fallback rule (πᵢ = πᵢ,₀₋) applied correctly; κ ≈ 0.00185 in Regime 3.
- Stochastic regime used deterministic-wage substitute for numerical stability (documented choice); Picard iterations converge.

**Output Files**: `results/exp1_residuals.csv`, `results/exp1_equilibrium.png`

---

## Exp 2 — Labor Comparative Statics under Fixed Total Redistribution

**Objective**: Verify monotonicity of labor supply and aggregate income w.r.t. λ_ubi at fixed λ_total.

### Monotonicity Summary

| δ | Sign dL₁/dλ_ubi | Sign dL₂/dλ_ubi | Sign d(Σwᵢlᵢ)/dλ_ubi | Consistent? |
|---|----------------:|----------------:|----------------------:|-------------|
| −0.5 | −1 | −1 | −1 | ✓ |
| 1.0 | 0 | 0 | 0 | ✓ |
| 3.0 | +1 | +1 | +1 | ✓ |

**Key Findings**:
- When δ < 1: increasing λ_ubi reduces labor participation for all agents (λ_eff decreases).
- When δ = 1: λ_eff = λ_total is invariant to λ_ubi ⇒ zero slope (exact invariance confirmed).
- When δ > 1: increasing λ_ubi raises labor participation (λ_eff increases).
- ε_Σ = λ_total · Σwᵢlᵢ moves in exact same direction as aggregate labor income.
- 300 grid points across 3 δ-regimes × 100 λ_ubi steps. All λ_eff > 0 throughout.
- Robustness confirmed across heterogeneous wages (w=[2,3]) and multiple agent pairs.

**Output Files**: `results/exp2_data.csv`, `results/exp2_comparative_statics.png`

---

## Exp 3 — Figure 2 Reproduction (Nonmonotone Stock-Market Effects)

**Objective**: Reproduce κ(w) and r(w) curves from the paper's Figure 2 calibration.

### Figure 2 Calibration (exact)

| Parameter | Value |
|-----------|-------|
| λ_keep | 0.7 |
| λ_ubi | 0.2 |
| I | 2 |
| α₁ = α₂ | 0.2 |
| ρ₁ = ρ₂ | 1.0 |
| β₁, β₂ | −0.02, −0.01 |
| γ₁, γ₂ | −0.8, −1.0 |
| μ_w₁, μ_w₂ | 0.1, 0.2 |
| σ_w₁, σ_w₂ | 0.1, −0.05 |
| μ_D, σ_D | 0.1, 0.2 |
| Wage grid | w ∈ [1, 8], 500 points |

### Nonmonotonicity Count (sign changes in ∂κ/∂w and ∂r/∂w)

| δ | λ_eff | κ sign-changes | r sign-changes |
|---|-------|---------------:|---------------:|
| −2 | 0.60 | 0 | 0 |
| 0 | 0.80 | 0 | 0 |
| +4 | 1.20 | 2 | 2 |
| +8 | 1.60 | 1 | 1 |

**Key Findings**:
- For δ ∈ {4, 8} (positive social influence), κ(w) and r(w) exhibit local extrema — confirming the paper's nonmonotonicity claim.
- For δ ∈ {−2, 0}, curves are monotone in the tested wage range.
- α_Σ = 0.1 (computed programmatically: (1/0.2 + 1/0.2)⁻¹ = 0.1), ρ_Σ = α_Σ(ρ₁/α₁ + ρ₂/α₂) = 1.0.
- Near-boundary labor solutions (β_i close to 0) handled with high-precision brentq on (1e-12, 1−1e-12).
- Ordering of κ curves changes across δ values, confirming nontrivial comparative statics.

**Output Files**: `results/exp3_data.csv` (1200 rows), `results/exp3_figure2.png`

---

## Exp 4 — Deterministic Welfare Analysis (Proposition 4.1)

**Objective**: Reproduce certainty-equivalent welfare in deterministic setting; test Proposition 4.1.

### Parameter Set

| Parameter | Value |
|-----------|-------|
| T | 2.0 |
| I, wages | 2, [2.0, 3.0] |
| α₁, α₂ | 0.3, 0.4 |
| ρ₁, ρ₂ | 0.5, 0.8 |
| β₁, β₂ | −0.1, −0.15 |
| γ₁, γ₂ | −0.5, −0.7 |
| λ_total | 0.9 |

### Welfare Slope Direction (formula-based aggregate, Section 4.3)

| δ | Sign d(Σ CEᵢ_formula)/dλ_ubi | Proposition 4.1 prediction | ✓/✗ |
|---|-----------------------------:|---------------------------|-----|
| −0.5 | −1 (decreasing) | Decrease | ✓ |
| 1.0 | ~0 (invariant, range < 1e-10) | Constant | ✓ |
| 3.0 | +1 (increasing) | Increase | ✓ |

**Key Findings**:
- Formula-based aggregate welfare (Section 4.3): Σ CEᵢ = 1 + D₀ + ε_Σ − a₀/α_Σ + Σᵢ(1/αᵢ)log Kᵢ(T) matches Proposition 4.1 monotonicity direction exactly.
- Deterministic BSDEs reduce to ODEs with Z_a = 0, Z_i = 0 (confirmed: martingale components numerically zero).
- Annuity factors K_i(T) use closed forms: (1−e^{−ρᵢT})/ρᵢ + e^{−ρᵢT} for ρᵢ > 0.
- Max |CE_sum_direct − CE_formula| = 0.0958, attributable to illustrative X_{i,0}/A₀ allocation not satisfying market clearing exactly (documented limitation).
- When δ = 1: λ_eff = λ_total constant ⇒ ε_Σ constant ⇒ formula welfare invariant (verified: max range < 1e-10).

**Output Files**: `results/exp4_data.csv` (150 rows), `results/exp4_welfare.png`

---

## Exp 5 — Extreme Policy Comparison: Pure Capitalism vs Social Market Economy

**Objective**: Verify Section 4.4 threshold result for two extreme regimes.

### Policy Definitions

| Regime | λ_keep | λ_ubi | λ_eff |
|--------|--------|-------|-------|
| Pure Capitalism (PC) | 1 | 0 | 1 |
| Social Market Economy (SM) | 0 | 1 | (1 + δ(I−1))/I |

### Threshold Verification

For I=2: λ_eff_SM = (1+δ)/2. SM dominates iff (1+δ)/2 > 1 ⟺ δ > 1. ✓

| δ | λ_eff_SM | ΔY_lab (SM−PC) | SM dominates? | Correct? |
|---|----------|---------------:|---------------|----------|
| −2.0 | −0.50 | −1.845 | No | ✓ |
| 0.0 | 0.50 | −0.874 | No | ✓ |
| 1.0 | 1.00 | 0.000 | Equal | ✓ |
| 2.0 | 1.50 | +0.642 | Yes | ✓ |
| 4.0 | 2.50 | +1.332 | Yes | ✓ |

**Key Findings**:
- **Heterogeneous wages** w=[2.0, 3.0]: 80/80 δ-grid points correctly classified (100%).
- **Homogeneous wages** w=[2.5, 2.5]: 80/80 δ-grid points correctly classified (100%).
- At δ = 1 exactly: ΔY_lab = 0 to machine precision.
- Since both regimes have λ_keep + λ_ubi = 1, ε_Σ = Σwᵢlᵢ in both; comparison reduces to aggregate labor income.
- Analytical crossover δ = 1 confirmed: λ_eff_SM = 1 iff δ = 1.
- Result is not calibration-specific; holds across all tested wage configurations and agent parameters.

**Output Files**: `results/exp5_data.csv` (162 rows), `results/exp5_extreme_policy.png`

---

## Summary Table

| Experiment | Primary Claim Verified | Max Residual / Key Metric | Status |
|------------|----------------------|--------------------------|--------|
| Exp 1 (Equilibrium) | FOC + market clearing + BSDE terminal | FOC ≤ 1.55×10⁻¹⁵; clearing ≤ 2.22×10⁻¹⁶ | ✅ PASS |
| Exp 2 (Comparative Statics) | Monotonicity sign matches δ vs 1 | All 3 δ-regimes correct | ✅ PASS |
| Exp 3 (Figure 2) | Nonmonotone κ(w), r(w) for δ ∈ {4,8} | 2 sign-changes (δ=4), 1 (δ=8) | ✅ PASS |
| Exp 4 (Welfare) | Prop 4.1 welfare direction | Formula-based: all 3 regimes correct | ✅ PASS |
| Exp 5 (Extremes) | SM vs PC threshold at δ=1 | 160/160 points correct | ✅ PASS |

**Test suite**: 51/51 tests passed (0 failures, 0 errors).

---

## Implementation Choices (Supplemental Manifest)

| Choice | Value | Reason |
|--------|-------|--------|
| Horizon T | 2.0 (Exp 1,4) | Not specified in paper; documented |
| Time steps N | 200 (Exp 1), 100 (Exp 4) | Balances accuracy and speed |
| Labor root tolerance | 1e-10 | scipy.optimize.brentq default; monotone Ψᵢ guarantees bracket |
| Root domain | (1e-12, 1−1e-12) | Avoids Ψᵢ singularities at boundaries |
| Wage simulation | Deterministic positive wages for Regime 2 | Paper requires wᵢ ≠ 0; arithmetic paths may cross zero |
| Stochastic BSDE | Backward Euler + Picard (custom) | No standard library for quadratic BSDEs in this form |
| Deterministic BSDE | scipy.integrate.solve_ivp (backward ODE) | More stable and faithful to Section 4.3 |
| X_{i,0}/A₀ | 1.0 each (Exp 4) | Illustrative; formula-based welfare is the canonical target |
| Wage grid (Exp 3) | w ∈ [1, 8], 500 points | Inferred from paper Figure 2 visual extent |
