"""
Statistical Utilities for FactMarrow Confidence Scoring

Helper functions for statistical computations used throughout
the confidence scoring framework.

@PRISM - Data Science & Statistical Analysis
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

import numpy as np
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════
# EFFECT SIZE CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════════


def odds_ratio_to_cohens_d(odds_ratio: float) -> float:
    """
    Convert odds ratio to Cohen's d for standardized comparison.
    
    Uses the approximation: d ≈ log(OR) × √3 / π
    
    Args:
        odds_ratio: Odds ratio from study
        
    Returns:
        Cohen's d effect size
    """
    if odds_ratio <= 0:
        raise ValueError("Odds ratio must be positive")
    return math.log(odds_ratio) * math.sqrt(3) / math.pi


def relative_risk_to_cohens_d(relative_risk: float, p_control: float) -> float:
    """
    Convert relative risk to Cohen's d.
    
    Args:
        relative_risk: Relative risk from study
        p_control: Baseline probability in control group
        
    Returns:
        Cohen's d effect size
    """
    if relative_risk <= 0 or p_control <= 0 or p_control >= 1:
        raise ValueError("Invalid relative risk or control probability")
    
    # Convert RR to OR first
    p_treatment = p_control * relative_risk
    p_treatment = min(0.999, max(0.001, p_treatment))  # Bound
    
    odds_control = p_control / (1 - p_control)
    odds_treatment = p_treatment / (1 - p_treatment)
    odds_ratio = odds_treatment / odds_control
    
    return odds_ratio_to_cohens_d(odds_ratio)


def cohens_d_to_correlation(d: float) -> float:
    """
    Convert Cohen's d to correlation coefficient r.
    
    Uses: r = d / √(d² + 4)
    
    Args:
        d: Cohen's d effect size
        
    Returns:
        Correlation coefficient r
    """
    return d / math.sqrt(d**2 + 4)


def correlation_to_cohens_d(r: float) -> float:
    """
    Convert correlation coefficient to Cohen's d.
    
    Uses: d = 2r / √(1 - r²)
    
    Args:
        r: Correlation coefficient
        
    Returns:
        Cohen's d effect size
    """
    if abs(r) >= 1:
        raise ValueError("Correlation must be between -1 and 1")
    return 2 * r / math.sqrt(1 - r**2)


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size magnitude.
    
    Standard thresholds:
    - Small: d = 0.2
    - Medium: d = 0.5
    - Large: d = 0.8
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLE SIZE AND POWER CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════


def required_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> int:
    """
    Calculate required sample size per group for detecting effect.
    
    Args:
        effect_size: Expected Cohen's d
        alpha: Significance level
        power: Desired statistical power
        two_tailed: Whether test is two-tailed
        
    Returns:
        Required sample size per group
    """
    if effect_size == 0:
        return float('inf')
    
    # Z-scores for alpha and power
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_power = stats.norm.ppf(power)
    
    # Sample size formula: n = 2 * ((z_α + z_β) / d)²
    n = 2 * ((z_alpha + z_power) / effect_size)**2
    
    return int(math.ceil(n))


def achieved_power(
    effect_size: float,
    sample_size: int,
    alpha: float = 0.05,
    two_tailed: bool = True
) -> float:
    """
    Calculate achieved power for given sample size.
    
    Args:
        effect_size: Cohen's d
        sample_size: Sample size per group
        alpha: Significance level
        two_tailed: Whether test is two-tailed
        
    Returns:
        Achieved statistical power
    """
    if effect_size == 0 or sample_size <= 0:
        return alpha  # Power equals alpha when no effect
    
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    # Non-centrality parameter
    ncp = effect_size * math.sqrt(sample_size / 2)
    
    # Power
    power = 1 - stats.norm.cdf(z_alpha - ncp)
    
    return power


# ═══════════════════════════════════════════════════════════════════════════
# RECENCY WEIGHTING
# ═══════════════════════════════════════════════════════════════════════════


def compute_recency_weight(
    publication_date: datetime,
    reference_date: Optional[datetime] = None,
    half_life_years: float = 5.0,
    min_weight: float = 0.1
) -> float:
    """
    Compute recency weight using exponential decay.
    
    More recent studies get higher weight. Uses exponential decay
    with configurable half-life.
    
    Args:
        publication_date: Date of publication
        reference_date: Reference date (default: now)
        half_life_years: Years until weight halves
        min_weight: Minimum weight floor
        
    Returns:
        Recency weight between min_weight and 1.0
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    # Time difference in years
    delta = reference_date - publication_date
    years = delta.days / 365.25
    
    # Exponential decay
    decay_rate = math.log(2) / half_life_years
    weight = math.exp(-decay_rate * years)
    
    return max(min_weight, weight)


def compute_citation_recency_factor(
    citation_count: int,
    years_since_publication: float,
    field_average_citations_per_year: float = 5.0
) -> float:
    """
    Adjust recency weight based on citation velocity.
    
    Highly-cited older papers should retain more weight than
    poorly-cited recent papers.
    
    Args:
        citation_count: Total citations
        years_since_publication: Years since publication
        field_average_citations_per_year: Expected citations/year
        
    Returns:
        Citation-adjusted recency factor (0.5 to 1.5)
    """
    if years_since_publication <= 0:
        years_since_publication = 0.5  # Minimum
    
    citations_per_year = citation_count / years_since_publication
    ratio = citations_per_year / field_average_citations_per_year
    
    # Sigmoid transformation to bound factor
    factor = 0.5 + 1.0 / (1 + math.exp(-math.log(ratio + 0.1)))
    
    return min(1.5, max(0.5, factor))


# ═══════════════════════════════════════════════════════════════════════════
# HETEROGENEITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════


def interpret_i_squared(i_squared: float) -> Tuple[str, str]:
    """
    Interpret I² heterogeneity statistic.
    
    Standard thresholds (Cochrane Handbook):
    - 0-40%: might not be important
    - 30-60%: moderate heterogeneity
    - 50-90%: substantial heterogeneity
    - 75-100%: considerable heterogeneity
    
    Returns:
        Tuple of (magnitude, recommendation)
    """
    if i_squared < 25:
        return ("low", "Fixed-effects model appropriate")
    elif i_squared < 50:
        return ("moderate", "Consider random-effects model")
    elif i_squared < 75:
        return ("substantial", "Random-effects model recommended; explore sources")
    else:
        return ("considerable", "Investigate heterogeneity sources before pooling")


def predict_credible_interval_with_heterogeneity(
    pooled_effect: float,
    tau_squared: float,
    standard_error: float
) -> Tuple[float, float]:
    """
    Compute prediction interval accounting for heterogeneity.
    
    The prediction interval shows the range of effects expected
    in future studies, accounting for between-study variance.
    
    Args:
        pooled_effect: Meta-analytic pooled effect
        tau_squared: Between-study variance
        standard_error: SE of pooled effect
        
    Returns:
        95% prediction interval (lower, upper)
    """
    # Prediction interval variance includes tau²
    total_variance = standard_error**2 + tau_squared
    prediction_se = math.sqrt(total_variance)
    
    lower = pooled_effect - 1.96 * prediction_se
    upper = pooled_effect + 1.96 * prediction_se
    
    return (lower, upper)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLICATION BIAS DETECTION
# ═══════════════════════════════════════════════════════════════════════════


def egger_regression(
    effects: np.ndarray,
    standard_errors: np.ndarray
) -> Dict[str, float]:
    """
    Egger's regression test for funnel plot asymmetry.
    
    Tests for small-study effects (potential publication bias).
    
    Regression: effect/SE ~ 1/SE
    Intercept ≠ 0 suggests asymmetry
    
    Args:
        effects: Array of effect sizes
        standard_errors: Array of standard errors
        
    Returns:
        Dictionary with intercept, p-value, interpretation
    """
    if len(effects) < 3:
        return {
            "intercept": None,
            "p_value": None,
            "interpretation": "Insufficient studies for test"
        }
    
    # Precision (1/SE)
    precision = 1.0 / standard_errors
    
    # Standardized effect (effect/SE)
    standardized = effects / standard_errors
    
    # Weighted regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        precision, standardized
    )
    
    # The intercept indicates asymmetry
    if p_value < 0.10:
        interpretation = "Significant asymmetry detected (potential publication bias)"
    else:
        interpretation = "No significant asymmetry"
    
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "p_value": float(p_value),
        "interpretation": interpretation
    }


def trim_and_fill(
    effects: np.ndarray,
    variances: np.ndarray,
    side: str = "right"
) -> Dict[str, Any]:
    """
    Trim-and-fill method for publication bias adjustment.
    
    Estimates number of missing studies and adjusts pooled effect.
    
    Args:
        effects: Array of effect sizes
        variances: Array of variances
        side: Which side to fill ("left", "right", or "auto")
        
    Returns:
        Adjusted pooled effect and number of imputed studies
    """
    n = len(effects)
    if n < 3:
        return {
            "adjusted_effect": None,
            "n_missing": 0,
            "message": "Insufficient studies"
        }
    
    # Simple weights
    weights = 1.0 / variances
    
    # Initial pooled estimate
    pooled = np.sum(weights * effects) / np.sum(weights)
    
    # Rank-based estimation of missing studies
    # (Simplified implementation)
    deviations = effects - pooled
    
    if side == "auto":
        # Determine which side has fewer extreme values
        left_extreme = np.sum(deviations < -np.std(deviations))
        right_extreme = np.sum(deviations > np.std(deviations))
        side = "left" if left_extreme < right_extreme else "right"
    
    # Count asymmetric studies
    if side == "right":
        n_extreme = np.sum(deviations > np.std(deviations))
    else:
        n_extreme = np.sum(deviations < -np.std(deviations))
    
    # Estimate missing (simplified)
    n_missing = max(0, n_extreme - n // 4)
    
    # If missing, impute mirror studies
    if n_missing > 0:
        if side == "right":
            extreme_effects = effects[deviations > np.std(deviations)][:n_missing]
            imputed_effects = 2 * pooled - extreme_effects
        else:
            extreme_effects = effects[deviations < -np.std(deviations)][:n_missing]
            imputed_effects = 2 * pooled - extreme_effects
        
        # Recalculate with imputed
        all_effects = np.concatenate([effects, imputed_effects])
        imputed_variances = variances[np.argsort(np.abs(deviations))[-n_missing:]]
        all_variances = np.concatenate([variances, imputed_variances])
        all_weights = 1.0 / all_variances
        
        adjusted_effect = np.sum(all_weights * all_effects) / np.sum(all_weights)
    else:
        adjusted_effect = pooled
    
    return {
        "original_effect": float(pooled),
        "adjusted_effect": float(adjusted_effect),
        "n_missing": int(n_missing),
        "side": side
    }


# ═══════════════════════════════════════════════════════════════════════════
# BAYESIAN UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


def beta_hdi(
    alpha: float,
    beta: float,
    credible_mass: float = 0.95
) -> Tuple[float, float]:
    """
    Compute Highest Density Interval for Beta distribution.
    
    The HDI contains the most probable values, unlike
    equal-tailed intervals.
    
    Args:
        alpha: Beta distribution alpha parameter
        beta: Beta distribution beta parameter
        credible_mass: Probability mass to contain (default 0.95)
        
    Returns:
        HDI (lower, upper)
    """
    from scipy.optimize import minimize_scalar
    
    # Find the narrowest interval containing credible_mass
    def interval_width(lower_tail):
        lower = stats.beta.ppf(lower_tail, alpha, beta)
        upper = stats.beta.ppf(lower_tail + credible_mass, alpha, beta)
        return upper - lower
    
    # Optimize to find minimum width
    result = minimize_scalar(
        interval_width,
        bounds=(0, 1 - credible_mass),
        method='bounded'
    )
    
    optimal_lower_tail = result.x
    lower = stats.beta.ppf(optimal_lower_tail, alpha, beta)
    upper = stats.beta.ppf(optimal_lower_tail + credible_mass, alpha, beta)
    
    return (float(lower), float(upper))


def bayesian_model_comparison(
    posterior_a: Tuple[float, float],
    posterior_b: Tuple[float, float],
    n_samples: int = 10000
) -> Dict[str, float]:
    """
    Compare two competing claims using Bayesian approach.
    
    Useful when evidence supports two different interpretations.
    
    Args:
        posterior_a: Beta params (α, β) for claim A
        posterior_b: Beta params (α, β) for claim B
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Probability that A > B and Bayes factor
    """
    # Sample from posteriors
    samples_a = np.random.beta(posterior_a[0], posterior_a[1], n_samples)
    samples_b = np.random.beta(posterior_b[0], posterior_b[1], n_samples)
    
    # P(A > B)
    prob_a_greater = np.mean(samples_a > samples_b)
    
    # Bayes factor (approximate)
    # BF = P(A > B) / P(B > A)
    if prob_a_greater > 0 and prob_a_greater < 1:
        bayes_factor = prob_a_greater / (1 - prob_a_greater)
    else:
        bayes_factor = float('inf') if prob_a_greater > 0.5 else 0.0
    
    return {
        "prob_a_greater": float(prob_a_greater),
        "prob_b_greater": float(1 - prob_a_greater),
        "bayes_factor": float(min(bayes_factor, 1000)),  # Cap for display
        "interpretation": interpret_bayes_factor(bayes_factor)
    }


def interpret_bayes_factor(bf: float) -> str:
    """
    Interpret Bayes factor strength (Kass & Raftery scale).
    
    BF > 1: Evidence for A
    BF < 1: Evidence for B
    """
    if bf > 100:
        return "Decisive evidence for A"
    elif bf > 30:
        return "Very strong evidence for A"
    elif bf > 10:
        return "Strong evidence for A"
    elif bf > 3:
        return "Moderate evidence for A"
    elif bf > 1:
        return "Weak evidence for A"
    elif bf > 1/3:
        return "Weak evidence for B"
    elif bf > 1/10:
        return "Moderate evidence for B"
    elif bf > 1/30:
        return "Strong evidence for B"
    elif bf > 1/100:
        return "Very strong evidence for B"
    else:
        return "Decisive evidence for B"


# ═══════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORE ADJUSTMENTS
# ═══════════════════════════════════════════════════════════════════════════


def shrinkage_adjustment(
    raw_score: float,
    n_evidence: int,
    prior_mean: float = 0.5,
    shrinkage_k: float = 3.0
) -> float:
    """
    Apply empirical Bayes shrinkage to confidence score.
    
    With limited evidence, shrink toward prior (conservative).
    With abundant evidence, trust observed more.
    
    Formula: adjusted = (n * raw + k * prior) / (n + k)
    
    Args:
        raw_score: Raw confidence score (0-1)
        n_evidence: Number of evidence items
        prior_mean: Prior mean to shrink toward
        shrinkage_k: Shrinkage strength parameter
        
    Returns:
        Shrinkage-adjusted score
    """
    return (n_evidence * raw_score + shrinkage_k * prior_mean) / (n_evidence + shrinkage_k)


def conservative_adjustment(
    confidence: float,
    uncertainty: float,
    conservatism: float = 0.2
) -> float:
    """
    Apply conservative adjustment based on uncertainty.
    
    Higher uncertainty leads to more conservative (lower) confidence.
    
    Args:
        confidence: Raw confidence (0-1)
        uncertainty: Uncertainty measure (0-1)
        conservatism: How much to adjust (0-1)
        
    Returns:
        Conservatively adjusted confidence
    """
    adjustment = conservatism * uncertainty * confidence
    return confidence - adjustment


def ensemble_confidence(
    scores: List[float],
    weights: Optional[List[float]] = None,
    method: str = "weighted_mean"
) -> float:
    """
    Combine multiple confidence scores.
    
    Args:
        scores: List of confidence scores
        weights: Optional weights (default: equal)
        method: "weighted_mean", "median", "geometric_mean", "min"
        
    Returns:
        Combined confidence score
    """
    if not scores:
        return 0.5
    
    scores_arr = np.array(scores)
    
    if weights is None:
        weights_arr = np.ones(len(scores)) / len(scores)
    else:
        weights_arr = np.array(weights)
        weights_arr = weights_arr / weights_arr.sum()
    
    if method == "weighted_mean":
        return float(np.sum(weights_arr * scores_arr))
    
    elif method == "median":
        return float(np.median(scores_arr))
    
    elif method == "geometric_mean":
        # Avoid log(0)
        scores_clipped = np.clip(scores_arr, 1e-10, 1)
        log_scores = np.log(scores_clipped)
        return float(np.exp(np.sum(weights_arr * log_scores)))
    
    elif method == "min":
        # Most conservative
        return float(np.min(scores_arr))
    
    else:
        return float(np.mean(scores_arr))
