"""
FactMarrow Confidence Scoring Module

Statistical framework for claim verification confidence scoring using:
1. Bayesian inference for belief updating
2. Meta-analysis for multi-source evidence combination
3. Uncertainty quantification with calibration
4. Causal inference for health claim assessment

@PRISM - Data Science & Statistical Analysis
"""

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from scipy.special import expit, logit  # sigmoid and inverse

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════


class ClaimType(str, Enum):
    """Types of claims in public health documents"""
    QUANTITATIVE = "quantitative"      # Statistics, percentages, measurements
    CAUSAL = "causal"                  # Cause-effect relationships
    CORRELATIONAL = "correlational"    # Associations without causation
    PREVALENCE = "prevalence"          # Disease/condition rates
    EFFICACY = "efficacy"              # Treatment effectiveness
    SAFETY = "safety"                  # Safety/risk claims
    DEFINITIONAL = "definitional"      # What something is
    METHODOLOGICAL = "methodological"  # Study design claims
    PRESCRIPTIVE = "prescriptive"      # Recommendations


class SourceType(str, Enum):
    """Source quality tiers for evidence weighting"""
    SYSTEMATIC_REVIEW = "systematic_review"     # Cochrane, meta-analyses
    RCT = "rct"                                 # Randomized controlled trials
    COHORT_STUDY = "cohort_study"              # Prospective/retrospective cohorts
    CASE_CONTROL = "case_control"              # Case-control studies
    CASE_SERIES = "case_series"                # Case reports/series
    EXPERT_OPINION = "expert_opinion"          # Expert consensus
    GOVERNMENT_AGENCY = "government_agency"    # CDC, WHO, FDA
    PEER_REVIEWED = "peer_reviewed"            # General peer-reviewed
    PREPRINT = "preprint"                      # Not yet peer-reviewed
    NEWS_OUTLET = "news_outlet"                # Media sources
    SOCIAL_MEDIA = "social_media"              # Social platforms
    UNKNOWN = "unknown"                        # Unclassified sources


class EvidenceDirection(str, Enum):
    """Direction of evidence relative to claim"""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    INCONCLUSIVE = "inconclusive"


@dataclass
class EvidenceItem:
    """Single piece of evidence for/against a claim"""
    source_id: str
    source_type: SourceType
    direction: EvidenceDirection
    effect_size: Optional[float] = None           # Standardized effect (Cohen's d, OR, RR)
    effect_size_variance: Optional[float] = None  # Variance of effect estimate
    sample_size: Optional[int] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    study_quality_score: float = 0.5              # 0-1 quality rating
    recency_weight: float = 1.0                   # Decay for older studies
    raw_text: Optional[str] = None


@dataclass
class ConfidenceResult:
    """Complete confidence scoring result"""
    claim_text: str
    claim_type: ClaimType
    
    # Bayesian posterior
    posterior_probability: float           # P(claim true | evidence)
    credible_interval: Tuple[float, float] # 95% credible interval
    
    # Meta-analysis results
    pooled_effect: Optional[float] = None
    heterogeneity_i2: Optional[float] = None  # I² statistic
    heterogeneity_p: Optional[float] = None   # Cochran's Q p-value
    
    # Uncertainty decomposition
    epistemic_uncertainty: float = 0.0     # Reducible (lack of knowledge)
    aleatoric_uncertainty: float = 0.0     # Irreducible (inherent variability)
    total_uncertainty: float = 0.0
    
    # Causal assessment (for causal claims)
    causal_strength: Optional[float] = None
    confounders_identified: List[str] = field(default_factory=list)
    bradford_hill_score: Optional[float] = None
    
    # Quality metrics
    evidence_count: int = 0
    source_diversity_score: float = 0.0
    calibration_adjustment: float = 0.0
    
    # Final score
    final_confidence: float = 0.0          # 0-100 calibrated score
    confidence_grade: str = "D"            # A, B, C, D, F
    explanation: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# 1. BAYESIAN CONFIDENCE FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════


class BayesianConfidenceEstimator:
    """
    Bayesian framework for updating belief in claim veracity.
    
    Uses Beta-Binomial conjugate prior for tractable updates:
    - Prior: Beta(α, β) representing initial belief
    - Likelihood: Binomial for evidence observations
    - Posterior: Beta(α + supports, β + contradicts)
    """
    
    # Claim-type specific prior parameters (α, β for Beta distribution)
    # Based on empirical base rates of different claim types being accurate
    CLAIM_TYPE_PRIORS: Dict[ClaimType, Tuple[float, float]] = {
        ClaimType.QUANTITATIVE: (3.0, 2.0),       # Often accurate when sourced
        ClaimType.CAUSAL: (2.0, 3.0),             # Frequently overstated
        ClaimType.CORRELATIONAL: (2.5, 2.0),     # Usually has some basis
        ClaimType.PREVALENCE: (3.5, 1.5),        # Generally reliable from agencies
        ClaimType.EFFICACY: (2.0, 2.5),          # Often overstated
        ClaimType.SAFETY: (2.0, 2.0),            # Neutral prior, context-dependent
        ClaimType.DEFINITIONAL: (4.0, 1.0),      # Usually accurate
        ClaimType.METHODOLOGICAL: (3.0, 2.0),    # Verifiable
        ClaimType.PRESCRIPTIVE: (2.0, 2.0),      # Context-dependent
    }
    
    # Source reliability weights (how much to trust each source type)
    # Higher weight = more informative evidence
    SOURCE_WEIGHTS: Dict[SourceType, float] = {
        SourceType.SYSTEMATIC_REVIEW: 3.0,
        SourceType.RCT: 2.5,
        SourceType.COHORT_STUDY: 2.0,
        SourceType.CASE_CONTROL: 1.5,
        SourceType.CASE_SERIES: 1.0,
        SourceType.EXPERT_OPINION: 1.2,
        SourceType.GOVERNMENT_AGENCY: 2.5,
        SourceType.PEER_REVIEWED: 1.8,
        SourceType.PREPRINT: 0.8,
        SourceType.NEWS_OUTLET: 0.4,
        SourceType.SOCIAL_MEDIA: 0.1,
        SourceType.UNKNOWN: 0.3,
    }
    
    def __init__(self, calibration_model: Optional['CalibrationModel'] = None):
        self.calibration_model = calibration_model
    
    def get_prior(self, claim_type: ClaimType) -> Tuple[float, float]:
        """Get prior Beta parameters for claim type."""
        return self.CLAIM_TYPE_PRIORS.get(claim_type, (2.0, 2.0))
    
    def compute_posterior(
        self,
        claim_type: ClaimType,
        evidence: List[EvidenceItem],
        prior_override: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float, float]:
        """
        Compute posterior probability using Bayesian update.
        
        Args:
            claim_type: Type of claim for prior selection
            evidence: List of evidence items
            prior_override: Optional custom prior (α, β)
            
        Returns:
            Tuple of (posterior_mean, credible_lower, credible_upper)
        """
        # Initialize prior
        if prior_override:
            alpha, beta = prior_override
        else:
            alpha, beta = self.get_prior(claim_type)
        
        # Weighted evidence aggregation
        weighted_supports = 0.0
        weighted_contradicts = 0.0
        
        for item in evidence:
            weight = self.SOURCE_WEIGHTS.get(item.source_type, 0.5)
            weight *= item.study_quality_score
            weight *= item.recency_weight
            
            if item.direction == EvidenceDirection.SUPPORTS:
                weighted_supports += weight
            elif item.direction == EvidenceDirection.CONTRADICTS:
                weighted_contradicts += weight
            elif item.direction == EvidenceDirection.INCONCLUSIVE:
                # Inconclusive evidence slightly increases uncertainty
                weighted_supports += weight * 0.3
                weighted_contradicts += weight * 0.3
        
        # Update posterior
        alpha_post = alpha + weighted_supports
        beta_post = beta + weighted_contradicts
        
        # Posterior statistics
        posterior_mean = alpha_post / (alpha_post + beta_post)
        
        # 95% credible interval using Beta distribution quantiles
        credible_lower = stats.beta.ppf(0.025, alpha_post, beta_post)
        credible_upper = stats.beta.ppf(0.975, alpha_post, beta_post)
        
        return posterior_mean, credible_lower, credible_upper
    
    def handle_contradictory_evidence(
        self,
        supporting: List[EvidenceItem],
        contradicting: List[EvidenceItem]
    ) -> Dict[str, Any]:
        """
        Analyze contradictory evidence patterns.
        
        Uses hierarchical approach:
        1. Weight by source quality
        2. Check for systematic differences
        3. Flag potential publication bias
        """
        if not contradicting:
            return {
                "contradiction_detected": False,
                "resolution_confidence": 1.0,
                "likely_explanation": None
            }
        
        # Calculate quality-weighted evidence strength
        support_strength = sum(
            self.SOURCE_WEIGHTS[e.source_type] * e.study_quality_score
            for e in supporting
        )
        contradict_strength = sum(
            self.SOURCE_WEIGHTS[e.source_type] * e.study_quality_score
            for e in contradicting
        )
        
        total_strength = support_strength + contradict_strength
        if total_strength == 0:
            resolution_confidence = 0.5
        else:
            resolution_confidence = abs(support_strength - contradict_strength) / total_strength
        
        # Analyze contradiction patterns
        support_types = set(e.source_type for e in supporting)
        contradict_types = set(e.source_type for e in contradicting)
        
        explanation = None
        if SourceType.NEWS_OUTLET in contradict_types and \
           SourceType.PEER_REVIEWED in support_types:
            explanation = "Contradiction likely from lower-quality sources"
        elif len(contradicting) > len(supporting) * 2:
            explanation = "Preponderance of contradicting evidence"
        elif support_strength > contradict_strength * 2:
            explanation = "Higher-quality evidence supports claim"
        else:
            explanation = "Genuine scientific disagreement detected"
        
        return {
            "contradiction_detected": True,
            "support_strength": support_strength,
            "contradict_strength": contradict_strength,
            "resolution_confidence": resolution_confidence,
            "likely_explanation": explanation,
            "recommendation": "manual_review" if resolution_confidence < 0.3 else "auto_resolve"
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. META-ANALYSIS INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════


class MetaAnalysisEngine:
    """
    Combines evidence from multiple sources using meta-analytic techniques.
    
    Implements:
    - Fixed-effects model (when homogeneous)
    - Random-effects model (DerSimonian-Laird)
    - Heterogeneity assessment (I², Q statistic)
    - Quality-weighted pooling
    """
    
    # Minimum studies for valid meta-analysis
    MIN_STUDIES = 2
    
    def __init__(self):
        pass
    
    def compute_pooled_effect(
        self,
        evidence: List[EvidenceItem],
        method: str = "random"
    ) -> Optional[Dict[str, Any]]:
        """
        Pool effect sizes across studies.
        
        Args:
            evidence: List of evidence with effect sizes
            method: "fixed" or "random" effects model
            
        Returns:
            Dictionary with pooled effect, CI, heterogeneity stats
        """
        # Filter to evidence with effect sizes
        valid_evidence = [
            e for e in evidence 
            if e.effect_size is not None and e.effect_size_variance is not None
            and e.effect_size_variance > 0
        ]
        
        if len(valid_evidence) < self.MIN_STUDIES:
            return None
        
        effects = np.array([e.effect_size for e in valid_evidence])
        variances = np.array([e.effect_size_variance for e in valid_evidence])
        weights = 1.0 / variances
        
        # Apply quality weights
        quality_weights = np.array([e.study_quality_score for e in valid_evidence])
        combined_weights = weights * quality_weights
        
        # Fixed-effects pooled estimate
        fixed_effect = np.sum(combined_weights * effects) / np.sum(combined_weights)
        fixed_variance = 1.0 / np.sum(combined_weights)
        
        # Heterogeneity assessment
        heterogeneity = self._compute_heterogeneity(effects, variances, weights)
        
        if method == "random" and heterogeneity["tau_squared"] > 0:
            # Random-effects model (DerSimonian-Laird)
            random_weights = 1.0 / (variances + heterogeneity["tau_squared"])
            random_weights *= quality_weights
            pooled_effect = np.sum(random_weights * effects) / np.sum(random_weights)
            pooled_variance = 1.0 / np.sum(random_weights)
        else:
            pooled_effect = fixed_effect
            pooled_variance = fixed_variance
        
        pooled_se = np.sqrt(pooled_variance)
        ci_lower = pooled_effect - 1.96 * pooled_se
        ci_upper = pooled_effect + 1.96 * pooled_se
        
        return {
            "pooled_effect": float(pooled_effect),
            "pooled_se": float(pooled_se),
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "method": method,
            "n_studies": len(valid_evidence),
            "heterogeneity": heterogeneity
        }
    
    def _compute_heterogeneity(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute heterogeneity statistics.
        
        Returns:
            Dictionary with Q, I², tau², p-value
        """
        k = len(effects)
        if k < 2:
            return {"Q": 0, "I_squared": 0, "tau_squared": 0, "p_value": 1.0}
        
        # Weighted mean
        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        
        # Cochran's Q statistic
        Q = np.sum(weights * (effects - weighted_mean) ** 2)
        
        # Degrees of freedom
        df = k - 1
        
        # P-value for Q
        p_value = 1 - stats.chi2.cdf(Q, df)
        
        # I² statistic (percentage of variability due to heterogeneity)
        if Q > df:
            I_squared = 100 * (Q - df) / Q
        else:
            I_squared = 0.0
        
        # Tau² (between-study variance) - DerSimonian-Laird estimator
        C = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
        if C > 0:
            tau_squared = max(0, (Q - df) / C)
        else:
            tau_squared = 0
        
        return {
            "Q": float(Q),
            "df": int(df),
            "I_squared": float(I_squared),
            "tau_squared": float(tau_squared),
            "p_value": float(p_value)
        }
    
    def weight_by_source_quality(
        self,
        evidence: List[EvidenceItem]
    ) -> List[EvidenceItem]:
        """
        Apply hierarchical source quality weighting.
        
        Uses evidence pyramid:
        1. Systematic reviews/meta-analyses (highest)
        2. RCTs
        3. Cohort studies
        4. Case-control
        5. Case series
        6. Expert opinion
        7. News/social media (lowest)
        """
        # Quality multipliers based on evidence hierarchy
        QUALITY_MULTIPLIERS = {
            SourceType.SYSTEMATIC_REVIEW: 1.0,
            SourceType.RCT: 0.85,
            SourceType.COHORT_STUDY: 0.70,
            SourceType.CASE_CONTROL: 0.55,
            SourceType.CASE_SERIES: 0.40,
            SourceType.EXPERT_OPINION: 0.50,
            SourceType.GOVERNMENT_AGENCY: 0.80,
            SourceType.PEER_REVIEWED: 0.65,
            SourceType.PREPRINT: 0.35,
            SourceType.NEWS_OUTLET: 0.15,
            SourceType.SOCIAL_MEDIA: 0.05,
            SourceType.UNKNOWN: 0.20,
        }
        
        weighted_evidence = []
        for item in evidence:
            multiplier = QUALITY_MULTIPLIERS.get(item.source_type, 0.2)
            # Create new item with adjusted quality score
            new_item = EvidenceItem(
                source_id=item.source_id,
                source_type=item.source_type,
                direction=item.direction,
                effect_size=item.effect_size,
                effect_size_variance=item.effect_size_variance,
                sample_size=item.sample_size,
                confidence_interval=item.confidence_interval,
                p_value=item.p_value,
                study_quality_score=item.study_quality_score * multiplier,
                recency_weight=item.recency_weight,
                raw_text=item.raw_text
            )
            weighted_evidence.append(new_item)
        
        return weighted_evidence


# ═══════════════════════════════════════════════════════════════════════════
# 3. UNCERTAINTY QUANTIFICATION
# ═══════════════════════════════════════════════════════════════════════════


class UncertaintyQuantifier:
    """
    Quantifies and decomposes uncertainty in confidence estimates.
    
    Separates:
    - Epistemic uncertainty: Reducible through more evidence
    - Aleatoric uncertainty: Irreducible inherent variability
    """
    
    def __init__(self):
        pass
    
    def decompose_uncertainty(
        self,
        posterior_mean: float,
        credible_interval: Tuple[float, float],
        evidence: List[EvidenceItem],
        meta_analysis_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Decompose total uncertainty into epistemic and aleatoric components.
        
        Epistemic (model uncertainty):
        - Lack of evidence
        - Conflicting sources
        - Poor quality sources
        
        Aleatoric (data uncertainty):
        - Inherent variability in studies
        - Natural variation in populations
        """
        # Total uncertainty from credible interval width
        total_uncertainty = (credible_interval[1] - credible_interval[0]) / 2
        
        # Epistemic: reducible through more/better evidence
        epistemic_factors = []
        
        # 1. Evidence scarcity
        evidence_count = len(evidence)
        scarcity_factor = 1.0 / (1.0 + np.log1p(evidence_count))
        epistemic_factors.append(scarcity_factor)
        
        # 2. Source quality variance
        if evidence:
            quality_scores = [e.study_quality_score for e in evidence]
            quality_variance = np.var(quality_scores) if len(quality_scores) > 1 else 0.5
            epistemic_factors.append(quality_variance)
        else:
            epistemic_factors.append(1.0)
        
        # 3. Conflicting evidence
        supports = sum(1 for e in evidence if e.direction == EvidenceDirection.SUPPORTS)
        contradicts = sum(1 for e in evidence if e.direction == EvidenceDirection.CONTRADICTS)
        if supports + contradicts > 0:
            conflict_ratio = min(supports, contradicts) / (supports + contradicts)
        else:
            conflict_ratio = 0.5
        epistemic_factors.append(conflict_ratio)
        
        epistemic_uncertainty = np.mean(epistemic_factors) * total_uncertainty
        
        # Aleatoric: inherent variability
        if meta_analysis_result and "heterogeneity" in meta_analysis_result:
            # Use I² as measure of inherent variability
            i_squared = meta_analysis_result["heterogeneity"]["I_squared"]
            aleatoric_ratio = i_squared / 100.0
        else:
            # Estimate from effect size variance
            if evidence:
                effect_vars = [e.effect_size_variance for e in evidence 
                              if e.effect_size_variance is not None]
                if effect_vars:
                    aleatoric_ratio = min(1.0, np.mean(effect_vars))
                else:
                    aleatoric_ratio = 0.3  # Default
            else:
                aleatoric_ratio = 0.5
        
        aleatoric_uncertainty = aleatoric_ratio * total_uncertainty
        
        # Normalize to sum to total
        total_decomposed = epistemic_uncertainty + aleatoric_uncertainty
        if total_decomposed > 0:
            epistemic_uncertainty = epistemic_uncertainty * total_uncertainty / total_decomposed
            aleatoric_uncertainty = aleatoric_uncertainty * total_uncertainty / total_decomposed
        
        return {
            "epistemic_uncertainty": float(epistemic_uncertainty),
            "aleatoric_uncertainty": float(aleatoric_uncertainty),
            "total_uncertainty": float(total_uncertainty),
            "reducibility_ratio": float(epistemic_uncertainty / total_uncertainty) if total_uncertainty > 0 else 0.5
        }
    
    def compute_confidence_interval(
        self,
        estimate: float,
        uncertainty: float,
        method: str = "normal"
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for estimate.
        
        Args:
            estimate: Point estimate (0-1)
            uncertainty: Standard error or width
            method: "normal", "wilson", or "clopper_pearson"
        """
        if method == "normal":
            # Simple normal approximation
            lower = max(0, estimate - 1.96 * uncertainty)
            upper = min(1, estimate + 1.96 * uncertainty)
        
        elif method == "wilson":
            # Wilson score interval (better for proportions near 0 or 1)
            z = 1.96
            n = 100  # Effective sample size proxy
            p = estimate
            
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denominator
            spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
            
            lower = max(0, center - spread)
            upper = min(1, center + spread)
        
        else:  # clopper_pearson (exact)
            # Use Beta distribution quantiles
            alpha = 0.05
            n = 100
            x = int(estimate * n)
            
            lower = stats.beta.ppf(alpha/2, x, n-x+1) if x > 0 else 0
            upper = stats.beta.ppf(1-alpha/2, x+1, n-x) if x < n else 1
        
        return (float(lower), float(upper))


class CalibrationModel:
    """
    Calibration curves to ensure confidence scores are well-calibrated.
    
    A well-calibrated system: when it says 80% confidence,
    the claim should be true ~80% of the time.
    
    Uses Platt scaling and isotonic regression.
    """
    
    def __init__(self):
        # Calibration parameters (learned from validation data)
        self.platt_a: float = 1.0  # Slope
        self.platt_b: float = 0.0  # Intercept
        
        # Isotonic regression bins
        self.isotonic_bins: Optional[np.ndarray] = None
        self.isotonic_values: Optional[np.ndarray] = None
        
        # Calibration quality metrics
        self.expected_calibration_error: float = 0.0
        self.maximum_calibration_error: float = 0.0
    
    def fit_platt_scaling(
        self,
        raw_scores: np.ndarray,
        true_labels: np.ndarray
    ) -> None:
        """
        Fit Platt scaling parameters.
        
        Transforms: P(y=1|x) = 1 / (1 + exp(A*f(x) + B))
        """
        from scipy.optimize import minimize
        
        def neg_log_likelihood(params):
            a, b = params
            calibrated = expit(a * raw_scores + b)
            calibrated = np.clip(calibrated, 1e-15, 1 - 1e-15)
            nll = -np.sum(true_labels * np.log(calibrated) + 
                         (1 - true_labels) * np.log(1 - calibrated))
            return nll
        
        result = minimize(neg_log_likelihood, [1.0, 0.0], method='BFGS')
        self.platt_a, self.platt_b = result.x
    
    def calibrate(self, raw_score: float) -> float:
        """Apply Platt scaling calibration."""
        return float(expit(self.platt_a * raw_score + self.platt_b))
    
    def compute_calibration_error(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        
        ECE = Σ (n_i/N) * |accuracy_i - confidence_i|
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = predictions[in_bin].mean()
                avg_accuracy = labels[in_bin].mean()
                
                gap = abs(avg_accuracy - avg_confidence)
                ece += prop_in_bin * gap
                mce = max(mce, gap)
        
        self.expected_calibration_error = float(ece)
        self.maximum_calibration_error = float(mce)
        
        return {
            "ECE": self.expected_calibration_error,
            "MCE": self.maximum_calibration_error
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. CAUSAL INFERENCE FOR CLAIMS
# ═══════════════════════════════════════════════════════════════════════════


class CausalInferenceAnalyzer:
    """
    Assesses causal evidence strength for health claims.
    
    Uses Bradford Hill criteria and identifies potential confounders.
    """
    
    # Common confounders in public health claims
    COMMON_CONFOUNDERS = {
        "demographics": ["age", "sex", "gender", "race", "ethnicity"],
        "socioeconomic": ["income", "education", "occupation", "insurance"],
        "lifestyle": ["smoking", "alcohol", "diet", "exercise", "bmi", "obesity"],
        "medical": ["comorbidities", "medications", "medical history", "genetics"],
        "environmental": ["pollution", "urban/rural", "climate", "housing"],
        "temporal": ["season", "year", "time of day", "duration"],
    }
    
    # Bradford Hill criteria weights
    BRADFORD_HILL_CRITERIA = {
        "strength": 0.15,           # Strong association
        "consistency": 0.15,        # Replicated across studies
        "specificity": 0.10,        # Specific to exposure
        "temporality": 0.15,        # Cause precedes effect
        "biological_gradient": 0.10, # Dose-response
        "plausibility": 0.10,       # Biological mechanism
        "coherence": 0.10,          # Consistency with knowledge
        "experiment": 0.10,         # Experimental evidence
        "analogy": 0.05,            # Similar relationships exist
    }
    
    def __init__(self):
        pass
    
    def assess_causal_strength(
        self,
        claim_text: str,
        evidence: List[EvidenceItem],
        claim_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess overall causal evidence strength using Bradford Hill criteria.
        
        Returns:
            Dictionary with causal strength score and component analysis
        """
        scores = {}
        
        # 1. Strength of association
        effect_sizes = [e.effect_size for e in evidence 
                       if e.effect_size is not None]
        if effect_sizes:
            # Cohen's d interpretation: 0.2 small, 0.5 medium, 0.8 large
            avg_effect = np.mean(np.abs(effect_sizes))
            scores["strength"] = min(1.0, avg_effect / 0.8)
        else:
            scores["strength"] = 0.3  # Default when unknown
        
        # 2. Consistency across studies
        if len(evidence) >= 2:
            supporting = sum(1 for e in evidence if e.direction == EvidenceDirection.SUPPORTS)
            consistency_ratio = supporting / len(evidence)
            scores["consistency"] = consistency_ratio
        else:
            scores["consistency"] = 0.5  # Single study can't assess consistency
        
        # 3. Specificity (heuristic: specific terms in claim)
        specificity_markers = ["only", "specifically", "exclusively", "uniquely"]
        if any(marker in claim_text.lower() for marker in specificity_markers):
            scores["specificity"] = 0.7
        else:
            scores["specificity"] = 0.4
        
        # 4. Temporality (from study designs)
        rct_or_cohort = sum(1 for e in evidence 
                          if e.source_type in [SourceType.RCT, SourceType.COHORT_STUDY])
        if evidence:
            scores["temporality"] = min(1.0, rct_or_cohort / len(evidence) + 0.3)
        else:
            scores["temporality"] = 0.3
        
        # 5. Biological gradient (dose-response keywords)
        gradient_markers = ["dose", "response", "higher", "lower", "more", "less", 
                           "increase", "decrease", "gradient"]
        if any(marker in claim_text.lower() for marker in gradient_markers):
            scores["biological_gradient"] = 0.6
        else:
            scores["biological_gradient"] = 0.3
        
        # 6. Plausibility (from peer-reviewed sources)
        peer_reviewed = sum(1 for e in evidence 
                          if e.source_type in [SourceType.PEER_REVIEWED, 
                                               SourceType.SYSTEMATIC_REVIEW])
        if evidence:
            scores["plausibility"] = min(1.0, 0.3 + 0.7 * peer_reviewed / len(evidence))
        else:
            scores["plausibility"] = 0.3
        
        # 7. Coherence (consistent with other knowledge - proxy by quality)
        if evidence:
            avg_quality = np.mean([e.study_quality_score for e in evidence])
            scores["coherence"] = avg_quality
        else:
            scores["coherence"] = 0.3
        
        # 8. Experiment (RCT evidence)
        rct_count = sum(1 for e in evidence if e.source_type == SourceType.RCT)
        if evidence:
            scores["experiment"] = min(1.0, rct_count / len(evidence) + 0.2)
        else:
            scores["experiment"] = 0.2
        
        # 9. Analogy (difficult to assess automatically)
        scores["analogy"] = 0.5  # Default neutral
        
        # Weighted overall score
        overall_score = sum(
            scores[criterion] * weight 
            for criterion, weight in self.BRADFORD_HILL_CRITERIA.items()
        )
        
        return {
            "overall_causal_strength": float(overall_score),
            "criteria_scores": scores,
            "interpretation": self._interpret_causal_strength(overall_score),
            "evidence_count": len(evidence)
        }
    
    def _interpret_causal_strength(self, score: float) -> str:
        """Interpret causal strength score."""
        if score >= 0.75:
            return "Strong causal evidence"
        elif score >= 0.55:
            return "Moderate causal evidence"
        elif score >= 0.35:
            return "Weak causal evidence - likely correlational"
        else:
            return "Insufficient evidence for causal claim"
    
    def identify_confounders(
        self,
        claim_text: str,
        mentioned_factors: List[str]
    ) -> Dict[str, Any]:
        """
        Identify potential confounders mentioned and missing in claim.
        
        Args:
            claim_text: The claim being analyzed
            mentioned_factors: Factors explicitly mentioned/controlled
            
        Returns:
            Dictionary with identified and missing confounders
        """
        claim_lower = claim_text.lower()
        mentioned_lower = [m.lower() for m in mentioned_factors]
        
        identified_confounders = []
        missing_confounders = []
        
        for category, factors in self.COMMON_CONFOUNDERS.items():
            for factor in factors:
                # Check if mentioned in claim or as controlled factor
                is_mentioned = (factor in claim_lower or 
                               any(factor in m for m in mentioned_lower))
                
                if is_mentioned:
                    identified_confounders.append({
                        "factor": factor,
                        "category": category,
                        "status": "mentioned"
                    })
                else:
                    # Assess relevance (heuristic based on claim content)
                    relevance = self._assess_confounder_relevance(factor, claim_text)
                    if relevance > 0.5:
                        missing_confounders.append({
                            "factor": factor,
                            "category": category,
                            "relevance": relevance,
                            "status": "potentially_missing"
                        })
        
        # Sort missing by relevance
        missing_confounders.sort(key=lambda x: x["relevance"], reverse=True)
        
        return {
            "identified": identified_confounders,
            "potentially_missing": missing_confounders[:10],  # Top 10 most relevant
            "confounder_coverage": len(identified_confounders) / 
                                  (len(identified_confounders) + len(missing_confounders) + 1),
            "warning": len(missing_confounders) > 5
        }
    
    def _assess_confounder_relevance(self, factor: str, claim_text: str) -> float:
        """
        Assess how relevant a confounder is to a given claim.
        
        Uses keyword matching and domain heuristics.
        """
        claim_lower = claim_text.lower()
        
        # Health-related claims typically need demographic controls
        health_keywords = ["disease", "treatment", "drug", "medicine", "health", 
                          "mortality", "death", "infection", "cancer", "diabetes"]
        
        # If claim is health-related, demographic confounders are relevant
        if any(kw in claim_lower for kw in health_keywords):
            if factor in self.COMMON_CONFOUNDERS["demographics"]:
                return 0.8
            if factor in self.COMMON_CONFOUNDERS["lifestyle"]:
                return 0.7
            if factor in self.COMMON_CONFOUNDERS["medical"]:
                return 0.75
        
        # Socioeconomic confounders for access/disparity claims
        access_keywords = ["access", "disparity", "inequality", "cost", "afford"]
        if any(kw in claim_lower for kw in access_keywords):
            if factor in self.COMMON_CONFOUNDERS["socioeconomic"]:
                return 0.85
        
        # Default moderate relevance
        return 0.4
    
    def distinguish_correlation_causation(
        self,
        claim_text: str,
        evidence: List[EvidenceItem]
    ) -> Dict[str, Any]:
        """
        Analyze whether claim asserts causation and if evidence supports it.
        
        Looks for causal language and checks if study designs support causation.
        """
        # Causal language patterns
        CAUSAL_PATTERNS = [
            "cause", "causes", "caused", "causing",
            "lead to", "leads to", "led to",
            "result in", "results in", "resulted in",
            "due to", "because of", "owing to",
            "effect of", "effects of", "impact of",
            "prevent", "prevents", "prevented",
            "reduce", "reduces", "reduced",
            "increase", "increases", "increased"
        ]
        
        CORRELATIONAL_PATTERNS = [
            "associated with", "linked to", "correlated with",
            "relationship between", "connection between",
            "related to", "co-occur", "together with"
        ]
        
        claim_lower = claim_text.lower()
        
        # Detect claim type
        is_causal_claim = any(pattern in claim_lower for pattern in CAUSAL_PATTERNS)
        is_correlational_claim = any(pattern in claim_lower for pattern in CORRELATIONAL_PATTERNS)
        
        # Check if evidence supports causation
        causal_study_designs = [SourceType.RCT, SourceType.SYSTEMATIC_REVIEW]
        causal_evidence = sum(1 for e in evidence if e.source_type in causal_study_designs)
        
        if is_causal_claim:
            if causal_evidence >= 1:
                assessment = "Causal claim with experimental support"
                risk = "low"
            elif evidence:
                assessment = "Causal claim lacking experimental evidence - interpret cautiously"
                risk = "high"
            else:
                assessment = "Causal claim with no supporting evidence"
                risk = "very_high"
        elif is_correlational_claim:
            assessment = "Appropriately correlational claim"
            risk = "low"
        else:
            assessment = "Claim type unclear - requires manual review"
            risk = "medium"
        
        return {
            "is_causal_claim": is_causal_claim,
            "is_correlational_claim": is_correlational_claim,
            "experimental_evidence_count": causal_evidence,
            "total_evidence": len(evidence),
            "assessment": assessment,
            "overclaim_risk": risk,
            "recommendation": "downgrade_to_correlation" if (is_causal_claim and causal_evidence == 0) else "accept"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONFIDENCE SCORER - INTEGRATES ALL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════


class ClaimConfidenceScorer:
    """
    Main confidence scoring system integrating all statistical methods.
    
    Combines:
    - Bayesian posterior probability
    - Meta-analytic pooled effects
    - Uncertainty quantification
    - Causal inference assessment
    """
    
    GRADE_THRESHOLDS = {
        "A": 0.85,  # Strong confidence, high quality evidence
        "B": 0.70,  # Good confidence, reasonable evidence
        "C": 0.50,  # Moderate confidence, mixed evidence
        "D": 0.30,  # Low confidence, weak evidence
        # Below D threshold = "F"
    }
    
    def __init__(self):
        self.bayesian = BayesianConfidenceEstimator()
        self.meta_analysis = MetaAnalysisEngine()
        self.uncertainty = UncertaintyQuantifier()
        self.causal = CausalInferenceAnalyzer()
        self.calibration = CalibrationModel()
    
    def score_claim(
        self,
        claim_text: str,
        claim_type: ClaimType,
        evidence: List[EvidenceItem],
        apply_calibration: bool = True
    ) -> ConfidenceResult:
        """
        Compute comprehensive confidence score for a claim.
        
        Args:
            claim_text: The claim being verified
            claim_type: Category of claim
            evidence: List of evidence items
            apply_calibration: Whether to apply calibration adjustment
            
        Returns:
            ConfidenceResult with all scoring components
        """
        # 1. Apply source quality weighting
        weighted_evidence = self.meta_analysis.weight_by_source_quality(evidence)
        
        # 2. Bayesian posterior
        posterior_mean, cred_lower, cred_upper = self.bayesian.compute_posterior(
            claim_type, weighted_evidence
        )
        
        # 3. Handle contradictory evidence
        supporting = [e for e in weighted_evidence if e.direction == EvidenceDirection.SUPPORTS]
        contradicting = [e for e in weighted_evidence if e.direction == EvidenceDirection.CONTRADICTS]
        contradiction_analysis = self.bayesian.handle_contradictory_evidence(
            supporting, contradicting
        )
        
        # 4. Meta-analysis (if applicable)
        meta_result = self.meta_analysis.compute_pooled_effect(weighted_evidence)
        
        # 5. Uncertainty decomposition
        uncertainty_result = self.uncertainty.decompose_uncertainty(
            posterior_mean, (cred_lower, cred_upper), weighted_evidence, meta_result
        )
        
        # 6. Causal assessment (for causal claims)
        causal_strength = None
        bradford_hill = None
        confounders = []
        
        if claim_type == ClaimType.CAUSAL:
            causal_result = self.causal.assess_causal_strength(
                claim_text, weighted_evidence
            )
            causal_strength = causal_result["overall_causal_strength"]
            bradford_hill = causal_strength
            
            confounder_result = self.causal.identify_confounders(claim_text, [])
            confounders = [c["factor"] for c in confounder_result["potentially_missing"][:5]]
        
        # 7. Source diversity score
        source_types = set(e.source_type for e in evidence)
        source_diversity = len(source_types) / len(SourceType) if evidence else 0
        
        # 8. Compute final confidence (0-100)
        raw_confidence = posterior_mean
        
        # Adjust for evidence quantity
        evidence_factor = min(1.0, 0.5 + 0.5 * np.log1p(len(evidence)) / np.log1p(10))
        
        # Adjust for causal claims without experimental evidence
        if claim_type == ClaimType.CAUSAL and causal_strength is not None:
            causal_factor = 0.7 + 0.3 * causal_strength
        else:
            causal_factor = 1.0
        
        # Adjust for high uncertainty
        uncertainty_penalty = 1.0 - 0.2 * uncertainty_result["epistemic_uncertainty"]
        
        # Combine adjustments
        adjusted_confidence = raw_confidence * evidence_factor * causal_factor * uncertainty_penalty
        
        # Apply calibration if available and requested
        if apply_calibration:
            adjusted_confidence = self.calibration.calibrate(adjusted_confidence)
        
        # Convert to 0-100 scale
        final_confidence = float(np.clip(adjusted_confidence * 100, 0, 100))
        
        # Assign grade
        grade = "F"
        for g, threshold in sorted(self.GRADE_THRESHOLDS.items(), 
                                   key=lambda x: x[1], reverse=True):
            if final_confidence / 100 >= threshold:
                grade = g
                break
        
        # Generate explanation
        explanation = self._generate_explanation(
            claim_type, evidence, posterior_mean, uncertainty_result,
            contradiction_analysis, meta_result, causal_strength
        )
        
        return ConfidenceResult(
            claim_text=claim_text,
            claim_type=claim_type,
            posterior_probability=float(posterior_mean),
            credible_interval=(float(cred_lower), float(cred_upper)),
            pooled_effect=meta_result["pooled_effect"] if meta_result else None,
            heterogeneity_i2=meta_result["heterogeneity"]["I_squared"] if meta_result else None,
            heterogeneity_p=meta_result["heterogeneity"]["p_value"] if meta_result else None,
            epistemic_uncertainty=uncertainty_result["epistemic_uncertainty"],
            aleatoric_uncertainty=uncertainty_result["aleatoric_uncertainty"],
            total_uncertainty=uncertainty_result["total_uncertainty"],
            causal_strength=causal_strength,
            confounders_identified=confounders,
            bradford_hill_score=bradford_hill,
            evidence_count=len(evidence),
            source_diversity_score=source_diversity,
            calibration_adjustment=0.0,
            final_confidence=final_confidence,
            confidence_grade=grade,
            explanation=explanation
        )
    
    def _generate_explanation(
        self,
        claim_type: ClaimType,
        evidence: List[EvidenceItem],
        posterior: float,
        uncertainty: Dict,
        contradiction: Dict,
        meta: Optional[Dict],
        causal: Optional[float]
    ) -> str:
        """Generate human-readable explanation of confidence score."""
        parts = []
        
        # Evidence summary
        n_evidence = len(evidence)
        if n_evidence == 0:
            parts.append("No supporting evidence found.")
        elif n_evidence == 1:
            parts.append("Based on 1 source.")
        else:
            parts.append(f"Based on {n_evidence} sources.")
        
        # Posterior interpretation
        if posterior >= 0.8:
            parts.append("Strong agreement across sources.")
        elif posterior >= 0.6:
            parts.append("Moderate agreement across sources.")
        elif posterior >= 0.4:
            parts.append("Mixed evidence.")
        else:
            parts.append("Limited or contradicting evidence.")
        
        # Contradiction warning
        if contradiction.get("contradiction_detected"):
            parts.append(f"Note: {contradiction.get('likely_explanation', 'Conflicting sources detected')}.")
        
        # Heterogeneity warning
        if meta and meta["heterogeneity"]["I_squared"] > 50:
            parts.append(f"High variability between sources (I²={meta['heterogeneity']['I_squared']:.0f}%).")
        
        # Causal claim warning
        if claim_type == ClaimType.CAUSAL:
            if causal and causal < 0.5:
                parts.append("Caution: Limited experimental evidence for causal claim.")
        
        # Uncertainty note
        if uncertainty["reducibility_ratio"] > 0.6:
            parts.append("Confidence could improve with more high-quality evidence.")
        
        return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE AND TESTING
# ═══════════════════════════════════════════════════════════════════════════


def example_usage():
    """Demonstrate the confidence scoring system."""
    
    # Initialize scorer
    scorer = ClaimConfidenceScorer()
    
    # Example claim
    claim = "Vitamin D supplementation reduces the risk of respiratory infections by 30%"
    
    # Example evidence
    evidence = [
        EvidenceItem(
            source_id="pmid:12345678",
            source_type=SourceType.SYSTEMATIC_REVIEW,
            direction=EvidenceDirection.SUPPORTS,
            effect_size=0.65,  # Odds ratio ~0.70 as Cohen's d
            effect_size_variance=0.04,
            sample_size=5000,
            confidence_interval=(0.55, 0.85),
            p_value=0.001,
            study_quality_score=0.9,
            recency_weight=1.0
        ),
        EvidenceItem(
            source_id="pmid:23456789",
            source_type=SourceType.RCT,
            direction=EvidenceDirection.SUPPORTS,
            effect_size=0.55,
            effect_size_variance=0.06,
            sample_size=1000,
            study_quality_score=0.85,
            recency_weight=0.95
        ),
        EvidenceItem(
            source_id="pmid:34567890",
            source_type=SourceType.COHORT_STUDY,
            direction=EvidenceDirection.SUPPORTS,
            effect_size=0.40,
            effect_size_variance=0.08,
            sample_size=3000,
            study_quality_score=0.75,
            recency_weight=0.9
        ),
        EvidenceItem(
            source_id="news:example",
            source_type=SourceType.NEWS_OUTLET,
            direction=EvidenceDirection.CONTRADICTS,
            study_quality_score=0.3,
            recency_weight=1.0
        ),
    ]
    
    # Score the claim
    result = scorer.score_claim(
        claim_text=claim,
        claim_type=ClaimType.EFFICACY,
        evidence=evidence
    )
    
    print("=" * 60)
    print("CLAIM CONFIDENCE SCORING RESULT")
    print("=" * 60)
    print(f"Claim: {result.claim_text}")
    print(f"Type: {result.claim_type.value}")
    print()
    print(f"Final Confidence: {result.final_confidence:.1f}% (Grade: {result.confidence_grade})")
    print(f"Posterior Probability: {result.posterior_probability:.3f}")
    print(f"95% Credible Interval: ({result.credible_interval[0]:.3f}, {result.credible_interval[1]:.3f})")
    print()
    print(f"Epistemic Uncertainty: {result.epistemic_uncertainty:.3f}")
    print(f"Aleatoric Uncertainty: {result.aleatoric_uncertainty:.3f}")
    print()
    if result.pooled_effect is not None:
        print(f"Pooled Effect Size: {result.pooled_effect:.3f}")
        print(f"Heterogeneity I²: {result.heterogeneity_i2:.1f}%")
    print()
    print(f"Evidence Count: {result.evidence_count}")
    print(f"Source Diversity: {result.source_diversity_score:.2f}")
    print()
    print(f"Explanation: {result.explanation}")
    
    return result


if __name__ == "__main__":
    example_usage()
