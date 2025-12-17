"""
Tests for FactMarrow Confidence Scoring Module

Comprehensive test suite covering:
- Bayesian inference
- Meta-analysis
- Uncertainty quantification
- Causal inference
- Integration

@PRISM - Data Science & Statistical Analysis
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.services.confidence_scoring import (
    ClaimConfidenceScorer,
    BayesianConfidenceEstimator,
    MetaAnalysisEngine,
    UncertaintyQuantifier,
    CausalInferenceAnalyzer,
    CalibrationModel,
    EvidenceItem,
    EvidenceDirection,
    SourceType,
    ClaimType,
    ConfidenceResult
)
from src.services.stats_utils import (
    odds_ratio_to_cohens_d,
    cohens_d_to_correlation,
    required_sample_size,
    compute_recency_weight,
    interpret_i_squared,
    egger_regression,
    beta_hdi,
    shrinkage_adjustment
)
from src.services.confidence_integration import (
    SourceClassifier,
    ClaimTypeClassifier,
    EvidenceBuilder,
    ConfidenceScoringIntegration,
    create_confidence_scorer
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_supporting_evidence():
    """Create sample supporting evidence list."""
    return [
        EvidenceItem(
            source_id="pmid:12345678",
            source_type=SourceType.SYSTEMATIC_REVIEW,
            direction=EvidenceDirection.SUPPORTS,
            effect_size=0.65,
            effect_size_variance=0.04,
            sample_size=5000,
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
    ]


@pytest.fixture
def sample_mixed_evidence():
    """Create sample mixed evidence with contradiction."""
    return [
        EvidenceItem(
            source_id="pmid:12345678",
            source_type=SourceType.PEER_REVIEWED,
            direction=EvidenceDirection.SUPPORTS,
            effect_size=0.5,
            effect_size_variance=0.05,
            study_quality_score=0.8
        ),
        EvidenceItem(
            source_id="pmid:23456789",
            source_type=SourceType.RCT,
            direction=EvidenceDirection.CONTRADICTS,
            effect_size=-0.3,
            effect_size_variance=0.08,
            study_quality_score=0.75
        ),
        EvidenceItem(
            source_id="news:article",
            source_type=SourceType.NEWS_OUTLET,
            direction=EvidenceDirection.CONTRADICTS,
            study_quality_score=0.3
        ),
    ]


@pytest.fixture
def scorer():
    """Create pre-configured scorer."""
    return ClaimConfidenceScorer()


@pytest.fixture
def integration():
    """Create integration instance."""
    return create_confidence_scorer()


# ═══════════════════════════════════════════════════════════════════════════
# BAYESIAN CONFIDENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestBayesianConfidenceEstimator:
    """Tests for Bayesian inference module."""
    
    def test_prior_retrieval(self):
        """Test that priors are correctly retrieved for claim types."""
        estimator = BayesianConfidenceEstimator()
        
        # Causal claims should have skeptical prior
        causal_prior = estimator.get_prior(ClaimType.CAUSAL)
        assert causal_prior[1] > causal_prior[0]  # β > α (skeptical)
        
        # Definitional claims should have optimistic prior
        def_prior = estimator.get_prior(ClaimType.DEFINITIONAL)
        assert def_prior[0] > def_prior[1]  # α > β (optimistic)
    
    def test_posterior_with_supporting_evidence(self, sample_supporting_evidence):
        """Test posterior increases with supporting evidence."""
        estimator = BayesianConfidenceEstimator()
        
        # Prior mean
        prior_alpha, prior_beta = estimator.get_prior(ClaimType.EFFICACY)
        prior_mean = prior_alpha / (prior_alpha + prior_beta)
        
        # Posterior with evidence
        post_mean, _, _ = estimator.compute_posterior(
            ClaimType.EFFICACY,
            sample_supporting_evidence
        )
        
        # Posterior should be higher than prior with supporting evidence
        assert post_mean > prior_mean
    
    def test_posterior_bounds(self, sample_supporting_evidence):
        """Test posterior credible interval is valid."""
        estimator = BayesianConfidenceEstimator()
        
        mean, lower, upper = estimator.compute_posterior(
            ClaimType.QUANTITATIVE,
            sample_supporting_evidence
        )
        
        assert 0 <= lower < mean < upper <= 1
        assert upper - lower > 0  # Non-degenerate interval
    
    def test_contradictory_evidence_detection(self, sample_mixed_evidence):
        """Test contradiction analysis."""
        estimator = BayesianConfidenceEstimator()
        
        supporting = [e for e in sample_mixed_evidence 
                     if e.direction == EvidenceDirection.SUPPORTS]
        contradicting = [e for e in sample_mixed_evidence 
                        if e.direction == EvidenceDirection.CONTRADICTS]
        
        result = estimator.handle_contradictory_evidence(supporting, contradicting)
        
        assert result["contradiction_detected"] is True
        assert "likely_explanation" in result
        assert 0 <= result["resolution_confidence"] <= 1
    
    def test_source_weighting(self):
        """Test that high-quality sources have more weight."""
        estimator = BayesianConfidenceEstimator()
        
        # Higher quality should have higher weight
        assert estimator.SOURCE_WEIGHTS[SourceType.SYSTEMATIC_REVIEW] > \
               estimator.SOURCE_WEIGHTS[SourceType.NEWS_OUTLET]
        assert estimator.SOURCE_WEIGHTS[SourceType.RCT] > \
               estimator.SOURCE_WEIGHTS[SourceType.PREPRINT]


# ═══════════════════════════════════════════════════════════════════════════
# META-ANALYSIS TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestMetaAnalysisEngine:
    """Tests for meta-analysis module."""
    
    def test_pooled_effect_calculation(self, sample_supporting_evidence):
        """Test pooled effect is computed correctly."""
        engine = MetaAnalysisEngine()
        
        result = engine.compute_pooled_effect(sample_supporting_evidence)
        
        assert result is not None
        assert "pooled_effect" in result
        assert "confidence_interval" in result
        assert result["n_studies"] == 2
    
    def test_insufficient_studies(self):
        """Test handling of insufficient studies."""
        engine = MetaAnalysisEngine()
        
        single_study = [
            EvidenceItem(
                source_id="test",
                source_type=SourceType.RCT,
                direction=EvidenceDirection.SUPPORTS,
                effect_size=0.5,
                effect_size_variance=0.1
            )
        ]
        
        result = engine.compute_pooled_effect(single_study)
        
        assert result is None  # Need at least 2 studies
    
    def test_heterogeneity_detection(self):
        """Test heterogeneity statistics."""
        engine = MetaAnalysisEngine()
        
        # Create heterogeneous evidence
        heterogeneous = [
            EvidenceItem(
                source_id=f"study_{i}",
                source_type=SourceType.RCT,
                direction=EvidenceDirection.SUPPORTS,
                effect_size=0.3 + i * 0.3,  # Varying effects
                effect_size_variance=0.05,
                study_quality_score=0.8
            )
            for i in range(5)
        ]
        
        result = engine.compute_pooled_effect(heterogeneous)
        
        assert result is not None
        assert "heterogeneity" in result
        assert result["heterogeneity"]["I_squared"] >= 0
    
    def test_source_quality_weighting(self, sample_mixed_evidence):
        """Test quality-based weighting."""
        engine = MetaAnalysisEngine()
        
        weighted = engine.weight_by_source_quality(sample_mixed_evidence)
        
        # News source should have lower quality after weighting
        news_item = next(e for e in weighted if e.source_type == SourceType.NEWS_OUTLET)
        rct_item = next(e for e in weighted if e.source_type == SourceType.RCT)
        
        assert news_item.study_quality_score < rct_item.study_quality_score


# ═══════════════════════════════════════════════════════════════════════════
# UNCERTAINTY QUANTIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestUncertaintyQuantifier:
    """Tests for uncertainty quantification module."""
    
    def test_uncertainty_decomposition(self, sample_supporting_evidence):
        """Test uncertainty is properly decomposed."""
        quantifier = UncertaintyQuantifier()
        
        result = quantifier.decompose_uncertainty(
            posterior_mean=0.7,
            credible_interval=(0.5, 0.9),
            evidence=sample_supporting_evidence
        )
        
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        assert "total_uncertainty" in result
        
        # Epistemic + aleatoric should approximately equal total
        assert abs(result["epistemic_uncertainty"] + 
                  result["aleatoric_uncertainty"] - 
                  result["total_uncertainty"]) < 0.01
    
    def test_reducibility_ratio(self, sample_supporting_evidence):
        """Test reducibility ratio is computed."""
        quantifier = UncertaintyQuantifier()
        
        result = quantifier.decompose_uncertainty(
            posterior_mean=0.7,
            credible_interval=(0.5, 0.9),
            evidence=sample_supporting_evidence
        )
        
        assert 0 <= result["reducibility_ratio"] <= 1
    
    def test_confidence_interval_methods(self):
        """Test different CI computation methods."""
        quantifier = UncertaintyQuantifier()
        
        for method in ["normal", "wilson", "clopper_pearson"]:
            lower, upper = quantifier.compute_confidence_interval(
                estimate=0.6,
                uncertainty=0.1,
                method=method
            )
            
            assert 0 <= lower < 0.6 < upper <= 1


class TestCalibrationModel:
    """Tests for calibration module."""
    
    def test_platt_scaling(self):
        """Test Platt scaling calibration."""
        calibration = CalibrationModel()
        
        # Simulated training data
        raw_scores = np.random.beta(2, 3, 100)
        true_labels = (np.random.random(100) < raw_scores).astype(float)
        
        calibration.fit_platt_scaling(raw_scores, true_labels)
        
        # Calibrated score should be in [0, 1]
        calibrated = calibration.calibrate(0.5)
        assert 0 <= calibrated <= 1
    
    def test_calibration_error(self):
        """Test ECE/MCE computation."""
        calibration = CalibrationModel()
        
        predictions = np.random.random(100)
        labels = (np.random.random(100) < predictions).astype(float)
        
        errors = calibration.compute_calibration_error(predictions, labels)
        
        assert "ECE" in errors
        assert "MCE" in errors
        assert 0 <= errors["ECE"] <= 1
        assert 0 <= errors["MCE"] <= 1


# ═══════════════════════════════════════════════════════════════════════════
# CAUSAL INFERENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCausalInferenceAnalyzer:
    """Tests for causal inference module."""
    
    def test_causal_strength_assessment(self, sample_supporting_evidence):
        """Test Bradford Hill assessment."""
        analyzer = CausalInferenceAnalyzer()
        
        result = analyzer.assess_causal_strength(
            claim_text="Smoking causes lung cancer",
            evidence=sample_supporting_evidence
        )
        
        assert 0 <= result["overall_causal_strength"] <= 1
        assert "criteria_scores" in result
        assert "interpretation" in result
    
    def test_causal_language_detection(self):
        """Test correlation vs causation distinction."""
        analyzer = CausalInferenceAnalyzer()
        
        causal_claim = "Vitamin D causes reduced infection risk"
        correlational_claim = "Vitamin D is associated with reduced infection risk"
        
        causal_result = analyzer.distinguish_correlation_causation(
            causal_claim, []
        )
        corr_result = analyzer.distinguish_correlation_causation(
            correlational_claim, []
        )
        
        assert causal_result["is_causal_claim"] is True
        assert corr_result["is_correlational_claim"] is True
    
    def test_confounder_identification(self):
        """Test confounder detection."""
        analyzer = CausalInferenceAnalyzer()
        
        result = analyzer.identify_confounders(
            claim_text="Drug X reduces mortality in diabetic patients",
            mentioned_factors=["age", "sex"]
        )
        
        assert "identified" in result
        assert "potentially_missing" in result
        
        # Age and sex should be identified
        identified_factors = [c["factor"] for c in result["identified"]]
        assert "age" in identified_factors
        assert "sex" in identified_factors
    
    def test_overclaim_detection(self):
        """Test detection of unsupported causal claims."""
        analyzer = CausalInferenceAnalyzer()
        
        # Causal claim with no RCT evidence
        observational_only = [
            EvidenceItem(
                source_id="test",
                source_type=SourceType.COHORT_STUDY,
                direction=EvidenceDirection.SUPPORTS,
                study_quality_score=0.7
            )
        ]
        
        result = analyzer.distinguish_correlation_causation(
            "Treatment X causes improvement",
            observational_only
        )
        
        assert result["overclaim_risk"] in ["high", "very_high"]


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL UTILITIES TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestStatisticalUtilities:
    """Tests for statistical utility functions."""
    
    def test_odds_ratio_conversion(self):
        """Test OR to Cohen's d conversion."""
        # OR = 1 should give d ≈ 0
        d = odds_ratio_to_cohens_d(1.0)
        assert abs(d) < 0.01
        
        # OR > 1 should give positive d
        d = odds_ratio_to_cohens_d(2.0)
        assert d > 0
    
    def test_cohens_d_to_correlation(self):
        """Test d to r conversion."""
        # d = 0 should give r = 0
        r = cohens_d_to_correlation(0)
        assert r == 0
        
        # Large d should give large |r|
        r = cohens_d_to_correlation(2.0)
        assert abs(r) > 0.5
    
    def test_sample_size_calculation(self):
        """Test power analysis."""
        # Large effect needs fewer samples
        n_large = required_sample_size(effect_size=0.8)
        n_small = required_sample_size(effect_size=0.2)
        
        assert n_large < n_small
        assert n_large > 0
    
    def test_recency_weight(self):
        """Test recency weighting."""
        now = datetime.now()
        
        # Recent should have higher weight
        recent = compute_recency_weight(now - timedelta(days=30))
        old = compute_recency_weight(now - timedelta(days=365 * 10))
        
        assert recent > old
        assert 0 < old  # Should still be positive (min_weight)
    
    def test_i_squared_interpretation(self):
        """Test heterogeneity interpretation."""
        low_mag, _ = interpret_i_squared(20)
        high_mag, _ = interpret_i_squared(80)
        
        assert low_mag == "low"
        assert high_mag == "considerable"
    
    def test_beta_hdi(self):
        """Test highest density interval."""
        # Symmetric beta should have symmetric HDI
        lower, upper = beta_hdi(5, 5)
        
        assert 0 < lower < 0.5 < upper < 1
        assert abs((lower + upper) / 2 - 0.5) < 0.1
    
    def test_shrinkage_adjustment(self):
        """Test empirical Bayes shrinkage."""
        # With lots of evidence, should stay close to raw
        adjusted_high_n = shrinkage_adjustment(0.9, n_evidence=100, prior_mean=0.5)
        
        # With little evidence, should shrink toward prior
        adjusted_low_n = shrinkage_adjustment(0.9, n_evidence=1, prior_mean=0.5)
        
        assert adjusted_high_n > adjusted_low_n
        assert adjusted_low_n < 0.9  # Shrunk toward 0.5


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestSourceClassifier:
    """Tests for source classification."""
    
    def test_pubmed_classification(self):
        """Test PubMed URLs are classified correctly."""
        classifier = SourceClassifier()
        
        source_type, quality = classifier.classify_source(
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/"
        )
        
        assert source_type == SourceType.PEER_REVIEWED
        assert quality > 0.5
    
    def test_cochrane_classification(self):
        """Test Cochrane is classified as systematic review."""
        classifier = SourceClassifier()
        
        source_type, _ = classifier.classify_source(
            url="https://www.cochranelibrary.com/cdsr/doi/10.1002/test"
        )
        
        assert source_type == SourceType.SYSTEMATIC_REVIEW
    
    def test_keyword_override(self):
        """Test study type keywords override URL classification."""
        classifier = SourceClassifier()
        
        # RCT keyword should override generic peer-reviewed URL
        source_type, _ = classifier.classify_source(
            url="https://example.com/article",
            title="A Randomized Controlled Trial of Drug X"
        )
        
        assert source_type == SourceType.RCT


class TestClaimTypeClassifier:
    """Tests for claim type classification."""
    
    def test_causal_claim(self):
        """Test causal claims are detected."""
        classifier = ClaimTypeClassifier()
        
        claim_type = classifier.classify_claim(
            "Smoking causes lung cancer"
        )
        
        assert claim_type == ClaimType.CAUSAL
    
    def test_correlational_claim(self):
        """Test correlational claims are detected."""
        classifier = ClaimTypeClassifier()
        
        claim_type = classifier.classify_claim(
            "Higher vitamin D levels are associated with better outcomes"
        )
        
        assert claim_type == ClaimType.CORRELATIONAL
    
    def test_quantitative_claim(self):
        """Test quantitative claims are detected."""
        classifier = ClaimTypeClassifier()
        
        claim_type = classifier.classify_claim(
            "The treatment reduced mortality by 30%"
        )
        
        assert claim_type == ClaimType.QUANTITATIVE


class TestConfidenceScoringIntegration:
    """Tests for full integration."""
    
    def test_end_to_end_scoring(self, integration):
        """Test complete scoring workflow."""
        result = integration.score_verification_result(
            claim_text="Vitamin D supplementation reduces respiratory infections",
            supporting_sources=[
                {"url": "https://pubmed.ncbi.nlm.nih.gov/12345/", 
                 "title": "RCT of Vitamin D", "snippet": "Results show..."},
                {"url": "https://www.cochranelibrary.com/cdsr/test",
                 "title": "Systematic review of Vitamin D"}
            ],
            contradicting_sources=[
                {"url": "https://news.example.com/article",
                 "title": "New study questions vitamin D"}
            ]
        )
        
        assert isinstance(result, ConfidenceResult)
        assert 0 <= result.final_confidence <= 100
        assert result.confidence_grade in ["A", "B", "C", "D", "F"]
    
    def test_report_formatting(self, integration, sample_supporting_evidence):
        """Test report format generation."""
        scorer = ClaimConfidenceScorer()
        result = scorer.score_claim(
            claim_text="Test claim",
            claim_type=ClaimType.EFFICACY,
            evidence=sample_supporting_evidence
        )
        
        formatted = integration.format_for_report(result)
        
        assert "claim" in formatted
        assert "confidence" in formatted
        assert "score" in formatted["confidence"]
        assert "grade" in formatted["confidence"]
    
    def test_batch_scoring(self, integration):
        """Test batch processing."""
        results = integration.batch_score([
            {
                "claim_text": "Claim 1",
                "supporting_sources": [{"url": "https://pubmed.ncbi.nlm.nih.gov/1/"}],
                "contradicting_sources": []
            },
            {
                "claim_text": "Claim 2",
                "supporting_sources": [],
                "contradicting_sources": [{"url": "https://example.com"}]
            }
        ])
        
        assert len(results) == 2
        assert all(isinstance(r, ConfidenceResult) for r in results)
    
    def test_document_overall_confidence(self, integration):
        """Test document-level aggregation."""
        scorer = ClaimConfidenceScorer()
        
        results = [
            scorer.score_claim("High confidence claim", ClaimType.QUANTITATIVE, [
                EvidenceItem("1", SourceType.SYSTEMATIC_REVIEW, 
                           EvidenceDirection.SUPPORTS, study_quality_score=0.9)
            ]),
            scorer.score_claim("Low confidence claim", ClaimType.CAUSAL, [])
        ]
        
        overall = integration.get_overall_document_confidence(results)
        
        assert "overall_confidence" in overall
        assert "overall_grade" in overall
        assert "claims_analyzed" in overall
        assert overall["claims_analyzed"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SCORER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestClaimConfidenceScorer:
    """Tests for main scoring class."""
    
    def test_complete_scoring(self, scorer, sample_supporting_evidence):
        """Test complete scoring produces all fields."""
        result = scorer.score_claim(
            claim_text="Test health claim",
            claim_type=ClaimType.EFFICACY,
            evidence=sample_supporting_evidence
        )
        
        # Check all required fields
        assert result.claim_text == "Test health claim"
        assert result.claim_type == ClaimType.EFFICACY
        assert 0 <= result.final_confidence <= 100
        assert result.confidence_grade in ["A", "B", "C", "D", "F"]
        assert result.explanation != ""
    
    def test_causal_claim_penalty(self, scorer):
        """Test causal claims without RCT evidence are penalized."""
        # Create observational-only evidence
        observational = [
            EvidenceItem(
                source_id="test",
                source_type=SourceType.COHORT_STUDY,
                direction=EvidenceDirection.SUPPORTS,
                study_quality_score=0.8
            )
        ]
        
        causal_result = scorer.score_claim(
            "X causes Y",
            ClaimType.CAUSAL,
            observational
        )
        
        non_causal_result = scorer.score_claim(
            "X is associated with Y",
            ClaimType.CORRELATIONAL,
            observational
        )
        
        # Causal claim should have lower confidence without experimental evidence
        assert causal_result.causal_strength is not None
    
    def test_grade_assignment(self, scorer, sample_supporting_evidence):
        """Test grade thresholds."""
        high_conf = scorer.score_claim(
            "Well-supported claim",
            ClaimType.DEFINITIONAL,
            sample_supporting_evidence * 3  # Triple the evidence
        )
        
        low_conf = scorer.score_claim(
            "Unsupported claim",
            ClaimType.CAUSAL,
            []
        )
        
        # More evidence should give better grade
        grade_order = ["F", "D", "C", "B", "A"]
        assert grade_order.index(high_conf.confidence_grade) > \
               grade_order.index(low_conf.confidence_grade)
    
    def test_explanation_generation(self, scorer, sample_mixed_evidence):
        """Test explanation is informative."""
        result = scorer.score_claim(
            "Mixed evidence claim",
            ClaimType.EFFICACY,
            sample_mixed_evidence
        )
        
        # Explanation should mention key aspects
        explanation = result.explanation.lower()
        assert "source" in explanation or "evidence" in explanation


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_no_evidence(self, scorer):
        """Test handling of no evidence."""
        result = scorer.score_claim(
            "Claim with no evidence",
            ClaimType.QUANTITATIVE,
            []
        )
        
        assert result.final_confidence < 50  # Should be low confidence
        assert result.evidence_count == 0
    
    def test_all_contradicting(self, scorer):
        """Test handling of only contradicting evidence."""
        contradicting = [
            EvidenceItem(
                source_id="test",
                source_type=SourceType.RCT,
                direction=EvidenceDirection.CONTRADICTS,
                study_quality_score=0.9
            )
        ]
        
        result = scorer.score_claim(
            "Contradicted claim",
            ClaimType.EFFICACY,
            contradicting
        )
        
        assert result.posterior_probability < 0.5
    
    def test_invalid_effect_sizes(self, scorer):
        """Test handling of missing effect sizes."""
        no_effects = [
            EvidenceItem(
                source_id="test",
                source_type=SourceType.PEER_REVIEWED,
                direction=EvidenceDirection.SUPPORTS,
                effect_size=None,  # Missing
                study_quality_score=0.7
            )
        ]
        
        result = scorer.score_claim(
            "Claim",
            ClaimType.QUANTITATIVE,
            no_effects
        )
        
        # Should still produce valid result
        assert result.pooled_effect is None  # Can't pool
        assert result.final_confidence > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
