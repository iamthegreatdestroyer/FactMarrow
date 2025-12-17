"""
Confidence Scoring Integration for FactMarrow Orchestrator

Integrates the statistical confidence scoring framework with
the existing agent orchestration system.

@PRISM - Data Science & Statistical Analysis
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

# Import confidence scoring components
from src.services.confidence_scoring import (
    ClaimConfidenceScorer,
    EvidenceItem,
    EvidenceDirection,
    SourceType,
    ClaimType,
    ConfidenceResult
)
from src.services.stats_utils import (
    compute_recency_weight,
    interpret_cohens_d,
    shrinkage_adjustment
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════


class SourceClassifier:
    """
    Classifies sources into quality tiers based on URL, metadata, and content.
    """
    
    # Domain patterns for source classification
    DOMAIN_PATTERNS = {
        SourceType.GOVERNMENT_AGENCY: [
            "cdc.gov", "who.int", "fda.gov", "nih.gov", "hhs.gov",
            "gov.uk/dhsc", "health.gov", "europa.eu/health"
        ],
        SourceType.SYSTEMATIC_REVIEW: [
            "cochranelibrary.com", "cochrane.org"
        ],
        SourceType.PEER_REVIEWED: [
            "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov/pmc",
            "nature.com", "science.org", "thelancet.com",
            "nejm.org", "bmj.com", "jamanetwork.com",
            "cell.com", "springer.com", "wiley.com",
            "journals.plos.org", "frontiersin.org"
        ],
        SourceType.PREPRINT: [
            "arxiv.org", "biorxiv.org", "medrxiv.org",
            "ssrn.com", "preprints.org"
        ],
        SourceType.NEWS_OUTLET: [
            "nytimes.com", "washingtonpost.com", "bbc.com",
            "cnn.com", "reuters.com", "apnews.com",
            "theguardian.com", "npr.org"
        ],
        SourceType.SOCIAL_MEDIA: [
            "twitter.com", "x.com", "facebook.com",
            "reddit.com", "tiktok.com", "instagram.com"
        ]
    }
    
    # Keywords indicating study types
    STUDY_TYPE_KEYWORDS = {
        SourceType.SYSTEMATIC_REVIEW: [
            "systematic review", "meta-analysis", "metaanalysis",
            "cochrane", "pooled analysis"
        ],
        SourceType.RCT: [
            "randomized controlled trial", "randomised controlled trial",
            "rct", "randomized trial", "double-blind", "placebo-controlled"
        ],
        SourceType.COHORT_STUDY: [
            "cohort study", "prospective study", "longitudinal study",
            "follow-up study"
        ],
        SourceType.CASE_CONTROL: [
            "case-control", "case control", "matched controls"
        ],
        SourceType.CASE_SERIES: [
            "case series", "case report", "case study"
        ]
    }
    
    def classify_source(
        self,
        url: Optional[str] = None,
        title: Optional[str] = None,
        abstract: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[SourceType, float]:
        """
        Classify a source and estimate quality score.
        
        Args:
            url: Source URL
            title: Title of source
            abstract: Abstract or snippet
            metadata: Additional metadata
            
        Returns:
            Tuple of (SourceType, quality_score)
        """
        source_type = SourceType.UNKNOWN
        quality_score = 0.5
        
        # Check URL patterns
        if url:
            url_lower = url.lower()
            for stype, patterns in self.DOMAIN_PATTERNS.items():
                if any(pattern in url_lower for pattern in patterns):
                    source_type = stype
                    break
        
        # Check content for study type keywords (overrides URL-based classification)
        text_to_search = " ".join(filter(None, [title, abstract])).lower()
        
        for stype, keywords in self.STUDY_TYPE_KEYWORDS.items():
            if any(kw in text_to_search for kw in keywords):
                source_type = stype
                break
        
        # Assign base quality scores
        BASE_QUALITY = {
            SourceType.SYSTEMATIC_REVIEW: 0.95,
            SourceType.RCT: 0.85,
            SourceType.COHORT_STUDY: 0.75,
            SourceType.CASE_CONTROL: 0.65,
            SourceType.CASE_SERIES: 0.50,
            SourceType.EXPERT_OPINION: 0.55,
            SourceType.GOVERNMENT_AGENCY: 0.85,
            SourceType.PEER_REVIEWED: 0.70,
            SourceType.PREPRINT: 0.45,
            SourceType.NEWS_OUTLET: 0.30,
            SourceType.SOCIAL_MEDIA: 0.10,
            SourceType.UNKNOWN: 0.25,
        }
        
        quality_score = BASE_QUALITY.get(source_type, 0.25)
        
        # Adjust for additional quality indicators
        if metadata:
            # Impact factor adjustment
            if "impact_factor" in metadata:
                if metadata["impact_factor"] > 10:
                    quality_score = min(1.0, quality_score + 0.1)
                elif metadata["impact_factor"] > 5:
                    quality_score = min(1.0, quality_score + 0.05)
            
            # Sample size adjustment
            if "sample_size" in metadata:
                if metadata["sample_size"] > 10000:
                    quality_score = min(1.0, quality_score + 0.05)
                elif metadata["sample_size"] > 1000:
                    quality_score = min(1.0, quality_score + 0.02)
            
            # Peer review confirmation
            if metadata.get("peer_reviewed", False):
                quality_score = min(1.0, quality_score + 0.1)
        
        return source_type, quality_score


class ClaimTypeClassifier:
    """
    Classifies claims into types based on linguistic patterns.
    """
    
    CLAIM_PATTERNS = {
        ClaimType.CAUSAL: [
            "cause", "causes", "caused", "causing",
            "leads to", "lead to", "results in", "result in",
            "due to", "because of", "prevents", "reduces risk",
            "increases risk", "effect of"
        ],
        ClaimType.CORRELATIONAL: [
            "associated with", "linked to", "correlated with",
            "relationship between", "related to", "connection"
        ],
        ClaimType.QUANTITATIVE: [
            r"\d+%", r"\d+ percent", "increase of", "decrease of",
            "fold", "times higher", "times lower", "ratio",
            "per 100,000", "incidence", "rate of"
        ],
        ClaimType.PREVALENCE: [
            "prevalence", "incidence", "cases per", "rate of",
            "affected by", "suffer from", "diagnosed with"
        ],
        ClaimType.EFFICACY: [
            "effective", "efficacy", "works", "treatment",
            "therapy", "intervention", "cure", "heal"
        ],
        ClaimType.SAFETY: [
            "safe", "safety", "risk", "adverse", "side effect",
            "harm", "danger", "toxic"
        ],
        ClaimType.DEFINITIONAL: [
            "is defined as", "refers to", "means", "is a type of",
            "is characterized by"
        ],
        ClaimType.PRESCRIPTIVE: [
            "should", "recommend", "guidelines", "advise",
            "must", "need to", "important to"
        ]
    }
    
    def classify_claim(self, claim_text: str) -> ClaimType:
        """
        Classify a claim based on linguistic patterns.
        
        Args:
            claim_text: The claim to classify
            
        Returns:
            Classified ClaimType
        """
        import re
        claim_lower = claim_text.lower()
        
        # Check each pattern type
        match_scores = {}
        for claim_type, patterns in self.CLAIM_PATTERNS.items():
            matches = sum(1 for p in patterns if re.search(p, claim_lower))
            if matches > 0:
                match_scores[claim_type] = matches
        
        if not match_scores:
            return ClaimType.QUANTITATIVE  # Default
        
        # Return type with most matches
        return max(match_scores.keys(), key=lambda k: match_scores[k])


# ═══════════════════════════════════════════════════════════════════════════
# EVIDENCE BUILDER
# ═══════════════════════════════════════════════════════════════════════════


class EvidenceBuilder:
    """
    Builds EvidenceItem objects from verification results.
    """
    
    def __init__(self):
        self.source_classifier = SourceClassifier()
    
    def from_verification_source(
        self,
        source: Dict[str, Any],
        supports_claim: bool,
        publication_date: Optional[datetime] = None
    ) -> EvidenceItem:
        """
        Create EvidenceItem from a verification source dictionary.
        
        Args:
            source: Source dictionary with url, title, snippet, etc.
            supports_claim: Whether source supports the claim
            publication_date: Publication date if known
            
        Returns:
            Constructed EvidenceItem
        """
        # Classify source
        source_type, base_quality = self.source_classifier.classify_source(
            url=source.get("url"),
            title=source.get("title"),
            abstract=source.get("snippet") or source.get("abstract"),
            metadata=source.get("metadata")
        )
        
        # Determine direction
        if supports_claim:
            direction = EvidenceDirection.SUPPORTS
        elif source.get("contradicts", False):
            direction = EvidenceDirection.CONTRADICTS
        elif source.get("inconclusive", False):
            direction = EvidenceDirection.INCONCLUSIVE
        else:
            direction = EvidenceDirection.NEUTRAL
        
        # Compute recency weight
        if publication_date:
            recency_weight = compute_recency_weight(publication_date)
        elif source.get("date"):
            try:
                pub_date = datetime.fromisoformat(source["date"])
                recency_weight = compute_recency_weight(pub_date)
            except:
                recency_weight = 0.8  # Default for unparseable dates
        else:
            recency_weight = 0.7  # Default for unknown dates
        
        # Extract effect size if available
        effect_size = source.get("effect_size")
        effect_variance = source.get("effect_variance")
        if effect_size is not None and effect_variance is None:
            # Estimate variance from sample size or CI
            sample_size = source.get("sample_size", 100)
            effect_variance = 4.0 / sample_size  # Rough approximation
        
        return EvidenceItem(
            source_id=source.get("url") or source.get("id", "unknown"),
            source_type=source_type,
            direction=direction,
            effect_size=effect_size,
            effect_size_variance=effect_variance,
            sample_size=source.get("sample_size"),
            confidence_interval=source.get("confidence_interval"),
            p_value=source.get("p_value"),
            study_quality_score=base_quality,
            recency_weight=recency_weight,
            raw_text=source.get("snippet") or source.get("abstract")
        )
    
    def from_pubmed_result(self, result: Dict[str, Any]) -> EvidenceItem:
        """
        Create EvidenceItem from PubMed API result.
        """
        # Extract publication date
        pub_date = None
        if "pubdate" in result:
            try:
                pub_date = datetime.strptime(result["pubdate"], "%Y %b %d")
            except:
                try:
                    pub_date = datetime.strptime(result["pubdate"][:4], "%Y")
                except:
                    pass
        
        return self.from_verification_source(
            source={
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{result.get('uid', '')}/",
                "title": result.get("title", ""),
                "abstract": result.get("abstract", ""),
                "date": pub_date.isoformat() if pub_date else None,
                "metadata": {
                    "peer_reviewed": True,
                    "source": "pubmed"
                }
            },
            supports_claim=result.get("supports", True),
            publication_date=pub_date
        )


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════


class ConfidenceScoringIntegration:
    """
    Integration layer between FactMarrow orchestrator and confidence scoring.
    
    Provides methods to be called by the Verification Specialist agent
    and Report Writer agent.
    """
    
    def __init__(self):
        self.scorer = ClaimConfidenceScorer()
        self.evidence_builder = EvidenceBuilder()
        self.claim_classifier = ClaimTypeClassifier()
        
        # Cache for scoring results
        self._cache: Dict[str, ConfidenceResult] = {}
    
    def score_verification_result(
        self,
        claim_text: str,
        supporting_sources: List[Dict[str, Any]],
        contradicting_sources: List[Dict[str, Any]],
        neutral_sources: Optional[List[Dict[str, Any]]] = None
    ) -> ConfidenceResult:
        """
        Score a claim based on verification results from agents.
        
        This is the main integration point called by the orchestrator.
        
        Args:
            claim_text: The claim being verified
            supporting_sources: Sources that support the claim
            contradicting_sources: Sources that contradict the claim
            neutral_sources: Sources with inconclusive evidence
            
        Returns:
            Complete ConfidenceResult
        """
        # Classify claim type
        claim_type = self.claim_classifier.classify_claim(claim_text)
        
        # Build evidence items
        evidence = []
        
        for source in supporting_sources:
            evidence.append(self.evidence_builder.from_verification_source(
                source, supports_claim=True
            ))
        
        for source in contradicting_sources:
            source["contradicts"] = True
            evidence.append(self.evidence_builder.from_verification_source(
                source, supports_claim=False
            ))
        
        if neutral_sources:
            for source in neutral_sources:
                source["inconclusive"] = True
                evidence.append(self.evidence_builder.from_verification_source(
                    source, supports_claim=True
                ))
        
        # Score the claim
        result = self.scorer.score_claim(
            claim_text=claim_text,
            claim_type=claim_type,
            evidence=evidence
        )
        
        # Cache result
        self._cache[claim_text[:100]] = result
        
        return result
    
    def format_for_report(self, result: ConfidenceResult) -> Dict[str, Any]:
        """
        Format confidence result for inclusion in verification report.
        
        Returns:
            Dictionary suitable for report generation
        """
        return {
            "claim": result.claim_text,
            "confidence": {
                "score": round(result.final_confidence, 1),
                "grade": result.confidence_grade,
                "posterior_probability": round(result.posterior_probability, 3),
                "credible_interval": [
                    round(result.credible_interval[0], 3),
                    round(result.credible_interval[1], 3)
                ]
            },
            "uncertainty": {
                "epistemic": round(result.epistemic_uncertainty, 3),
                "aleatoric": round(result.aleatoric_uncertainty, 3),
                "total": round(result.total_uncertainty, 3)
            },
            "evidence_summary": {
                "count": result.evidence_count,
                "source_diversity": round(result.source_diversity_score, 2)
            },
            "meta_analysis": {
                "pooled_effect": round(result.pooled_effect, 3) if result.pooled_effect else None,
                "heterogeneity_i2": round(result.heterogeneity_i2, 1) if result.heterogeneity_i2 else None
            } if result.pooled_effect else None,
            "causal_assessment": {
                "strength": round(result.causal_strength, 3) if result.causal_strength else None,
                "confounders_to_check": result.confounders_identified[:5]
            } if result.causal_strength else None,
            "explanation": result.explanation
        }
    
    def batch_score(
        self,
        verification_results: List[Dict[str, Any]]
    ) -> List[ConfidenceResult]:
        """
        Score multiple claims efficiently.
        
        Args:
            verification_results: List of dicts with claim_text, supporting, contradicting
            
        Returns:
            List of ConfidenceResults
        """
        results = []
        for item in verification_results:
            result = self.score_verification_result(
                claim_text=item["claim_text"],
                supporting_sources=item.get("supporting_sources", []),
                contradicting_sources=item.get("contradicting_sources", []),
                neutral_sources=item.get("neutral_sources")
            )
            results.append(result)
        
        return results
    
    def get_overall_document_confidence(
        self,
        claim_results: List[ConfidenceResult]
    ) -> Dict[str, Any]:
        """
        Compute overall document confidence from individual claim scores.
        
        Args:
            claim_results: List of scored claims
            
        Returns:
            Document-level confidence summary
        """
        if not claim_results:
            return {
                "overall_confidence": 0,
                "overall_grade": "F",
                "claims_analyzed": 0
            }
        
        # Weight by evidence count (more evidence = more important)
        weights = [max(1, r.evidence_count) for r in claim_results]
        total_weight = sum(weights)
        
        weighted_confidence = sum(
            r.final_confidence * w for r, w in zip(claim_results, weights)
        ) / total_weight
        
        # Apply shrinkage for few claims
        adjusted_confidence = shrinkage_adjustment(
            weighted_confidence / 100,
            n_evidence=len(claim_results),
            prior_mean=0.5,
            shrinkage_k=2.0
        ) * 100
        
        # Grade distribution
        grade_counts = {}
        for r in claim_results:
            grade_counts[r.confidence_grade] = grade_counts.get(r.confidence_grade, 0) + 1
        
        # Overall grade (most frequent, with tie-break to lower)
        grade_order = ["F", "D", "C", "B", "A"]
        overall_grade = max(grade_counts.keys(), 
                          key=lambda g: (grade_counts[g], -grade_order.index(g)))
        
        return {
            "overall_confidence": round(adjusted_confidence, 1),
            "overall_grade": overall_grade,
            "claims_analyzed": len(claim_results),
            "grade_distribution": grade_counts,
            "high_confidence_claims": sum(1 for r in claim_results if r.final_confidence >= 70),
            "low_confidence_claims": sum(1 for r in claim_results if r.final_confidence < 50),
            "average_evidence_per_claim": sum(r.evidence_count for r in claim_results) / len(claim_results)
        }


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION FOR EASY INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════


def create_confidence_scorer() -> ConfidenceScoringIntegration:
    """
    Factory function to create pre-configured confidence scorer.
    
    Usage in orchestrator:
        from src.services.confidence_integration import create_confidence_scorer
        
        scorer = create_confidence_scorer()
        result = scorer.score_verification_result(
            claim_text="...",
            supporting_sources=[...],
            contradicting_sources=[...]
        )
    """
    return ConfidenceScoringIntegration()
