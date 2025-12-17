# FactMarrow Statistical Confidence Scoring Framework

## Overview

FactMarrow's confidence scoring system uses sophisticated statistical methods to assign calibrated confidence scores to verified claims. The framework integrates four major components:

1. **Bayesian Confidence Framework** - Prior-based belief updating
2. **Meta-Analysis Integration** - Multi-source evidence combination
3. **Uncertainty Quantification** - Epistemic/aleatoric decomposition
4. **Causal Inference Analysis** - Bradford Hill criteria assessment

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Claim Verification Request                       │
│            (claim_text, supporting_sources, contradicting)          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SourceClassifier                               │
│   • URL pattern matching (PubMed, Cochrane, CDC, etc.)             │
│   • Study type keyword detection (RCT, meta-analysis, etc.)        │
│   • Quality score assignment (0.1 - 0.95)                          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ClaimTypeClassifier                            │
│   • Causal language detection                                       │
│   • Quantitative pattern recognition                                │
│   • Claim categorization (CAUSAL, CORRELATIONAL, EFFICACY, etc.)   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ClaimConfidenceScorer                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │    Bayesian      │  │   Meta-Analysis  │  │   Uncertainty    │  │
│  │    Estimator     │  │     Engine       │  │   Quantifier     │  │
│  │                  │  │                  │  │                  │  │
│  │ • Beta priors    │  │ • Random effects │  │ • Epistemic      │  │
│  │ • Weighted       │  │ • I² statistic   │  │ • Aleatoric      │  │
│  │   updating       │  │ • DerSimonian-   │  │ • Decomposition  │  │
│  │ • Contradiction  │  │   Laird          │  │ • Calibration    │  │
│  │   handling       │  │ • Quality weight │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                │                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   CausalInferenceAnalyzer                     │  │
│  │   • Bradford Hill criteria scoring                            │  │
│  │   • Confounder identification                                 │  │
│  │   • Correlation vs causation distinction                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ConfidenceResult                               │
│   • final_confidence: 0-100                                         │
│   • confidence_grade: A/B/C/D/F                                     │
│   • posterior_probability, credible_interval                        │
│   • uncertainty decomposition                                       │
│   • causal assessment (if applicable)                              │
│   • human-readable explanation                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Bayesian Confidence Framework

### Prior Probabilities by Claim Type

Different claim types have different base rates of accuracy:

| Claim Type     | Prior α | Prior β | Prior Mean | Rationale                        |
| -------------- | ------- | ------- | ---------- | -------------------------------- |
| Definitional   | 4.0     | 1.0     | 0.80       | Usually accurate, verifiable     |
| Prevalence     | 3.5     | 1.5     | 0.70       | Generally reliable from agencies |
| Quantitative   | 3.0     | 2.0     | 0.60       | Often accurate when sourced      |
| Methodological | 3.0     | 2.0     | 0.60       | Verifiable from publications     |
| Correlational  | 2.5     | 2.0     | 0.56       | Usually has some basis           |
| Safety         | 2.0     | 2.0     | 0.50       | Context-dependent                |
| Prescriptive   | 2.0     | 2.0     | 0.50       | Context-dependent                |
| Efficacy       | 2.0     | 2.5     | 0.44       | Often overstated                 |
| **Causal**     | 2.0     | 3.0     | 0.40       | **Frequently overstated**        |

### Bayesian Update Formula

Using Beta-Binomial conjugate prior:

```
Prior: Beta(α, β)
Evidence: n_support supporting, n_contradict contradicting

Posterior: Beta(α + Σw_support, β + Σw_contradict)

where w_i = source_weight × quality_score × recency_weight
```

### Source Reliability Weights

| Source Type       | Weight | Evidence Level |
| ----------------- | ------ | -------------- |
| Systematic Review | 3.0    | Level I        |
| RCT               | 2.5    | Level II       |
| Government Agency | 2.5    | Authority      |
| Cohort Study      | 2.0    | Level III      |
| Peer-Reviewed     | 1.8    | Level IV       |
| Case-Control      | 1.5    | Level IV       |
| Expert Opinion    | 1.2    | Level V        |
| Case Series       | 1.0    | Level V        |
| Preprint          | 0.8    | Unreviewed     |
| News Outlet       | 0.4    | Secondary      |
| Social Media      | 0.1    | Unreliable     |

### Handling Contradictory Evidence

When evidence conflicts:

1. **Quality-weighted comparison**: Higher quality sources dominate
2. **Pattern analysis**: Check if contradiction is from lower-tier sources
3. **Resolution confidence**: Computed as normalized strength difference
4. **Recommendation**: Auto-resolve if confidence > 0.3, else flag for manual review

```python
resolution_confidence = |support_strength - contradict_strength| / total_strength
```

---

## 2. Meta-Analysis Integration

### Random-Effects Model (DerSimonian-Laird)

When heterogeneity exists between studies:

```
θ_pooled = Σ(w*_i × θ_i) / Σw*_i

where w*_i = 1 / (v_i + τ²)
      v_i = within-study variance
      τ² = between-study variance (estimated)
```

### Heterogeneity Assessment

**Cochran's Q Statistic:**

```
Q = Σ w_i × (θ_i - θ_pooled)²
```

**I² Interpretation:**

| I² Range | Interpretation | Recommendation                     |
| -------- | -------------- | ---------------------------------- |
| 0-25%    | Low            | Fixed-effects appropriate          |
| 25-50%   | Moderate       | Consider random-effects            |
| 50-75%   | Substantial    | Use random-effects; investigate    |
| 75-100%  | Considerable   | Investigate sources before pooling |

### Quality-Based Weighting

Evidence pyramid multipliers:

```python
QUALITY_MULTIPLIERS = {
    "systematic_review": 1.00,
    "rct": 0.85,
    "government_agency": 0.80,
    "cohort_study": 0.70,
    "peer_reviewed": 0.65,
    "case_control": 0.55,
    "expert_opinion": 0.50,
    "case_series": 0.40,
    "preprint": 0.35,
    "news_outlet": 0.15,
    "social_media": 0.05,
}
```

---

## 3. Uncertainty Quantification

### Epistemic vs Aleatoric Uncertainty

**Epistemic Uncertainty** (reducible):

- Lack of evidence
- Poor quality sources
- Conflicting evidence
- _Can be reduced with more/better data_

**Aleatoric Uncertainty** (irreducible):

- Natural variation in study results
- Inherent variability in populations
- Measurement noise
- _Cannot be reduced with more data_

### Decomposition Formula

```
Total Uncertainty = CI_width / 2

Epistemic factors:
  - Scarcity: 1 / (1 + log(n_evidence))
  - Quality variance: var(quality_scores)
  - Conflict ratio: min(supports, contradicts) / total

Aleatoric = I² / 100 × total_uncertainty (from heterogeneity)
Epistemic = (1 - I²/100) × total_uncertainty
```

### Calibration Curves

Using Platt scaling for calibration:

```
P_calibrated = sigmoid(A × P_raw + B)
```

**Expected Calibration Error (ECE):**

```
ECE = Σ (n_bin/N) × |accuracy_bin - confidence_bin|
```

Target: ECE < 0.05 (well-calibrated)

---

## 4. Causal Inference for Claims

### Bradford Hill Criteria Scoring

| Criterion           | Weight | Assessment Method            |
| ------------------- | ------ | ---------------------------- |
| Strength            | 0.15   | Effect size magnitude        |
| Consistency         | 0.15   | Agreement across studies     |
| Temporality         | 0.15   | RCT/cohort design prevalence |
| Plausibility        | 0.10   | Peer-reviewed source ratio   |
| Biological Gradient | 0.10   | Dose-response keywords       |
| Coherence           | 0.10   | Average quality score        |
| Experiment          | 0.10   | RCT evidence presence        |
| Specificity         | 0.10   | Specificity keywords         |
| Analogy             | 0.05   | Default (hard to automate)   |

**Interpretation:**

- Score ≥ 0.75: Strong causal evidence
- Score 0.55-0.74: Moderate causal evidence
- Score 0.35-0.54: Weak causal evidence (likely correlational)
- Score < 0.35: Insufficient for causal claim

### Confounder Identification

Common confounders checked:

**Demographics**: age, sex, gender, race, ethnicity
**Socioeconomic**: income, education, occupation, insurance
**Lifestyle**: smoking, alcohol, diet, exercise, BMI, obesity
**Medical**: comorbidities, medications, medical history, genetics
**Environmental**: pollution, urban/rural, climate, housing
**Temporal**: season, year, time of day, duration

### Correlation vs Causation Detection

**Causal Language Patterns:**

- "causes", "leads to", "results in", "due to"
- "prevents", "reduces risk", "increases risk"

**Correlational Language Patterns:**

- "associated with", "linked to", "correlated with"
- "relationship between", "related to"

**Overclaim Detection:**
If claim uses causal language but evidence is only observational → Flag as high risk

---

## Usage Examples

### Basic Scoring

```python
from src.services.confidence_integration import create_confidence_scorer

scorer = create_confidence_scorer()

result = scorer.score_verification_result(
    claim_text="Vitamin D supplementation reduces respiratory infections by 30%",
    supporting_sources=[
        {
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            "title": "Systematic Review of Vitamin D and Infections",
            "snippet": "Meta-analysis of 25 RCTs showed..."
        },
        {
            "url": "https://www.cdc.gov/report",
            "title": "CDC Guidelines on Vitamin D"
        }
    ],
    contradicting_sources=[
        {
            "url": "https://news.example.com/vitamin-d-controversy",
            "title": "New study questions vitamin D benefits"
        }
    ]
)

print(f"Confidence: {result.final_confidence:.1f}% (Grade: {result.confidence_grade})")
print(f"Explanation: {result.explanation}")
```

### Batch Processing

```python
claims = [
    {"claim_text": "Claim 1", "supporting_sources": [...], "contradicting_sources": [...]},
    {"claim_text": "Claim 2", "supporting_sources": [...], "contradicting_sources": [...]},
]

results = scorer.batch_score(claims)

# Document-level summary
summary = scorer.get_overall_document_confidence(results)
print(f"Document Confidence: {summary['overall_confidence']:.1f}%")
print(f"High-confidence claims: {summary['high_confidence_claims']}")
```

### Report Integration

```python
# Format for report
formatted = scorer.format_for_report(result)

# Returns structured dictionary:
{
    "claim": "...",
    "confidence": {
        "score": 72.5,
        "grade": "B",
        "posterior_probability": 0.725,
        "credible_interval": [0.58, 0.85]
    },
    "uncertainty": {
        "epistemic": 0.08,
        "aleatoric": 0.05,
        "total": 0.13
    },
    "evidence_summary": {
        "count": 3,
        "source_diversity": 0.42
    },
    "explanation": "Based on 3 sources. Strong agreement across sources."
}
```

---

## Confidence Grade Interpretation

| Grade | Score Range | Interpretation                                                        |
| ----- | ----------- | --------------------------------------------------------------------- |
| **A** | 85-100%     | Strong confidence. High-quality evidence consistently supports claim. |
| **B** | 70-84%      | Good confidence. Reasonable evidence with minor uncertainties.        |
| **C** | 50-69%      | Moderate confidence. Mixed evidence or quality concerns.              |
| **D** | 30-49%      | Low confidence. Weak or conflicting evidence.                         |
| **F** | 0-29%       | Very low confidence. Insufficient or contradicting evidence.          |

---

## Key Design Decisions

1. **Conservative by Default**: Causal claims without experimental evidence are penalized
2. **Quality Over Quantity**: One systematic review outweighs multiple news articles
3. **Recency Matters**: 5-year half-life for evidence weighting
4. **Transparency**: Every score includes human-readable explanation
5. **Calibration-Ready**: Framework supports Platt scaling when validation data available

---

## Future Enhancements

1. **Active Learning Calibration**: Learn from expert corrections
2. **Citation Network Analysis**: Weight by citation influence
3. **Semantic Similarity**: Compare claim to evidence text embeddings
4. **Temporal Trends**: Detect if scientific consensus is shifting
5. **Domain-Specific Priors**: Customize priors for specific health domains
