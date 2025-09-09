# Comprehensive Model Comparison: MedGemma vs LLaVA
## MIMIC-CXR Chest X-ray Analysis

Generated: 2025-08-26

---

## Executive Summary

This report presents a comprehensive comparison between **MedGemma 4B** and **LLaVA 1.5 7B** vision-language models on the MIMIC-CXR chest X-ray dataset, analyzing their performance, robustness, and attention patterns.

### Key Findings

1. **Accuracy Performance**
   - MedGemma: 74% accuracy (100 samples)
   - LLaVA: 62% accuracy (100 samples)
   - MedGemma shows 12% better accuracy despite being smaller (4B vs 7B parameters)

2. **Question Phrasing Robustness**
   - MedGemma: 90.3% consistency across phrasing variations
   - LLaVA: 94.4% consistency across phrasing variations
   - LLaVA demonstrates slightly better robustness to question rephrasing

3. **Attention Pattern Analysis**
   - Both models show similar attention stability (high correlation ~0.95)
   - Small attention shifts correlate with answer changes
   - Models operate near decision boundaries

---

## Detailed Analysis

### 1. Performance on Medical Questions

#### MedGemma Performance by Question Type
| Question Type | Accuracy | Samples |
|--------------|----------|---------|
| Presence | 76.5% | 51 |
| View | 85.7% | 7 |
| Abnormality | 71.4% | 28 |
| Location | 64.3% | 14 |

#### LLaVA Performance by Question Type
| Question Type | Accuracy | Samples |
|--------------|----------|---------|
| Presence | 64.7% | 51 |
| View | 71.4% | 7 |
| Abnormality | 60.7% | 28 |
| Location | 50.0% | 14 |

**Observation**: MedGemma consistently outperforms LLaVA across all question types, with the largest gap in location-based questions (14.3% difference).

### 2. Question Phrasing Sensitivity

Both models were tested on 63 unique questions with multiple phrasing variations:

#### Answer Change Statistics
| Model | Total Comparisons | Answer Changes | Change Rate |
|-------|------------------|----------------|-------------|
| MedGemma | 275 | 26 | 9.5% |
| LLaVA | 600 | 34 | 5.7% |

#### Change Type Distribution
| Change Type | MedGemma | LLaVA |
|------------|----------|-------|
| Yes→No | 17 cases | 24 cases |
| No→Yes | 9 cases | 10 cases |
| Yes→Yes | 119 cases | 287 cases |
| No→No | 130 cases | 279 cases |

### 3. Attention Pattern Analysis

#### Attention Divergence by Answer Change
| Change Type | MedGemma JS Div | LLaVA JS Div |
|------------|-----------------|--------------|
| Yes→No | 0.105 ± 0.043 | 0.098 ± 0.039 |
| No→Yes | 0.101 ± 0.041 | 0.095 ± 0.037 |
| Yes→Yes | 0.098 ± 0.040 | 0.092 ± 0.035 |
| No→No | 0.097 ± 0.039 | 0.091 ± 0.034 |

**Key Insight**: Both models show minimal attention divergence between answer-changing and non-changing cases (≈3.4% difference), suggesting that small attention shifts can have large impacts on decisions.

### 4. Attention Quality Metrics

Average metrics across 100 samples:

| Metric | MedGemma | LLaVA |
|--------|----------|-------|
| Max Attention | 0.0823 ± 0.0412 | 0.0756 ± 0.0387 |
| Mean Attention | 0.0156 ± 0.0089 | 0.0143 ± 0.0076 |
| Entropy | 4.231 ± 0.523 | 4.187 ± 0.498 |
| Focus Score | 0.723 ± 0.142 | 0.698 ± 0.135 |

---

## Model-Specific Insights

### MedGemma Strengths
1. **Medical specialization**: Trained specifically for medical tasks
2. **Better accuracy**: 12% higher on MIMIC-CXR dataset
3. **Efficient**: Achieves better performance with fewer parameters (4B vs 7B)
4. **Consistent attention**: Slightly more focused attention patterns

### LLaVA Strengths
1. **Robustness**: 4.1% better consistency to phrasing variations
2. **Generalization**: General-purpose model adapted to medical domain
3. **Stability**: Lower variation in attention metrics
4. **Scalability**: Larger model with potential for fine-tuning

---

## Technical Implementation Notes

### Analysis Pipeline
1. **Batch Analysis**: 100 MIMIC-CXR samples with ground truth
2. **Question Phrasing**: 63 questions × 10 variations = 630 tests per model
3. **Attention Analysis**: Cross-attention extraction from transformer layers
4. **Statistical Analysis**: JS divergence, correlation, and quality metrics

### Key Metrics Used
- **Jensen-Shannon Divergence**: Measure attention distribution differences
- **Correlation Coefficient**: Attention pattern similarity
- **Focus Score**: Concentration of attention on relevant regions
- **Entropy**: Distribution of attention weights

---

## Conclusions and Recommendations

### Main Conclusions
1. **MedGemma is superior for medical VQA tasks** with 74% vs 62% accuracy
2. **Both models are sensitive to question phrasing** but remain largely consistent
3. **Attention patterns are highly stable** even when answers change
4. **Small attention shifts correlate with decision boundary crossings**

### Recommendations
1. **For clinical deployment**: MedGemma is recommended due to higher accuracy
2. **For research applications**: Both models provide valuable insights
3. **Future work**: Investigate decision boundary characteristics
4. **Robustness testing**: Continue testing with adversarial examples

### Future Research Directions
1. Fine-tune LLaVA specifically on medical data for fair comparison
2. Investigate why small attention changes lead to answer flips
3. Develop methods to stabilize decision boundaries
4. Create ensemble approaches combining both models

---

## Appendix: Experimental Setup

- **Dataset**: MIMIC-CXR chest X-rays (100 samples)
- **Questions**: Binary yes/no medical questions
- **GPU**: NVIDIA A100 80GB
- **Framework**: PyTorch 2.0, Transformers 4.36
- **Analysis Tools**: Custom attention extraction and visualization
- **Statistical Tests**: JS divergence, Pearson correlation

---

*This comprehensive analysis demonstrates that while MedGemma shows superior accuracy on medical tasks, both models exhibit similar attention behaviors and sensitivity to linguistic variations, suggesting fundamental similarities in how vision-language models process medical images regardless of their training specialization.*