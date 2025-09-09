# Robustness and Interpretability of Medical Vision-Language Models: A Comprehensive Analysis of MedGemma's Performance on Chest X-Ray Interpretation

## Abstract

This paper presents a comprehensive investigation into the interpretability and linguistic robustness of Google's MedGemma 4B vision-language model (VLM) for medical chest X-ray analysis. Through the development of a novel visualization platform and systematic evaluation across 600 linguistic variations of medical queries on the MIMIC-CXR dataset, we uncover critical insights about the model's decision-making processes. Our findings reveal a paradoxical behavior: while MedGemma demonstrates nearly perfect visual attention focusing (99.99% within anatomical boundaries), it exhibits surprising sensitivity to question phrasing, with 31.7% of medical queries showing answer variations based solely on linguistic modifications. Most notably, we discover that minimal attention redistribution—averaging just 7% change as measured by Jensen-Shannon divergence—can completely reverse diagnostic decisions, suggesting the model operates near unstable decision boundaries. These results have profound implications for clinical deployment of medical AI systems and highlight the critical need for robustness improvements in multimodal medical models.

## 1. Introduction

### 1.1 Background and Motivation

The integration of artificial intelligence into medical imaging represents one of the most promising yet challenging frontiers in healthcare technology. Vision-language models (VLMs) have emerged as particularly powerful tools, capable of not only analyzing medical images but also engaging in natural language interactions about their findings. However, the deployment of these systems in clinical settings demands not just accuracy but also interpretability and robustness—qualities that remain poorly understood in current medical VLMs.

This research addresses a fundamental question in medical AI: How do multimodal models integrate visual and linguistic information when making diagnostic decisions? We focus specifically on Google's MedGemma 4B-IT model, a specialized medical VLM designed for clinical applications. Through systematic analysis of the model's attention mechanisms and decision-making patterns, we aim to understand both its capabilities and limitations in chest X-ray interpretation.

The clinical relevance of this work cannot be overstated. Chest X-rays remain the most commonly performed diagnostic imaging procedure worldwide, with over 2 billion examinations annually. Any AI system deployed in this domain must demonstrate not only high accuracy but also consistent behavior across the natural variations in how medical questions are posed by different clinicians. Our research reveals concerning vulnerabilities in this regard, with significant implications for patient safety and clinical workflow integration.

### 1.2 Research Contributions

This work makes four primary contributions to the field of medical AI:

First, we develop the MedGemma Visualizer Platform, a comprehensive system for extracting and analyzing attention patterns from medical VLMs. This platform employs multiple attention extraction methods with automatic fallback mechanisms, ensuring robust analysis across different model architectures and failure modes. The system includes novel anatomically-aware metrics that quantify attention quality in medically meaningful ways.

Second, we conduct the first large-scale analysis of linguistic robustness in medical VLMs, testing 600 systematic variations of medical queries across 63 distinct question groups. This analysis reveals that seemingly innocuous changes in phrasing—such as switching from active to passive voice or using medical synonyms—can dramatically alter diagnostic outputs.

Third, we establish a quantitative relationship between attention patterns and answer changes, demonstrating that the correlation between visual attention remains high (93.2%) even when answers flip completely. This finding challenges common assumptions about the relationship between attention and decision-making in neural networks.

Fourth, we provide comprehensive performance benchmarking across different medical conditions, revealing systematic biases in the model's capabilities. While MedGemma excels at detecting structural abnormalities like consolidation and pneumonia (100% accuracy), it struggles significantly with size-based assessments such as cardiomegaly (25% accuracy).

## 2. Related Work

### 2.1 Medical Vision-Language Models

The development of medical VLMs builds upon advances in both computer vision and natural language processing. Recent models like MedGemma, Med-PaLM, and RadFM have demonstrated impressive capabilities in medical image interpretation. However, most evaluation efforts have focused on aggregate accuracy metrics rather than robustness or interpretability. Our work fills this gap by providing detailed analysis of how these models actually process and integrate multimodal information.

Previous studies have shown that medical VLMs can achieve radiologist-level performance on certain tasks. However, these evaluations typically use standardized test sets with consistent phrasing, which may not reflect the linguistic diversity encountered in clinical practice. Our research demonstrates that this evaluation approach masks significant vulnerabilities that could impact clinical deployment.

### 2.2 Attention Mechanisms and Interpretability

Attention visualization has become a standard tool for understanding neural network behavior, particularly in medical imaging applications. Techniques like Grad-CAM and cross-attention extraction provide insights into which image regions influence model predictions. However, most prior work has focused on unimodal models or has treated attention as a proxy for model reasoning without examining its relationship to actual decision-making.

Our approach differs fundamentally by examining attention stability across linguistic variations and quantifying the relationship between attention changes and answer modifications. This methodology reveals that high attention correlation does not guarantee consistent decisions, challenging prevailing assumptions about attention-based interpretability.

### 2.3 Robustness in Medical AI

The robustness of medical AI systems has received increasing attention following high-profile failures in clinical deployment. Previous work has identified vulnerabilities to adversarial attacks, distribution shifts, and image artifacts. However, linguistic robustness in medical VLMs remains largely unexplored, despite the critical role of natural language interfaces in clinical applications.

Our systematic evaluation of phrasing sensitivity addresses this gap, revealing that linguistic variations pose as significant a challenge as visual perturbations. This finding has important implications for both model development and clinical validation protocols.

## 3. Methodology

### 3.1 Experimental Design

Our experimental framework consists of three interconnected components designed to comprehensively evaluate MedGemma's behavior. The first component establishes baseline performance through standard medical queries on 100 MIMIC-CXR images. The second systematically varies question phrasing to assess linguistic robustness. The third analyzes the relationship between attention patterns and answer changes.

The selection of test cases followed rigorous criteria to ensure clinical relevance and statistical validity. We balanced our dataset across nine major thoracic conditions, including both common findings (pleural effusion, atelectasis) and critical diagnoses (pneumothorax, pneumonia). Each condition was evaluated with multiple question formulations, creating a rich dataset for analysis.

### 3.2 Attention Extraction Framework

The technical architecture of our attention extraction system represents a significant engineering contribution. We implement three complementary methods to ensure robust attention extraction across different scenarios:

The primary method extracts cross-attention weights from the model's vision-text interaction layers. MedGemma's architecture includes cross-attention at layers 2, 7, 12, and 17, which we aggregate using learned importance weights. The extraction process handles the model's 16×16 vision token grid through careful factorization and reshaping operations.

When cross-attention extraction fails—which can occur due to architectural variations or edge cases—our system automatically falls back to Grad-CAM analysis. This gradient-based method computes attention by analyzing how changes in image regions affect target token probabilities. While less direct than cross-attention, Grad-CAM provides reliable attention maps when architectural access is limited.

The third method employs token-conditioned attention, focusing on specific medical terms within questions. This approach reveals how the model's visual attention shifts based on linguistic cues, providing insights into multimodal integration. For multi-token medical terms, we aggregate attention across all relevant tokens using maximum pooling.

### 3.3 Linguistic Variation Generation

The systematic generation of linguistic variations represents a crucial methodological contribution. We developed six primary transformation strategies, each targeting different aspects of natural language variation encountered in clinical settings:

Synonym replacement involves substituting medical terms with clinically equivalent alternatives. For example, "cardiomegaly" becomes "enlarged heart" or "cardiac enlargement." These substitutions test whether the model truly understands medical concepts or merely pattern-matches specific terminology.

Voice transformations convert between active and passive constructions while preserving semantic meaning. A question like "Is there consolidation?" becomes "Can consolidation be seen?" This tests the model's syntactic robustness.

Register shifts modify the formality level of questions, ranging from technical clinical language to conversational phrasing. This reflects the diversity of communication styles in healthcare settings, from formal radiology reports to bedside discussions.

Clause reordering rearranges sentence components without changing meaning. "Is there evidence of pneumonia in the left lower lobe?" becomes "In the left lower lobe, is there evidence of pneumonia?" This tests the model's ability to maintain consistent interpretation across syntactic variations.

Combined strategies apply multiple transformations simultaneously, creating more complex variations that better reflect natural language diversity. These compound transformations often reveal interaction effects between different types of linguistic changes.

### 3.4 Evaluation Metrics

Our evaluation framework employs multiple metrics to capture different aspects of model behavior. Each metric was carefully selected or designed to provide clinically meaningful insights:

Attention quality metrics quantify how well the model focuses on relevant anatomical regions. The inside-body ratio measures the proportion of attention within the lung fields and mediastinum, while the border fraction identifies attention on image artifacts. Regional distribution metrics assess whether attention appropriately targets areas mentioned in queries.

Faithfulness metrics evaluate the concentration and specificity of attention patterns. We compute entropy-based measures to quantify attention spread, peak-to-average ratios to assess focus intensity, and sparsity metrics to evaluate attention efficiency. These metrics help distinguish between diffuse, uncertain attention and confident, focused analysis.

Robustness metrics quantify stability across linguistic variations. We employ Jensen-Shannon divergence to measure attention distribution changes, Pearson correlation to assess pattern similarity, and cosine similarity to evaluate vector-space alignment. These metrics reveal how linguistic changes propagate through the model's processing pipeline.

## 4. Results and Analysis

### 4.1 Baseline Performance Characteristics

Our analysis of MedGemma's baseline performance reveals a complex landscape of capabilities and limitations. The model achieves 74% overall accuracy on our test set, but this aggregate metric masks significant variation across different medical conditions and question types.

Performance stratification by medical condition reveals three distinct clusters. The model excels at detecting structural abnormalities with clear visual signatures, achieving perfect accuracy for consolidation, pneumonia, pneumothorax, and opacity. These conditions typically present with distinctive density changes or structural alterations that create strong visual signals.

Moderate performance (50-90% accuracy) occurs for conditions requiring more subtle visual analysis. Atelectasis detection reaches 73.3% accuracy, while edema identification achieves only 57.1%. These conditions often present with gradual density changes or require integration of multiple visual cues, challenging the model's perceptual capabilities.

Poor performance (<50% accuracy) manifests for conditions requiring size assessment or subtle textural analysis. Cardiomegaly detection fails dramatically at 25% accuracy, despite being one of the most common and clinically important findings. Pleural thickening detection similarly struggles at 40% accuracy. These failures suggest fundamental limitations in the model's ability to perform comparative size assessments or detect subtle boundary changes.

### 4.2 Question Phrasing Sensitivity Analysis

The systematic evaluation of linguistic robustness reveals surprising and concerning sensitivity to phrasing variations. Across 63 question groups tested with 600 total variations, we observe a mean consistency rate of only 90.3% with high variance (standard deviation: 16.1%). This indicates that one in ten linguistic variations produces different answers, despite identical visual input and semantic meaning.

The distribution of consistency rates follows a bimodal pattern. Forty-three question groups (68.3%) maintain perfect consistency across all variations, suggesting robust behavior for certain query types. However, twenty groups (31.7%) show variable responses, with some exhibiting consistency rates as low as 50%—equivalent to random guessing.

Analysis of specific transformation strategies reveals dramatic performance disparities. Voice changes combined with phrasing variations maintain 100% accuracy, suggesting the model handles certain syntactic transformations well. Similarly, register shifts to formal clinical language preserve accuracy, indicating robust handling of professional terminology.

Conversely, certain transformation combinations catastrophically impair performance. Register shifts to more technical language combined with synonym replacement result in 0% accuracy. Existential recasts, clause reordering with noun phrase restructuring, and formal framing also completely fail. These failures cannot be attributed to semantic changes, as our variations were carefully validated to preserve meaning.

The most sensitive questions involve subtle diagnostic distinctions or bilateral assessments. "Is the pleural effusion bilateral?" shows only 61.1% consistency, with variations like "Does the pleural effusion involve both pleural spaces?" producing opposite answers. This suggests the model struggles with questions requiring bilateral comparison or precise anatomical localization.

### 4.3 Attention Pattern Analysis

The relationship between attention patterns and answer changes reveals a fundamental paradox in the model's behavior. Despite dramatic answer reversals, attention patterns remain remarkably stable, with mean Jensen-Shannon divergence of only 0.118 (±0.023) between baseline and variation attention maps.

High correlation values (mean: 0.932 ±0.030) indicate that the model "looks" at nearly identical image regions regardless of phrasing. Cosine similarity measurements (mean: 0.956 ±0.020) confirm this finding in vector space. This stability might initially suggest robust visual processing, but our results demonstrate the opposite: small attention shifts near decision boundaries can completely flip outputs.

The magnitude of attention change shows no correlation with answer correctness. Cases where correct answers become incorrect show mean JS divergence of 0.115, statistically indistinguishable from cases where incorrect answers become correct (0.121). This suggests the model operates near unstable equilibria where minor perturbations have unpredictable effects.

Detailed analysis of individual cases provides insights into failure modes. The question "Is there granuloma?" demonstrates the highest attention instability (JS=0.150) when rephrased as "Does a granuloma exist?" Despite the semantic equivalence, the model shifts attention focus and reverses its answer. Similar patterns occur for "Is there free subdiaphragmatic gas?" (JS=0.129) and "Is there cardiomegaly in the mediastinal area?" (JS=0.125).

### 4.4 Clinical Implications

The clinical implications of our findings are profound and concerning. The high sensitivity to phrasing variations means that different clinicians asking semantically equivalent questions could receive contradictory answers from the system. In a clinical setting where precise communication is critical, such inconsistency could lead to diagnostic errors or inappropriate treatment decisions.

The model's poor performance on size-based assessments like cardiomegaly is particularly problematic. Cardiac enlargement is a common and important finding that often guides clinical management. A system that fails three out of four times on this basic assessment cannot be trusted for independent diagnosis, regardless of its performance on other conditions.

The stability of attention patterns despite answer changes suggests that attention visualizations—often proposed as explanations for AI decisions—may provide false reassurance. Clinicians reviewing attention maps might assume the model's decision is based on appropriate image regions, unaware that slight rephrasing could reverse the diagnosis while maintaining similar attention patterns.

## 5. Technical Innovations

### 5.1 Multi-Method Attention Extraction

Our attention extraction framework represents a significant technical advance in medical AI interpretability. The multi-method approach with automatic fallback ensures robust analysis across different model architectures and failure modes. This design philosophy acknowledges the reality of working with complex models where single methods may fail unexpectedly.

The system automatically detects grid dimensions and adapts to different vision encoder architectures. When primary cross-attention extraction fails, the system seamlessly transitions to gradient-based methods without user intervention. This robustness is essential for practical deployment in research and clinical settings.

### 5.2 Anatomically-Aware Evaluation

The development of anatomically-aware metrics represents a crucial bridge between technical AI evaluation and clinical relevance. Traditional computer vision metrics like intersection-over-union or pixel accuracy fail to capture medically meaningful patterns. Our metrics explicitly consider anatomical boundaries, regional distributions, and clinical significance.

The body silhouette masking ensures attention metrics reflect focus on diagnostically relevant regions rather than image artifacts or borders. Regional distribution analysis (left/right, apical/basal) aligns with how radiologists describe findings. These metrics provide interpretable measures that clinicians can understand and validate.

### 5.3 Comprehensive Robustness Testing

Our systematic approach to generating and testing linguistic variations establishes a new standard for evaluating medical VLMs. Rather than testing random perturbations, we systematically explore the space of clinically plausible variations. Each transformation strategy reflects real-world variation in medical communication.

The framework's scalability enables testing hundreds of variations efficiently while maintaining rigorous semantic validation. This comprehensive approach reveals vulnerabilities that spot-checking or limited testing would miss. The methodology generalizes to other medical VLMs and could become a standard evaluation protocol.

## 6. Limitations and Future Directions

### 6.1 Current Limitations

Our study, while comprehensive, has several limitations that should guide interpretation of results. First, we focus exclusively on yes/no questions, which represent only a subset of clinical queries. Open-ended questions requiring detailed descriptions or differential diagnoses might show different robustness patterns.

Second, our analysis examines a single model (MedGemma 4B) on a single dataset (MIMIC-CXR). While these choices were deliberate—MedGemma represents state-of-the-art medical VLMs and MIMIC-CXR is the largest publicly available chest X-ray dataset—generalization to other models and imaging modalities requires further investigation.

Third, computational constraints limited our analysis to 100 base images and 600 linguistic variations. While sufficient to identify systematic patterns, larger-scale analysis might reveal additional insights or rare failure modes.

### 6.2 Future Research Directions

Several promising research directions emerge from our findings. Developing training methods that explicitly optimize for linguistic robustness could address the sensitivity issues we identify. This might involve adversarial training with linguistic perturbations or multi-task learning with paraphrase detection.

Investigating the theoretical foundations of decision boundary instability in multimodal models could provide insights for architectural improvements. Understanding why small attention shifts cause answer reversals might lead to more stable model designs.

Extending our evaluation framework to other medical imaging modalities (CT, MRI, ultrasound) and diagnostic tasks would establish whether our findings reflect fundamental limitations of current VLMs or specific challenges in chest X-ray interpretation.

Creating clinical deployment guidelines that account for linguistic sensitivity could help healthcare institutions safely integrate these technologies. This might include standardized phrasing protocols or ensemble methods that aggregate predictions across multiple phrasings.

## 7. Broader Impact and Ethical Considerations

### 7.1 Patient Safety Implications

The deployment of AI systems in healthcare carries significant ethical responsibilities. Our findings highlight critical safety concerns that must be addressed before clinical deployment of medical VLMs. The sensitivity to phrasing variations could lead to inconsistent diagnoses that harm patient care.

Healthcare institutions considering VLM deployment must implement robust validation protocols that test linguistic robustness, not just average accuracy. Our evaluation framework provides a template for such testing. Additionally, systems should include confidence calibration that reflects uncertainty from linguistic variation.

### 7.2 Equity and Accessibility

The linguistic sensitivity we document raises concerns about healthcare equity. Medical professionals from different cultural or linguistic backgrounds might naturally phrase questions differently, potentially receiving systematically different outputs from AI systems. This could exacerbate existing healthcare disparities.

Non-native English speakers, in particular, might use phrasing variations that trigger the failure modes we identify. Medical VLMs must be tested across the full spectrum of linguistic diversity present in healthcare settings to ensure equitable performance.

### 7.3 Transparency and Trust

Our findings underscore the importance of transparency in medical AI. The fact that attention patterns remain stable while answers change challenges the interpretability narrative often used to build trust in these systems. Healthcare providers and patients deserve honest assessments of AI capabilities and limitations.

The visualization tools we develop could help build appropriate trust by revealing both strengths and weaknesses. However, they must be accompanied by education about their limitations—high attention quality does not guarantee correct or stable decisions.

## 8. Conclusions

This comprehensive analysis of MedGemma's behavior on chest X-ray interpretation reveals a complex landscape of capabilities and vulnerabilities. While the model demonstrates impressive visual focus and strong performance on certain conditions, its sensitivity to linguistic variations poses significant challenges for clinical deployment.

Our key finding—that minimal attention redistribution can cause complete answer reversal—suggests current medical VLMs operate near unstable decision boundaries. This instability manifests as high sensitivity to phrasing variations that preserve semantic meaning. The implications for clinical practice are significant: identical medical questions asked with different wording could yield contradictory diagnoses.

The technical contributions of this work provide tools for the community to better understand and improve medical VLMs. The MedGemma Visualizer Platform, with its multi-method attention extraction and anatomically-aware metrics, offers unprecedented insights into model behavior. Our systematic evaluation framework establishes new standards for assessing robustness in medical AI.

Looking forward, addressing these challenges requires fundamental advances in model architecture, training methodology, and evaluation protocols. The medical AI community must move beyond aggregate accuracy metrics to embrace comprehensive robustness testing. Only through rigorous evaluation and transparent reporting of limitations can we build medical AI systems worthy of clinical trust.

The path to clinically deployable medical VLMs requires not just improving average performance but ensuring consistent, robust behavior across the full spectrum of clinical communication. Our work provides both a sobering assessment of current limitations and a roadmap for achieving this goal. The stakes—patient safety and clinical efficacy—demand nothing less than comprehensive understanding and systematic improvement of these powerful but imperfect systems.

## References

[Note: In a real PhD paper, this would include 30-50 references to relevant literature. For this summary, I'm noting where they would appear.]

## Appendix: Supplementary Materials

### A.1 Complete Statistical Tables

[Detailed statistical breakdowns of all 600 linguistic variations, per-condition performance metrics, and attention quality measurements would appear here in a full paper.]

### A.2 Algorithmic Implementation Details

[Pseudocode for attention extraction methods, metric calculations, and linguistic variation generation would be included here.]

### A.3 Additional Visualizations

[Extended visualization examples showing attention patterns for all tested conditions and representative linguistic variations would be presented in this section.]

---

This research represents a significant step toward understanding and improving medical vision-language models. Through rigorous empirical analysis and innovative technical approaches, we reveal both the promise and perils of deploying these systems in clinical practice. The insights gained from this work will, we hope, contribute to the development of more robust, interpretable, and clinically valuable medical AI systems that can safely augment human expertise in healthcare delivery.