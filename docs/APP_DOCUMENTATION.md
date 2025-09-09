# MedGemma 4B Multimodal VLM - Complete Technical Documentation

## System Architecture Overview

The MedGemma 4B Analysis Platform is a comprehensive tool for analyzing how Google's MedGemma 4B vision-language model processes chest X-rays from the MIMIC-CXR dataset. Developed by SAIL Lab at University of New Haven, it provides deep insights into model attention mechanisms and decision-making processes.

## Tab-by-Tab Technical Details

### Tab 1: MIMIC Question Analysis

**Purpose**: Standard medical question answering with attention visualization

**Technical Implementation**:

1. **Image Processing Pipeline**:
   - Load MIMIC-CXR images (DICOM → JPG conversion)
   - RGB conversion for model compatibility
   - Apply MedGemma-specific normalization:
     ```python
     mean = [0.48145466, 0.4578275, 0.40821073]
     std = [0.26862954, 0.26130258, 0.27577711]
     normalized = (image - mean) / std
     ```
   - Tokenize into vision patches (typically 256 or 196 tokens)

2. **Cross-Attention Extraction**:
   - During generation, capture cross-attention weights between text and image tokens
   - Extract from last 3-5 decoder layers (most relevant for answer)
   - Average across 32 attention heads
   - Focus on tokens corresponding to answer generation

3. **Visualization Pipeline**:
   - Generate tight body mask using Otsu thresholding
   - Apply morphological operations to clean mask
   - Strip border tokens (outer ring) to remove padding artifacts
   - Apply percentile clipping (2nd-98th) for better contrast
   - Overlay attention heatmap on original X-ray

4. **Metrics Computed**:
   - Regional focus (upper/lower, left/right quadrants)
   - Attention entropy (Shannon entropy of distribution)
   - Answer extraction and ground truth comparison

### Tab 2: Token-Conditioned Analysis

**Purpose**: Visualize attention specifically for medical terms (e.g., "pneumonia", "effusion")

**Technical Implementation**:

1. **Target Token Identification**:
   ```python
   def find_target_tokens(generated_ids, target_word):
       target_ids = tokenizer.encode(target_word)
       positions = []
       for i in range(len(generated_ids) - len(target_ids)):
           if generated_ids[i:i+len(target_ids)] == target_ids:
               positions.extend(range(i, i+len(target_ids)))
       return positions
   ```

2. **Multi-Method Attention Extraction**:

   **Method A - Cross-Attention (Primary)**:
   - Extract attention weights at target token positions
   - Average across relevant decoder layers
   - Provides direct model attention

   **Method B - Grad-CAM (Fallback)**:
   ```python
   # Enable gradients for vision components
   for name, param in model.named_parameters():
       if 'vision' in name:
           param.requires_grad = True
   
   # Forward pass with float32 precision
   outputs = model(inputs, pixel_values=images.float())
   
   # Compute gradient of target token w.r.t. vision features
   loss = outputs.logits[0, target_token_id]
   loss.backward()
   
   # Weight activations by gradients
   cam = (gradients * activations).sum(dim=channels).relu()
   ```

   **Method C - Simple Activation (Final Fallback)**:
   - Hook vision encoder layer (typically layer 23)
   - Extract activation magnitudes
   - Compute L2 norm across channels

3. **Quality Metrics**:
   - **Inside Body Ratio**: `attention_inside_mask / total_attention` (target ≥ 0.7)
   - **Border Fraction**: `border_attention / total_attention` (target ≤ 0.05)
   - **Left/Right Balance**: Distribution across lung fields
   - **Apical/Basal Split**: Upper vs lower lung attention

4. **Aspect-Aware Grid Reshaping**:
   ```python
   def factor_to_grid(n_tokens, H, W):
       aspect_ratio = W / H
       best_factors = None
       min_distortion = float('inf')
       
       for h in range(1, int(sqrt(n_tokens)) + 1):
           if n_tokens % h == 0:
               w = n_tokens // h
               distortion = abs((w/h) - aspect_ratio)
               if distortion < min_distortion:
                   best_factors = (h, w)
                   min_distortion = distortion
       
       return best_factors
   ```

### Tab 3: Prompt Sensitivity Analysis

**Purpose**: Analyze how different phrasings affect model behavior

**Technical Implementation**:

1. **Prompt Generation**:
   - Technical version: Medical terminology
   - Simple version: Patient-friendly language
   - Maintain semantic equivalence

2. **Attention Extraction**:
   - Generate with both prompts
   - Extract attention distributions
   - Ensure same image preprocessing

3. **Jensen-Shannon Divergence Computation**:
   ```python
   def compute_js_divergence(attn1, attn2):
       # Normalize to probability distributions
       p = attn1.flatten() / attn1.sum()
       q = attn2.flatten() / attn2.sum()
       
       # Compute JS divergence
       m = (p + q) / 2
       js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
       return js
   ```

4. **Interpretation Scale**:
   - JS ≈ 0.0: Identical attention
   - JS < 0.2: Minor variations
   - JS 0.2-0.5: Moderate differences
   - JS > 0.5: Significant divergence

5. **Visualization**:
   - Side-by-side attention heatmaps
   - Difference map (attn1 - attn2)
   - Answer consistency checking

## Core Technical Components

### Body-Aware Masking
```python
def tight_body_mask(gray_image):
    # Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray_image, (0, 0), 2)
    
    # Otsu's thresholding for binary mask
    _, mask = cv2.threshold(blurred, 0, 255, 
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert (body is usually darker)
    mask = 255 - mask
    
    # Morphological operations to clean
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, 
                           np.ones((15, 15), np.uint8))
    
    # Erode to remove border artifacts
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    
    return mask
```

### GPU Memory Management
- Automatic GPU selection based on available memory
- Requires minimum 15GB free memory
- Uses BFloat16 for model weights
- Converts to Float32 for computations
- Clears cache after each inference

### Error Handling Cascade
1. Primary: Cross-attention extraction
2. Fallback 1: Grad-CAM with gradient computation
3. Fallback 2: Simple activation norms
4. Final: Uniform attention with warning

### Numerical Stability
- BFloat16 → Float32 conversion for all computations
- Epsilon addition (1e-8) to prevent division by zero
- Percentile clipping for outlier handling
- Log-space computations for probabilities

## Quality Metrics Reference

| Metric | Target | Poor | Interpretation |
|--------|--------|------|----------------|
| Inside Body Ratio | ≥ 0.7 | < 0.5 | Attention focus on anatomy vs background |
| Border Fraction | ≤ 0.05 | > 0.15 | Artifacts from padding tokens |
| Attention Entropy | < 3.0 | > 4.5 | Focus vs dispersion |
| JS Divergence | < 0.2 | > 0.5 | Prompt sensitivity |
| Left/Right Balance | 0.3-0.7 | <0.2 or >0.8 | Laterality bias |

## Recent Improvements (2025-08-11)

1. **BFloat16 Support**: Full compatibility with BFloat16 models
2. **Metric Fixes**: Corrected inside_body_ratio calculation
3. **Aspect Preservation**: factor_to_grid for all reshaping
4. **Enhanced Fallbacks**: Multi-level error recovery
5. **User Warnings**: Clear indicators for degraded methods

## Platform Information

**Developed by**: SAIL Lab - University of New Haven
**Model**: Google MedGemma 4B-IT
**Dataset**: MIMIC-CXR
**Framework**: PyTorch + HuggingFace Transformers
**Version**: Fixed Implementation with BFloat16 Support

## Usage

```bash
# Standard launch
python medgemma_app.py

# With specific GPU
python medgemma_app.py --gpu 0

# Custom port
python medgemma_app.py --port 8080
```

## Disclaimer

This is a research platform for analyzing model behavior. Not intended for clinical use. Always consult qualified healthcare professionals for medical decisions.

---

*For technical support or research collaboration, contact SAIL Lab at University of New Haven*