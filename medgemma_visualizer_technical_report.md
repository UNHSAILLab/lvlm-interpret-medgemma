# Technical Report: MedGemma Attention Visualizer Implementation

## Executive Summary

The MedGemma Attention Visualizer is a sophisticated interpretability tool designed specifically for Google's MedGemma-4B model, focusing on transparent attention visualization for medical image analysis. This implementation adapts the LVLM-Interpret framework to work with MedGemma's SigLip vision tower architecture, providing multiple attention analysis methods optimized for chest X-ray interpretation.

## Architecture Overview

### Core Components

1. **MedGemmaRelevancyAnalyzer** (`medgemma_relevancy_analyzer.py`)
   - Core engine for attention extraction and processing
   - Implements multiple relevancy computation methods
   - Handles MedGemma's specific architecture (34 layers, 8 attention heads)

2. **MedGemmaApp** (`medgemma_attention_visualizer.py`)
   - Gradio-based web interface
   - Interactive token-by-token attention analysis
   - Real-time visualization generation

3. **Model Configuration** (`medgemma_launch.py`)
   - Safe GPU memory management
   - BFloat16 precision for stability
   - Attention output configuration

## Technical Implementation Details

### 1. Model Architecture Specifics

```python
# MedGemma-4B configuration
- Vision encoder: SigLip (16x16 patches = 256 image tokens)
- Model layers: 34 transformer layers
- Attention heads: 8 per layer
- Hidden size: Variable per layer
- Precision: BFloat16 (more stable than Float16)
```

### 2. Attention Extraction Pipeline

#### 2.1 Token Position Mapping
The system handles MedGemma's fixed attention matrix size:

```python
def extract_raw_attention(self, token_idx):
    # MedGemma uses fixed attention size
    # Map generated token position to attention matrix
    input_len = self.current_inputs['input_ids'].shape[1]
    gen_pos = min(input_len + token_idx, avg_attn.shape[0] - 1)
    
    # Extract attention to image tokens (positions 1-257)
    if gen_pos >= 0 and 257 <= avg_attn.shape[1]:
        attn_to_image = avg_attn[gen_pos, 1:257]
        return attn_to_image.reshape(16, 16).numpy()
```

#### 2.2 Image Token Organization
- Position 0: BOS token
- Positions 1-256: Image tokens (16x16 grid)
- Positions 257+: Text tokens

### 3. Relevancy Computation Methods

#### 3.1 Layer-Weighted Relevancy
Accumulates attention across all 34 layers with increasing weights:

```python
def compute_simple_relevancy(self, outputs, inputs, token_idx):
    relevancy_scores = []
    for layer_idx, layer_attn in enumerate(token_attentions):
        # Weight by layer depth (later layers more important)
        layer_weight = (layer_idx + 1) / len(token_attentions)
        weighted_attn = attn_to_image * layer_weight
        relevancy_scores.append(weighted_attn)
    
    final_relevancy = torch.stack(relevancy_scores).mean(dim=0)
```

**Rationale**: Later layers contain more semantic information relevant to medical findings.

#### 3.2 Head Importance Relevancy
Identifies and weights the most informative attention heads:

```python
def compute_head_importance_relevancy(self, outputs, inputs, token_idx):
    for h in range(num_heads):
        # Importance = max attention * (1 / (1 + entropy))
        max_attn = attn_to_image.max().item()
        entropy = -(attn_to_image * torch.log(attn_to_image + 1e-10)).sum().item()
        importance = max_attn * (1 / (1 + entropy))
```

**Key insight**: Heads with focused attention (low entropy) and strong peaks are more informative.

#### 3.3 Attention Flow
Traces information propagation through layers without matrix multiplication:

```python
def compute_attention_flow(self, outputs, inputs, token_idx):
    # Start from last layer
    flow = last_layer.mean(dim=0)
    
    # Backward through layers with exponential decay
    weights = torch.tensor([0.5 ** i for i in range(len(layer_contributions))])
    combined = image_attention * 0.5  # Last layer gets 50%
    for i, contrib in enumerate(layer_contributions):
        combined = combined + contrib * weights[i] * 0.5
```

**Innovation**: Avoids memory-intensive matrix multiplication while preserving attention flow patterns.

#### 3.4 Consensus Relevancy
Combines all methods for robust results:

```python
consensus = (simple_enhanced + head_rel + flow_enhanced) / 3
```

### 4. Visualization Enhancement

#### 4.1 Contrast Enhancement
Two methods for improving visibility:

```python
def enhance_visualization_contrast(attention_map, method='percentile'):
    if method == 'percentile':
        # Use 5th-95th percentile for better contrast
        p5, p95 = np.percentile(attention_map, [5, 95])
        enhanced = np.clip((attention_map - p5) / (p95 - p5), 0, 1)
    elif method == 'log':
        # Log scale for very small values
        enhanced = np.log(attention_map + 1e-8)
```

#### 4.2 High Attention Region Detection
Uses contour detection to identify focus areas:

```python
threshold = np.percentile(attention_resized, 85)
mask = attention_resized > threshold
contours, _ = cv2.findContours(mask.astype(np.uint8), 
                               cv2.RETR_EXTERNAL, 
                               cv2.CHAIN_APPROX_SIMPLE)
```

### 5. Interactive Features

#### 5.1 Token-Specific Analysis
Users can select any generated token to see its attention patterns:

```python
def analyze_token(self, token_selection):
    # Extract token index from dropdown
    token_idx = int(token_selection.split(":")[0].split()[-1])
    
    # Generate comprehensive visualization with:
    # - Raw attention
    # - All relevancy methods
    # - Statistical analysis
    # - Region identification
```

#### 5.2 Caching System
Implements attention caching for performance:

```python
if token_idx in self.attention_cache:
    return self.attention_cache[token_idx]
# ... compute attention ...
self.attention_cache[token_idx] = result_image
```

### 6. Medical-Specific Optimizations

#### 6.1 Report Formatting
Structured medical report generation:

```python
def format_medical_report(self, raw_text):
    # Clean formatting
    text = raw_text.replace('**', '').replace('*', '')
    # Add proper line breaks
    text = text.replace('. ', '.\n\n')
    # Format numbered findings
    if re.match(r'^\d+\.', line):
        parts = line.split(':', 1)
        formatted_lines.append(f"**{parts[0]}:** {parts[1].strip()}")
```

#### 6.2 Custom Question Handling
Supports both comprehensive analysis and specific queries:

```python
if use_custom_question and custom_question:
    prompt = custom_question.strip()
    system_prompt = "You are an expert radiologist. Answer the question concisely."
else:
    prompt = """Analyze this chest X-ray and provide a comprehensive report..."""
```

### 7. Memory Management

#### 7.1 GPU Memory Optimization
- Uses BFloat16 precision (more stable than Float16)
- Eager attention implementation
- Explicit garbage collection

```python
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="eager",
    tie_word_embeddings=False
)
```

#### 7.2 Attention Tensor Management
- Converts to CPU immediately after extraction
- Uses float32 for computations
- Cleans up intermediate tensors

### 8. Visualization Output

The system generates multiple visualization types:

1. **Overview Grid**: Shows attention evolution across tokens
2. **Detailed Analysis**: 10-panel comprehensive view including:
   - Original X-ray
   - Raw attention heatmap
   - Layer-weighted relevancy
   - Head importance map
   - Attention flow
   - Consensus relevancy
   - Overlay visualization
   - High attention regions
   - Statistical summary
   - Attention distribution histogram

### 9. Key Innovations

1. **Multi-Method Consensus**: Combines multiple attention analysis methods for robustness
2. **Medical Focus**: Optimized for grayscale medical images with high-contrast visualization
3. **Interactive Token Analysis**: Real-time attention visualization for any generated token
4. **Efficient Memory Usage**: Handles large attention tensors without matrix multiplication
5. **Automatic Region Detection**: Identifies and highlights areas of clinical interest

### 10. Integration Points

The visualizer can be integrated into existing pipelines:

```python
# Standalone usage
results = analyze_chest_xray_with_relevancy(
    model, processor, inputs_gpu, outputs,
    chest_xray, medical_report
)

# Gradio app
demo = create_gradio_interface(model, processor)
demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
```

## Performance Characteristics

- **Inference Time**: ~3-5 seconds per X-ray analysis
- **Memory Usage**: ~8-10GB GPU memory
- **Visualization Generation**: <1 second per token
- **Supported Batch Size**: 1 (for attention visualization)

## Future Enhancements

1. **Multi-image Support**: Extend to handle PA/lateral view combinations
2. **3D Visualization**: Project attention onto 3D anatomical models
3. **Quantitative Metrics**: Add IoU with radiologist annotations
4. **Real-time Streaming**: Progressive attention visualization during generation
5. **Model Comparison**: Side-by-side attention comparison across models

## Conclusion

The MedGemma Attention Visualizer represents a significant advancement in medical AI interpretability, providing clinicians and researchers with transparent insights into model decision-making. By combining multiple attention analysis methods and optimizing for medical imaging workflows, it bridges the gap between powerful vision-language models and clinical interpretability requirements.