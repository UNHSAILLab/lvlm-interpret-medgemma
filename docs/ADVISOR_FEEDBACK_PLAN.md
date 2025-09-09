# Implementation Plan for Advisor Feedback
## Addressing Faithfulness vs Plausibility in MedGemma Attention Visualization

---

## 🎯 Executive Summary

Your advisor correctly identifies that our current implementation provides **plausibility visualizations** (where the model might be looking) rather than **faithful explanations** (what causally drives the output). We need to:
1. Strengthen gradient-based methods for faithfulness
2. Add quantitative validation experiments
3. Fix token alignment issues
4. Verify spatial assumptions
5. Keep all current functionality while adding new capabilities

---

## 📋 Prioritized Implementation Plan

### Phase 1: Non-Breaking Improvements (Week 1)
*These can be added without modifying existing functionality*

#### 1.1 Add Faithfulness Metrics Tab
Create a new tab "Faithfulness Analysis" that includes:
```python
class FaithfulnessAnalyzer:
    def deletion_curve(self, image, attention_map, model, prompt, target_word):
        """Progressively delete top-k% important patches"""
        baseline_logprob = self.get_target_logprob(image, prompt, target_word)
        
        deletions = []
        for k in [10, 20, 30, 50, 70, 90]:
            masked_image = self.mask_top_k_percent(image, attention_map, k)
            new_logprob = self.get_target_logprob(masked_image, prompt, target_word)
            drop = baseline_logprob - new_logprob
            deletions.append((k, drop))
        
        return deletions  # Steeper curve = more faithful
    
    def insertion_curve(self, image, attention_map, model, prompt, target_word):
        """Start from blurred, progressively reveal important patches"""
        blurred = self.blur_image(image)
        
        insertions = []
        for k in [10, 20, 30, 50, 70, 90]:
            revealed_image = self.reveal_top_k_percent(blurred, image, attention_map, k)
            logprob = self.get_target_logprob(revealed_image, prompt, target_word)
            insertions.append((k, logprob))
        
        return insertions  # Faster recovery = more faithful
    
    def comprehensiveness_sufficiency(self, image, attention_map, model, prompt, target_word):
        """Compute both metrics"""
        baseline = self.get_target_logprob(image, prompt, target_word)
        
        # Comprehensiveness: Remove top 20% - should drop significantly
        masked = self.mask_top_k_percent(image, attention_map, 20)
        comp_score = (baseline - self.get_target_logprob(masked, prompt, target_word)) / baseline
        
        # Sufficiency: Keep only top 20% - should maintain performance
        kept = self.keep_only_top_k_percent(image, attention_map, 20)
        suff_score = self.get_target_logprob(kept, prompt, target_word) / baseline
        
        return comp_score, suff_score
```

#### 1.2 Add Sanity Check Tab
New tab "Sanity Checks" with:
```python
class SanityChecker:
    def checkerboard_test(self, model, processor):
        """Verify spatial token ordering assumption"""
        # Create image with bright upper-left quadrant
        test_img = np.zeros((224, 224, 3))
        test_img[:112, :112] = 255  # Bright upper-left
        
        # Get attention map
        attn = self.get_attention(test_img, "bright quadrant")
        
        # Check if attention concentrates in upper-left after reshape
        gh, gw = factor_to_grid(len(attn), 224, 224)
        grid = attn.reshape(gh, gw)
        
        ul_mass = grid[:gh//2, :gw//2].sum()
        total_mass = grid.sum()
        
        return ul_mass / total_mass > 0.5  # Should be True
    
    def model_randomization_test(self, model, image, prompt):
        """Progressively randomize vision layers"""
        original_attn = self.get_attention(model, image, prompt)
        
        degradation = []
        for layer_idx in range(23, 15, -1):  # Randomize from last to first
            self.randomize_layer(model, f"vision_tower.encoder.layers.{layer_idx}")
            random_attn = self.get_attention(model, image, prompt)
            
            # Measure correlation with original
            corr = np.corrcoef(original_attn.flatten(), random_attn.flatten())[0, 1]
            degradation.append((layer_idx, corr))
            
            self.restore_layer(model, f"vision_tower.encoder.layers.{layer_idx}")
        
        return degradation  # Should decay to ~0
    
    def label_randomization_test(self, model, image):
        """Random target words should give different maps"""
        random_words = ["pneumonia", "car", "banana", "quantum", "happiness"]
        
        maps = []
        for word in random_words:
            attn = self.get_attention(model, image, f"Is there {word}?")
            maps.append(attn)
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(maps)):
            for j in range(i+1, len(maps)):
                corr = np.corrcoef(maps[i].flatten(), maps[j].flatten())[0, 1]
                correlations.append(corr)
        
        return np.mean(correlations) < 0.3  # Should be low correlation
```

---

### Phase 2: Enhanced Gradient Methods (Week 1-2)
*Improve gradient computation without breaking existing Grad-CAM*

#### 2.1 Multi-Token Gradient
```python
def gradcam_multi_token(model, processor, image, prompt, target_phrase):
    """Compute gradient for entire phrase, not just first token"""
    
    # Get all subword tokens for the phrase
    tokens = processor.tokenizer.encode(target_phrase, add_special_tokens=False)
    
    # Teacher-force generation with the target phrase
    inputs = prepare_inputs(image, prompt)
    
    # Forward pass
    with torch.enable_grad():
        outputs = model(**inputs)
        
        # Sum log-probs over ALL tokens of target phrase
        total_loss = 0
        for t, token_id in enumerate(tokens):
            if t < outputs.logits.shape[1]:
                logprob = F.log_softmax(outputs.logits[0, t], dim=-1)
                total_loss += logprob[token_id]
        
        # Backprop to vision features
        total_loss.backward()
    
    # Extract and process gradients
    return process_gradients(...)
```

#### 2.2 Integrated Gradients Option
```python
def integrated_gradients(model, image, prompt, target_word, steps=50):
    """More faithful than raw gradients"""
    baseline = torch.zeros_like(image)
    
    # Interpolate between baseline and image
    alphas = torch.linspace(0, 1, steps)
    gradients = []
    
    for alpha in alphas:
        interpolated = baseline + alpha * (image - baseline)
        grad = compute_gradient(model, interpolated, prompt, target_word)
        gradients.append(grad)
    
    # Integrate
    integrated_grad = torch.stack(gradients).mean(dim=0)
    attribution = integrated_grad * (image - baseline)
    
    return attribution
```

---

### Phase 3: Fix Token Alignment (Week 2)
*Replace approximate token finding with exact alignment*

#### 3.1 Use Offset Mapping
```python
def find_target_tokens_exact(tokenizer, prompt, target_phrase):
    """Use offset mapping for exact alignment"""
    
    # Use fast tokenizer with offset mapping
    encoding = tokenizer(prompt, return_offsets_mapping=True)
    tokens = encoding.input_ids
    offsets = encoding.offset_mapping
    
    # Find target phrase in original text
    target_start = prompt.find(target_phrase)
    target_end = target_start + len(target_phrase)
    
    # Find tokens that overlap with target span
    target_token_ids = []
    for idx, (start, end) in enumerate(offsets):
        if start < target_end and end > target_start:
            target_token_ids.append(idx)
    
    return target_token_ids
```

---

### Phase 4: Validation Experiments (Week 2-3)
*Quantitative evaluation on MIMIC subset*

#### 4.1 Deletion/Insertion Benchmark
```python
def run_deletion_insertion_benchmark(dataset, model, method='gradcam'):
    """Full benchmark on dataset"""
    results = {
        'deletion_auc': [],
        'insertion_auc': [],
        'comprehensiveness': [],
        'sufficiency': []
    }
    
    for sample in dataset:
        image = sample['image']
        question = sample['question']
        target = extract_medical_term(question)  # e.g., "pneumonia"
        
        # Get attention map
        if method == 'gradcam':
            attn_map = gradcam_multi_token(model, image, question, target)
        elif method == 'cross_attention':
            attn_map = get_cross_attention(model, image, question, target)
        
        # Run experiments
        del_curve = deletion_curve(image, attn_map, model, question, target)
        ins_curve = insertion_curve(image, attn_map, model, question, target)
        
        results['deletion_auc'].append(compute_auc(del_curve))
        results['insertion_auc'].append(compute_auc(ins_curve))
        
        comp, suff = comprehensiveness_sufficiency(image, attn_map, model, question, target)
        results['comprehensiveness'].append(comp)
        results['sufficiency'].append(suff)
    
    # Compute statistics with bootstrap CI
    return compute_statistics_with_ci(results)
```

#### 4.2 Pointing Game (if bounding boxes available)
```python
def pointing_game_accuracy(dataset, model, method='gradcam'):
    """Evaluate if peak attention falls in ground truth region"""
    correct = 0
    total = 0
    
    for sample in dataset:
        if 'bbox' not in sample:
            continue
            
        image = sample['image']
        question = sample['question']
        bbox = sample['bbox']  # Ground truth region
        
        # Get attention map
        attn_map = get_attention_map(model, image, question, method)
        
        # Find peak
        peak_y, peak_x = np.unravel_index(np.argmax(attn_map), attn_map.shape)
        
        # Check if peak is inside bbox
        if bbox['x1'] <= peak_x <= bbox['x2'] and bbox['y1'] <= peak_y <= bbox['y2']:
            correct += 1
        total += 1
    
    return correct / total
```

---

### Phase 5: UI Integration (Week 3)
*Add new capabilities to the interface*

#### 5.1 New "Validation" Tab
```python
with gr.TabItem("Validation & Faithfulness"):
    gr.Markdown("## Quantitative Faithfulness Metrics")
    
    with gr.Row():
        with gr.Column():
            # Select sample and method
            val_sample = gr.Dropdown(label="Select Sample")
            val_method = gr.Radio(
                ["Cross-Attention", "Grad-CAM", "Multi-Token Gradient"],
                label="Method"
            )
            run_validation_btn = gr.Button("Run Validation")
        
        with gr.Column():
            # Results display
            deletion_plot = gr.Plot(label="Deletion Curve")
            insertion_plot = gr.Plot(label="Insertion Curve")
            metrics_table = gr.Dataframe(
                headers=["Metric", "Value", "Target", "Status"],
                label="Faithfulness Metrics"
            )
    
    with gr.Row():
        sanity_results = gr.Markdown(label="Sanity Check Results")
```

#### 5.2 Method Comparison Mode
```python
def compare_methods(image, prompt, target_word):
    """Compare all methods side-by-side"""
    methods = {
        'cross_attention': get_cross_attention,
        'gradcam_single': gradcam_on_vision,
        'gradcam_multi': gradcam_multi_token,
        'integrated_grad': integrated_gradients
    }
    
    results = {}
    for name, method in methods.items():
        attn_map = method(model, image, prompt, target_word)
        
        # Compute faithfulness metrics
        del_auc = compute_deletion_auc(image, attn_map, model, prompt, target_word)
        comp, suff = comprehensiveness_sufficiency(image, attn_map, model, prompt, target_word)
        
        results[name] = {
            'map': attn_map,
            'deletion_auc': del_auc,
            'comprehensiveness': comp,
            'sufficiency': suff
        }
    
    return create_comparison_visualization(results)
```

---

## 🛡️ Non-Breaking Implementation Strategy

### Preserve Existing Functionality
1. **Keep all current methods** - Add new ones alongside
2. **Add new tabs** - Don't modify existing tabs initially
3. **Create feature flags** - Allow enabling/disabling new features
4. **Backward compatibility** - Ensure old saved results still work

### Gradual Rollout
```python
class Config:
    # Feature flags
    ENABLE_FAITHFULNESS_TAB = True
    ENABLE_MULTI_TOKEN_GRADIENT = True
    ENABLE_INTEGRATED_GRADIENTS = False  # Experimental
    ENABLE_SANITY_CHECKS = True
    
    # Method priorities (fallback chain)
    METHOD_PRIORITY = [
        'multi_token_gradient',  # New, most faithful
        'gradcam_single',        # Current fallback
        'cross_attention',       # Fast but less faithful
        'activation_norm',       # Last resort
        'uniform'                # Final fallback
    ]
```

### Testing Strategy
1. **A/B Testing**: Run new methods alongside old ones
2. **Validation Dataset**: Create small annotated subset
3. **Regression Tests**: Ensure old functionality unchanged
4. **Performance Monitoring**: Track inference time

---

## 📊 Success Metrics

### Quantitative Targets
- **Deletion AUC**: > 0.7 (higher is better)
- **Insertion AUC**: > 0.6 (higher is better)
- **Comprehensiveness**: > 0.5 at 20% masking
- **Sufficiency**: > 0.3 at 20% retention
- **Pointing Game**: > 60% accuracy (if boxes available)

### Sanity Checks Must Pass
- ✅ Checkerboard test: > 50% concentration
- ✅ Model randomization: Correlation < 0.1 after full randomization
- ✅ Label randomization: Mean correlation < 0.3
- ✅ Robustness: < 20% shift under minor perturbations

---

## 🗓️ Implementation Timeline

### Week 1
- [ ] Implement faithfulness metrics (deletion/insertion)
- [ ] Add comprehensiveness/sufficiency computation
- [ ] Create sanity check functions
- [ ] Begin multi-token gradient implementation

### Week 2
- [ ] Complete multi-token gradient
- [ ] Fix token alignment with offset mapping
- [ ] Implement integrated gradients (optional)
- [ ] Create validation dataset subset

### Week 3
- [ ] Add new UI tabs
- [ ] Run full validation experiments
- [ ] Generate comparison reports
- [ ] Document results and findings

### Week 4
- [ ] Performance optimization
- [ ] Final testing and validation
- [ ] Documentation and paper writing
- [ ] Prepare for advisor review

---

## 🔬 Quick Experiments to Run First

### 1. Verify Spatial Assumption (Day 1)
```python
# Quick checkerboard test
test_img = create_checkerboard()
attn = get_attention(model, test_img, "checkerboard pattern")
verify_spatial_alignment(attn)
```

### 2. Compare Methods on Single Sample (Day 2)
```python
# Pick one clear pneumonia case
sample = dataset[find_clear_pneumonia_case()]
compare_all_methods(sample)
```

### 3. Mini Deletion Experiment (Day 3)
```python
# Run on 10 samples only
mini_results = run_deletion_insertion_benchmark(dataset[:10])
print(f"Initial deletion AUC: {np.mean(mini_results['deletion_auc'])}")
```

---

## 💡 Key Insights from Advisor

### What to Keep
✅ Token-conditioned cross-attention approach
✅ Robust fallback system
✅ Quality metrics (inside-body ratio, etc.)
✅ Clear method labeling

### What to Improve
⚠️ Make gradients primary for "faithfulness"
⚠️ Fix token alignment precision
⚠️ Verify spatial assumptions
⚠️ Add quantitative validation

### What to Add
➕ Deletion/insertion curves
➕ Comprehensiveness/sufficiency metrics
➕ Sanity checks
➕ Method comparison tools

---

## 📝 Notes for Implementation

1. **Start with validation experiments** - Know current accuracy before changing
2. **Keep parallel implementations** - Don't replace, add alongside
3. **Document everything** - Track what improves and what doesn't
4. **Regular advisor check-ins** - Share results frequently

---

*This plan addresses all advisor feedback while maintaining system stability and current functionality.*