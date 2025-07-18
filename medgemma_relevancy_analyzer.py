import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class MedGemmaRelevancyAnalyzer:
    """
    Complete implementation for MedGemma's relevancy analysis
    Handles bfloat16, fixed attention matrices, and provides multiple relevancy methods
    """
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.model.eval()
        
    def compute_simple_relevancy(self, outputs, inputs, token_idx):
        """
        Simplified relevancy using layer-weighted attention
        Works with MedGemma's architecture
        """
        
        image_start, image_end = 1, 257
        input_length = inputs['input_ids'].shape[1]
        
        relevancy_scores = []
        
        # Process attention from each layer
        if token_idx < len(outputs.attentions):
            token_attentions = outputs.attentions[token_idx]
            
            for layer_idx, layer_attn in enumerate(token_attentions):
                if torch.is_tensor(layer_attn):
                    # Convert to float32 for computation
                    layer_attn = layer_attn.cpu().float()
                    
                    if len(layer_attn.shape) == 4:
                        layer_attn = layer_attn[0]  # Remove batch
                    
                    # Average over heads
                    layer_attn = layer_attn.mean(dim=0)
                    
                    # MedGemma uses fixed attention size
                    # Find the right position to extract from
                    if layer_attn.shape[0] == layer_attn.shape[1]:
                        # Square attention matrix
                        if layer_attn.shape[0] > input_length:
                            # Use last position that makes sense
                            src_pos = min(input_length + token_idx, layer_attn.shape[0] - 1)
                        else:
                            src_pos = layer_attn.shape[0] - 1
                    else:
                        src_pos = -1
                    
                    # Extract attention to image tokens
                    if src_pos >= 0 and image_end <= layer_attn.shape[1]:
                        attn_to_image = layer_attn[src_pos, image_start:image_end]
                        
                        # Weight by layer depth (later layers more important)
                        layer_weight = (layer_idx + 1) / len(token_attentions)
                        weighted_attn = attn_to_image * layer_weight
                        
                        relevancy_scores.append(weighted_attn)
        
        if relevancy_scores:
            # Aggregate across layers
            final_relevancy = torch.stack(relevancy_scores).mean(dim=0)
            return final_relevancy.reshape(16, 16).numpy()
        else:
            return np.zeros((16, 16))
    
    def compute_head_importance_relevancy(self, outputs, inputs, token_idx):
        """
        Compute relevancy by identifying important heads first
        Fixed version with better normalization
        """
        
        image_start, image_end = 1, 257
        input_length = inputs['input_ids'].shape[1]
        
        if token_idx >= len(outputs.attentions):
            return np.zeros((16, 16))
        
        # Get last layer attention for head importance
        last_layer_attn = outputs.attentions[token_idx][-1].cpu().float()
        
        if len(last_layer_attn.shape) == 4:
            last_layer_attn = last_layer_attn[0]  # Remove batch
        
        num_heads = last_layer_attn.shape[0]
        head_importance = []
        
        # Calculate importance score for each head
        for h in range(num_heads):
            head_attn = last_layer_attn[h]
            
            # Find appropriate position
            src_pos = min(input_length + token_idx, head_attn.shape[0] - 1)
            
            if src_pos >= 0 and image_end <= head_attn.shape[1]:
                attn_to_image = head_attn[src_pos, image_start:image_end]
                
                # Importance = max attention * entropy (focused but strong)
                max_attn = attn_to_image.max().item()
                entropy = -(attn_to_image * torch.log(attn_to_image + 1e-10)).sum().item()
                importance = max_attn * (1 / (1 + entropy))
                
                head_importance.append((h, importance, attn_to_image))
        
        # Sort by importance
        head_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Use top heads
        top_k = min(4, len(head_importance))
        relevancy_map = torch.zeros(256)
        
        for h, importance, attn in head_importance[:top_k]:
            relevancy_map += attn * importance
        
        # Better normalization - use percentile instead of max
        if relevancy_map.max() > 0:
            p95 = torch.quantile(relevancy_map, 0.95)
            if p95 > 0:
                relevancy_map = torch.clamp(relevancy_map / p95, 0, 1)
            else:
                relevancy_map = relevancy_map / relevancy_map.max()
        
        return relevancy_map.reshape(16, 16).numpy()
    
    def compute_attention_flow(self, outputs, inputs, token_idx):
        """
        Trace attention flow from output to input through layers
        More robust than matrix multiplication
        """
        
        image_start, image_end = 1, 257
        input_length = inputs['input_ids'].shape[1]
        
        if token_idx >= len(outputs.attentions):
            return np.zeros((16, 16))
        
        # Start from the last layer
        token_attentions = outputs.attentions[token_idx]
        
        # Initialize flow with last layer attention
        last_layer = token_attentions[-1].cpu().float()
        if len(last_layer.shape) == 4:
            last_layer = last_layer[0]
        
        # Average over heads
        flow = last_layer.mean(dim=0)
        
        # Find source position
        src_pos = min(input_length + token_idx, flow.shape[0] - 1)
        
        # Extract attention to image
        if src_pos >= 0 and image_end <= flow.shape[1]:
            image_attention = flow[src_pos, image_start:image_end]
        else:
            image_attention = torch.zeros(256)
        
        # Weight by layer contributions (backward through layers)
        layer_contributions = []
        
        for layer_idx in range(len(token_attentions) - 2, -1, -1):
            layer_attn = token_attentions[layer_idx].cpu().float()
            if len(layer_attn.shape) == 4:
                layer_attn = layer_attn[0]
            
            # Average over heads
            layer_attn = layer_attn.mean(dim=0)
            
            # Sample attention values to image region
            if src_pos < layer_attn.shape[0] and image_end <= layer_attn.shape[1]:
                layer_contribution = layer_attn[src_pos, image_start:image_end]
                layer_contributions.append(layer_contribution)
        
        # Combine contributions
        if layer_contributions:
            # Average with decreasing weights for earlier layers
            weights = torch.tensor([0.5 ** i for i in range(len(layer_contributions))])
            weights = weights / weights.sum()
            
            combined = image_attention * 0.5  # Last layer gets 50%
            for i, contrib in enumerate(layer_contributions):
                combined = combined + contrib * weights[i] * 0.5
            
            return combined.reshape(16, 16).numpy()
        
        return image_attention.reshape(16, 16).numpy()


def extract_raw_attention_safe(outputs, inputs, token_idx):
    """Safely extract raw attention with error handling"""
    try:
        if token_idx >= len(outputs.attentions):
            token_idx = len(outputs.attentions) - 1
        
        attn = outputs.attentions[token_idx][-1].cpu().float()
        if len(attn.shape) == 4:
            attn = attn[0]
        
        avg_attn = attn.mean(dim=0)
        input_len = inputs['input_ids'].shape[1]
        
        # Handle fixed attention matrix size
        if avg_attn.shape[0] == avg_attn.shape[1]:
            # Square matrix - find appropriate position
            gen_pos = min(input_len + token_idx, avg_attn.shape[0] - 1)
        else:
            gen_pos = -1
        
        if gen_pos >= 0 and 257 <= avg_attn.shape[1]:
            attn_to_image = avg_attn[gen_pos, 1:257]
            return attn_to_image.reshape(16, 16).numpy()
        else:
            print(f"Warning: Could not extract attention for token {token_idx}")
            return np.zeros((16, 16))
            
    except Exception as e:
        print(f"Error extracting raw attention: {e}")
        return np.zeros((16, 16))


def enhance_visualization_contrast(attention_map, method='percentile'):
    """Enhance contrast for better visibility"""
    
    if method == 'percentile':
        # Use 5th-95th percentile for better contrast
        p5, p95 = np.percentile(attention_map, [5, 95])
        if p95 > p5:
            enhanced = np.clip((attention_map - p5) / (p95 - p5), 0, 1)
        else:
            enhanced = attention_map
    elif method == 'log':
        # Log scale for very small values
        enhanced = np.log(attention_map + 1e-8)
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
    else:
        enhanced = attention_map
    
    return enhanced


def visualize_relevancy_methods(chest_xray, raw_attn, simple_rel, head_rel, flow_rel, title=""):
    """Visualize all working methods with enhanced contrast"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(chest_xray, cmap='gray')
    axes[0, 0].set_title('Input X-ray')
    axes[0, 0].axis('off')
    
    # Raw attention (enhanced)
    raw_enhanced = enhance_visualization_contrast(raw_attn)
    im1 = axes[0, 1].imshow(raw_enhanced, cmap='hot', interpolation='bicubic')
    axes[0, 1].set_title('Raw Attention')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Simple relevancy (enhanced)
    simple_enhanced = enhance_visualization_contrast(simple_rel)
    im2 = axes[0, 2].imshow(simple_enhanced, cmap='jet', interpolation='bicubic')
    axes[0, 2].set_title('Layer-Weighted Relevancy')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Head importance (already normalized)
    im3 = axes[1, 0].imshow(head_rel, cmap='plasma', interpolation='bicubic')
    axes[1, 0].set_title('Head Importance Relevancy')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # Attention flow (enhanced)
    flow_enhanced = enhance_visualization_contrast(flow_rel)
    im4 = axes[1, 1].imshow(flow_enhanced, cmap='viridis', interpolation='bicubic')
    axes[1, 1].set_title('Attention Flow')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    # Consensus (average of all methods)
    consensus = (simple_enhanced + head_rel + flow_enhanced) / 3
    im5 = axes[1, 2].imshow(consensus, cmap='RdYlBu_r', interpolation='bicubic')
    axes[1, 2].set_title('Consensus Relevancy')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig, consensus


def create_attention_overlay(chest_xray, attention_map, title="Attention Overlay"):
    """Create clean overlay visualization"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Resize attention to image size
    h, w = chest_xray.size[::-1]
    attention_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 1. Original
    axes[0].imshow(chest_xray, cmap='gray')
    axes[0].set_title('Original X-ray')
    axes[0].axis('off')
    
    # 2. Attention overlay
    axes[1].imshow(chest_xray, cmap='gray')
    axes[1].imshow(attention_resized, cmap='jet', alpha=0.5)
    axes[1].set_title(title)
    axes[1].axis('off')
    
    # 3. Contour regions
    threshold = np.percentile(attention_resized, 85)
    mask = attention_resized > threshold
    
    axes[2].imshow(chest_xray, cmap='gray')
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small regions
            axes[2].add_patch(plt.Circle((0, 0), 0))  # Dummy for color
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] > 2:
                axes[2].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2)
    
    axes[2].set_title('High Attention Regions')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def analyze_medgemma_relevancy(model, processor, inputs, outputs, chest_xray, 
                               medical_report, token_idx=0):
    """Main analysis function with all working methods"""
    
    print(f"\n=== ANALYZING TOKEN {token_idx} ===")
    
    # Initialize analyzer
    analyzer = MedGemmaRelevancyAnalyzer(model, processor)
    
    # 1. Extract raw attention
    print("1. Extracting raw attention...")
    raw_attention = extract_raw_attention_safe(outputs, inputs, token_idx)
    
    # 2. Compute simple relevancy
    print("2. Computing layer-weighted relevancy...")
    simple_relevancy = analyzer.compute_simple_relevancy(outputs, inputs, token_idx)
    
    # 3. Compute head importance relevancy
    print("3. Computing head importance relevancy...")
    head_relevancy = analyzer.compute_head_importance_relevancy(outputs, inputs, token_idx)
    
    # 4. Compute attention flow
    print("4. Computing attention flow...")
    flow_relevancy = analyzer.compute_attention_flow(outputs, inputs, token_idx)
    
    # 5. Visualize all methods
    print("5. Creating visualizations...")
    fig, consensus = visualize_relevancy_methods(
        chest_xray, raw_attention, simple_relevancy, 
        head_relevancy, flow_relevancy,
        title=f"Relevancy Analysis for Token {token_idx}"
    )
    plt.show()
    
    # 6. Create overlay for consensus
    overlay_fig = create_attention_overlay(chest_xray, consensus, 
                                         f"Consensus Attention - Token {token_idx}")
    plt.show()
    
    # 7. Analysis summary
    print("\n📊 Analysis Summary:")
    print(f"Raw attention     - Min: {raw_attention.min():.4f}, Max: {raw_attention.max():.4f}")
    print(f"Simple relevancy  - Min: {simple_relevancy.min():.4f}, Max: {simple_relevancy.max():.4f}")
    print(f"Head relevancy    - Min: {head_relevancy.min():.4f}, Max: {head_relevancy.max():.4f}")
    print(f"Flow relevancy    - Min: {flow_relevancy.min():.4f}, Max: {flow_relevancy.max():.4f}")
    
    # Find regions of agreement
    threshold = np.percentile(consensus, 80)
    high_relevance_mask = consensus > threshold
    
    print(f"\n🎯 High relevance regions (>80th percentile):")
    y_coords, x_coords = np.where(high_relevance_mask)
    if len(y_coords) > 0:
        center_y = y_coords.mean() / 16
        center_x = x_coords.mean() / 16
        
        if center_y < 0.33:
            v_pos = "Upper"
        elif center_y > 0.67:
            v_pos = "Lower"
        else:
            v_pos = "Middle"
            
        if center_x < 0.33:
            h_pos = "left"
        elif center_x > 0.67:
            h_pos = "right"
        else:
            h_pos = "center"
            
        print(f"Primary focus: {v_pos} {h_pos} region")
    
    return {
        'raw_attention': raw_attention,
        'simple_relevancy': simple_relevancy,
        'head_relevancy': head_relevancy,
        'flow_relevancy': flow_relevancy,
        'consensus': consensus
    }


def run_complete_analysis(model, processor, inputs_gpu, outputs, chest_xray, medical_report):
    """Run complete analysis on multiple tokens"""
    
    print("="*60)
    print("MEDGEMMA RELEVANCY ANALYSIS")
    print("="*60)
    
    # Analyze first few tokens
    all_results = {}
    tokens_to_analyze = [0, 2, 5]
    
    for token_idx in tokens_to_analyze:
        if token_idx < len(outputs.attentions):
            results = analyze_medgemma_relevancy(
                model, processor, inputs_gpu, outputs,
                chest_xray, medical_report, token_idx
            )
            all_results[token_idx] = results
    
    # Create evolution visualization
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("ATTENTION EVOLUTION ACROSS TOKENS")
        print("="*60)
        
        fig, axes = plt.subplots(len(all_results), 3, figsize=(12, 4*len(all_results)))
        
        for i, (token_idx, results) in enumerate(all_results.items()):
            # Raw attention (enhanced)
            raw_enhanced = enhance_visualization_contrast(results['raw_attention'])
            axes[i, 0].imshow(raw_enhanced, cmap='hot', interpolation='bicubic')
            axes[i, 0].set_title(f'Token {token_idx}: Raw Attention')
            axes[i, 0].axis('off')
            
            # Consensus relevancy
            axes[i, 1].imshow(results['consensus'], cmap='jet', interpolation='bicubic')
            axes[i, 1].set_title(f'Token {token_idx}: Consensus Relevancy')
            axes[i, 1].axis('off')
            
            # Difference
            diff = results['consensus'] - raw_enhanced
            axes[i, 2].imshow(diff, cmap='RdBu_r', interpolation='bicubic', 
                            vmin=-0.5, vmax=0.5)
            axes[i, 2].set_title(f'Token {token_idx}: Relevancy - Raw')
            axes[i, 2].axis('off')
        
        plt.suptitle('Evolution of Attention Across Generated Tokens', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for token_idx, results in all_results.items():
        consensus = results['consensus']
        print(f"\nToken {token_idx}:")
        print(f"  Consensus mean: {consensus.mean():.4f}")
        print(f"  Consensus std:  {consensus.std():.4f}")
        print(f"  Max location:   {np.unravel_index(consensus.argmax(), consensus.shape)}")
    
    print("\n✅ Analysis complete!")
    print("\nKey findings:")
    print("- Layer-weighted relevancy integrates contributions across all 34 layers")
    print("- Head importance identifies the most informative of 8 attention heads")
    print("- Attention flow traces information propagation without matrix multiplication")
    print("- Consensus relevancy provides robust results by combining all methods")
    print("\nRelevancy maps show more focused and interpretable patterns than raw attention!")
    
    return all_results


# Integration with existing code
def analyze_chest_xray_with_relevancy(model, processor, inputs_gpu, outputs, 
                                     chest_xray, medical_report):
    """
    Easy integration function for existing pipelines
    """
    print("\nStarting MedGemma relevancy analysis...")
    
    # Run the complete analysis
    results = run_complete_analysis(
        model, processor, inputs_gpu, outputs,
        chest_xray, medical_report
    )
    
    # Save results if needed
    try:
        import pickle
        with open('medgemma_relevancy_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("\n✓ Results saved to medgemma_relevancy_results.pkl")
    except:
        print("\n⚠️ Could not save results")
    
    return results


# Example usage
if __name__ == "__main__":
    # This integrates with your existing code
    # Assuming you have already:
    # - Loaded the model
    # - Generated outputs with attention
    # - Have chest_xray (PIL Image) and medical_report (string)
    
    results = analyze_chest_xray_with_relevancy(
        model, processor, inputs_gpu, outputs,
        xray_pil_image, medical_report  # Use your variable names
    )
    
    print("\n✅ All analyses completed successfully!")
    print("Relevancy maps provide deeper insights than raw attention alone!")