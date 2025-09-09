import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import base64
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the relevancy analyzer from previous code
from medgemma_relevancy_analyzer import MedGemmaRelevancyAnalyzer, enhance_visualization_contrast

class MedGemmaApp:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.analyzer = MedGemmaRelevancyAnalyzer(model, processor)
        self.current_outputs = None
        self.current_inputs = None
        self.current_report = None
        self.current_image = None
        self.attention_cache = {}
        
    def format_medical_report(self, raw_text):
        """Format the medical report for better readability"""
        # Remove extra asterisks and clean up formatting
        text = raw_text.replace('**', '')
        text = text.replace('*', '')
        
        # Add proper line breaks
        text = text.replace('. ', '.\n\n')
        
        # Format numbered items
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Check if it's a numbered item
                if re.match(r'^\d+\.', line):
                    # Add markdown formatting for numbered items
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        formatted_lines.append(f"**{parts[0]}:** {parts[1].strip()}")
                    else:
                        formatted_lines.append(f"**{line}**")
                else:
                    formatted_lines.append(line)
        
        return '\n\n'.join(formatted_lines)
        
    def process_xray(self, image, use_custom_question, custom_question):
        """Process uploaded X-ray image with optional custom question"""
        if image is None:
            return "Please upload an image", None, None, None, None
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        self.current_image = image
        
        # Determine the prompt to use
        if use_custom_question and custom_question and custom_question.strip():
            prompt = custom_question.strip()
            system_prompt = "You are an expert radiologist. Answer the question about the chest X-ray concisely and accurately."
        else:
            # Default medical analysis prompt
            prompt = """Analyze this chest X-ray and provide a comprehensive report covering:
            1. Lung Fields: Describe any abnormalities, opacities, or normal findings
            2. Heart: Assess size and cardiac silhouette
            3. Bones: Note any visible bone abnormalities
            4. Soft Tissues: Any notable findings
            5. Medical Devices: Identify any visible medical devices
            6. Overall Impression: Summary of findings"""
            system_prompt = "You are an expert radiologist. Provide a detailed, structured analysis."
        
        # Create messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Move to GPU if available
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        inputs_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Generate with attention
        gen_kwargs = {
            "max_new_tokens": 200,
            "do_sample": False,
            "output_attentions": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs_gpu, **gen_kwargs)
        
        # Decode generated text
        generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
        raw_report = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up any repeated "output" tokens (potential tokenization issue)
        # This addresses the issue where custom questions result in repeated "output" text
        if raw_report.count("output") > 5:  # If there are many "output" repetitions
            # Remove consecutive "output" strings, keeping the actual medical content
            import re
            raw_report = re.sub(r'(\s*output\s*)+', ' ', raw_report).strip()
        
        # Format the report
        formatted_report = self.format_medical_report(raw_report)
        
        # Store for later use
        self.current_outputs = outputs
        self.current_inputs = inputs_gpu
        self.current_report = raw_report
        self.attention_cache = {}
        
        # Generate initial visualizations
        overview_fig = self.create_overview_visualization()
        token_options = self.get_token_options()
        
        return (
            formatted_report,
            overview_fig,
            gr.Dropdown(choices=token_options, value=token_options[0] if token_options else None, 
                       label="Select Token", interactive=True),
            None,  # Token visualization placeholder
            gr.Button(visible=True)  # Show analyze button
        )
    
    def create_overview_visualization(self):
        """Create overview of attention across all tokens"""
        if self.current_outputs is None:
            return None
        
        num_tokens = len(self.current_outputs.attentions)
        sample_tokens = [0, num_tokens//4, num_tokens//2, 3*num_tokens//4, num_tokens-1]
        sample_tokens = [t for t in sample_tokens if 0 <= t < num_tokens][:4]
        
        fig, axes = plt.subplots(2, len(sample_tokens), figsize=(4*len(sample_tokens), 8))
        
        if len(sample_tokens) == 1:
            axes = axes.reshape(2, 1)
        
        for idx, token_idx in enumerate(sample_tokens):
            # Extract raw attention
            raw_attn = self.extract_raw_attention(token_idx)
            
            # Compute simple relevancy
            simple_rel = self.analyzer.compute_simple_relevancy(
                self.current_outputs, self.current_inputs, token_idx
            )
            
            # Top row: Raw attention
            raw_enhanced = enhance_visualization_contrast(raw_attn)
            axes[0, idx].imshow(raw_enhanced, cmap='hot', interpolation='bicubic')
            axes[0, idx].set_title(f'Token {token_idx}: Raw Attention', fontsize=10)
            axes[0, idx].axis('off')
            
            # Bottom row: Relevancy
            rel_enhanced = enhance_visualization_contrast(simple_rel)
            axes[1, idx].imshow(rel_enhanced, cmap='jet', interpolation='bicubic')
            axes[1, idx].set_title(f'Token {token_idx}: Relevancy', fontsize=10)
            axes[1, idx].axis('off')
        
        plt.suptitle('Attention Overview Across Generated Tokens', fontsize=16)
        plt.tight_layout()
        
        # Convert to image for Gradio
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return Image.open(buf)
    
    def get_token_options(self):
        """Get list of tokens with their text for dropdown"""
        if self.current_report is None:
            return []
        
        words = self.current_report.split()
        num_tokens = len(self.current_outputs.attentions)
        
        options = []
        for i in range(min(num_tokens, len(words))):
            word = words[i] if i < len(words) else '...'
            # Truncate long words
            if len(word) > 15:
                word = word[:12] + '...'
            options.append(f"Token {i}: {word}")
        
        return options
    
    def analyze_token(self, token_selection):
        """Analyze attention for selected token"""
        if token_selection is None or self.current_outputs is None:
            return None
        
        # Extract token index
        token_idx = int(token_selection.split(":")[0].split()[-1])
        
        # Check cache
        if token_idx in self.attention_cache:
            return self.attention_cache[token_idx]
        
        # Create detailed visualization
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Original X-ray
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.current_image, cmap='gray')
        ax1.set_title('Input X-ray', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Raw attention
        raw_attn = self.extract_raw_attention(token_idx)
        raw_enhanced = enhance_visualization_contrast(raw_attn)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(raw_enhanced, cmap='hot', interpolation='bicubic')
        ax2.set_title('Raw Attention', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 3. Layer-weighted relevancy
        simple_rel = self.analyzer.compute_simple_relevancy(
            self.current_outputs, self.current_inputs, token_idx
        )
        simple_enhanced = enhance_visualization_contrast(simple_rel)
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(simple_enhanced, cmap='jet', interpolation='bicubic')
        ax3.set_title('Layer-Weighted Relevancy', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # 4. Head importance relevancy
        head_rel = self.analyzer.compute_head_importance_relevancy(
            self.current_outputs, self.current_inputs, token_idx
        )
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(head_rel, cmap='plasma', interpolation='bicubic')
        ax4.set_title('Head Importance', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # 5. Attention flow
        flow_rel = self.analyzer.compute_attention_flow(
            self.current_outputs, self.current_inputs, token_idx
        )
        flow_enhanced = enhance_visualization_contrast(flow_rel)
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(flow_enhanced, cmap='viridis', interpolation='bicubic')
        ax5.set_title('Attention Flow', fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        # 6. Consensus
        consensus = (simple_enhanced + head_rel + flow_enhanced) / 3
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.imshow(consensus, cmap='RdYlBu_r', interpolation='bicubic')
        ax6.set_title('Consensus Relevancy', fontsize=12, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
        
        # 7. Overlay on X-ray
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(self.current_image, cmap='gray')
        h, w = self.current_image.size[::-1]
        consensus_resized = cv2.resize(consensus, (w, h), interpolation=cv2.INTER_CUBIC)
        ax7.imshow(consensus_resized, cmap='jet', alpha=0.5)
        ax7.set_title('Attention Overlay', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
        # 8. High attention regions
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.imshow(self.current_image, cmap='gray')
        threshold = np.percentile(consensus_resized, 85)
        mask = consensus_resized > threshold
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                contour = contour.squeeze()
                if len(contour.shape) == 2 and contour.shape[0] > 2:
                    ax8.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2)
        ax8.set_title('High Attention Regions', fontsize=12, fontweight='bold')
        ax8.axis('off')
        
        # 9. Statistics
        ax9 = fig.add_subplot(gs[2:, :2])
        ax9.axis('off')
        
        stats_text = f"📊 ATTENTION STATISTICS FOR {token_selection}\n" + "="*60 + "\n\n"
        stats_text += f"• Raw attention     - Min: {raw_attn.min():.4f}, Max: {raw_attn.max():.4f}\n"
        stats_text += f"• Simple relevancy  - Min: {simple_rel.min():.4f}, Max: {simple_rel.max():.4f}\n"
        stats_text += f"• Head relevancy    - Min: {head_rel.min():.4f}, Max: {head_rel.max():.4f}\n"
        stats_text += f"• Flow relevancy    - Min: {flow_rel.min():.4f}, Max: {flow_rel.max():.4f}\n\n"
        
        # Find primary focus region
        threshold = np.percentile(consensus, 80)
        high_mask = consensus > threshold
        y_coords, x_coords = np.where(high_mask)
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
                
            stats_text += f"🎯 Primary focus region: {v_pos} {h_pos}\n"
            stats_text += f"📍 Coverage: {len(y_coords)} high-attention patches\n"
        
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # 10. Attention distribution
        ax10 = fig.add_subplot(gs[2:, 2:])
        ax10.hist(consensus.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax10.axvline(consensus.mean(), color='red', linestyle='--', 
                    label=f'Mean: {consensus.mean():.4f}')
        ax10.axvline(threshold, color='green', linestyle='--', 
                    label=f'80th %ile: {threshold:.4f}')
        ax10.set_xlabel('Attention Value')
        ax10.set_ylabel('Frequency')
        ax10.set_title('Consensus Attention Distribution', fontsize=12, fontweight='bold')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        plt.suptitle(f'Detailed Attention Analysis: {token_selection}', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        result_image = Image.open(buf)
        
        # Cache result
        self.attention_cache[token_idx] = result_image
        
        return result_image
    
    def extract_raw_attention(self, token_idx):
        """Extract raw attention for a token"""
        try:
            if token_idx >= len(self.current_outputs.attentions):
                token_idx = len(self.current_outputs.attentions) - 1
            
            attn = self.current_outputs.attentions[token_idx][-1].cpu().float()
            if len(attn.shape) == 4:
                attn = attn[0]
            
            avg_attn = attn.mean(dim=0)
            input_len = self.current_inputs['input_ids'].shape[1]
            
            gen_pos = min(input_len + token_idx, avg_attn.shape[0] - 1)
            
            if gen_pos >= 0 and 257 <= avg_attn.shape[1]:
                attn_to_image = avg_attn[gen_pos, 1:257]
                return attn_to_image.reshape(16, 16).numpy()
            else:
                return np.zeros((16, 16))
                
        except Exception as e:
            print(f"Error extracting attention: {e}")
            return np.zeros((16, 16))


def create_gradio_interface(model, processor):
    """Create the enhanced Gradio interface"""
    app = MedGemmaApp(model, processor)
    
    with gr.Blocks(title="MedGemma Chest X-ray Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏥 MedGemma Chest X-ray Analysis with Attention Visualization
        
        This tool provides chest X-ray analysis with transparent attention visualization.
        You can either use the default comprehensive analysis or ask specific questions about the X-ray.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Chest X-ray", type="pil")
                
                with gr.Group():
                    use_custom = gr.Checkbox(label="Ask a specific question", value=False)
                    custom_question = gr.Textbox(
                        label="Your Question", 
                        placeholder="e.g., 'Is there any sign of pneumonia?'",
                        lines=2,
                        visible=False
                    )
                
                analyze_btn = gr.Button("🔍 Analyze X-ray", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                medical_report = gr.Markdown(label="Medical Report", elem_id="report-box")
                
        with gr.Row():
            overview_plot = gr.Image(label="Attention Overview")
        
        with gr.Row():
            with gr.Column():
                token_dropdown = gr.Dropdown(label="Select Token for Detailed Analysis", 
                                           choices=[], interactive=False)
                analyze_token_btn = gr.Button("🔬 Analyze Selected Token", visible=False)
            
        with gr.Row():
            token_analysis = gr.Image(label="Token-Specific Attention Analysis")
        
        # Show/hide custom question input
        use_custom.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_custom],
            outputs=[custom_question]
        )
        
        # Connect main analysis function
        analyze_btn.click(
            fn=app.process_xray,
            inputs=[input_image, use_custom, custom_question],
            outputs=[medical_report, overview_plot, token_dropdown, 
                    token_analysis, analyze_token_btn]
        )
        
        # Connect token analysis
        analyze_token_btn.click(
            fn=app.analyze_token,
            inputs=[token_dropdown],
            outputs=[token_analysis]
        )
        
        # Examples section
        with gr.Accordion("📸 Example X-rays", open=False):
            gr.Examples(
                examples=[
                    ["normal_chest_xray.jpg", False, ""],
                    ["pneumonia_xray.jpg", True, "Is there evidence of pneumonia?"],
                    ["covid_xray.jpg", True, "Are there COVID-19 related findings?"],
                ],
                inputs=[input_image, use_custom, custom_question],
                label="Example Cases"
            )
        
        gr.Markdown("""
        ---
        ### 🔍 Understanding the Visualizations:
        
        **Attention Methods:**
        - **Raw Attention**: Direct attention weights from the model's last layer
        - **Layer-Weighted Relevancy**: Importance accumulated across all 34 layers
        - **Head Importance**: Focus from the most informative of 8 attention heads
        - **Attention Flow**: Information propagation through the network layers
        - **Consensus**: Combined relevancy from all methods for robust results
        
        **Interpretation Guide:**
        - 🔴 Red/Yellow areas: High attention regions where the model focuses most
        - 🔵 Blue areas: Low attention regions
        - Contours show the boundaries of high-attention regions
        """)
    
    return demo


# Standalone version of the relevancy analyzer
class MedGemmaRelevancyAnalyzer:
    """Relevancy analyzer implementation"""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.model.eval()
    
    def compute_simple_relevancy(self, outputs, inputs, token_idx):
        """Layer-weighted relevancy"""
        image_start, image_end = 1, 257
        input_length = inputs['input_ids'].shape[1]
        relevancy_scores = []
        
        if token_idx < len(outputs.attentions):
            token_attentions = outputs.attentions[token_idx]
            
            for layer_idx, layer_attn in enumerate(token_attentions):
                if torch.is_tensor(layer_attn):
                    layer_attn = layer_attn.cpu().float()
                    if len(layer_attn.shape) == 4:
                        layer_attn = layer_attn[0]
                    
                    layer_attn = layer_attn.mean(dim=0)
                    src_pos = min(input_length + token_idx, layer_attn.shape[0] - 1)
                    
                    if src_pos >= 0 and image_end <= layer_attn.shape[1]:
                        attn_to_image = layer_attn[src_pos, image_start:image_end]
                        layer_weight = (layer_idx + 1) / len(token_attentions)
                        weighted_attn = attn_to_image * layer_weight
                        relevancy_scores.append(weighted_attn)
        
        if relevancy_scores:
            final_relevancy = torch.stack(relevancy_scores).mean(dim=0)
            return final_relevancy.reshape(16, 16).numpy()
        return np.zeros((16, 16))
    
    def compute_head_importance_relevancy(self, outputs, inputs, token_idx):
        """Head importance relevancy"""
        image_start, image_end = 1, 257
        input_length = inputs['input_ids'].shape[1]
        
        if token_idx >= len(outputs.attentions):
            return np.zeros((16, 16))
        
        last_layer_attn = outputs.attentions[token_idx][-1].cpu().float()
        if len(last_layer_attn.shape) == 4:
            last_layer_attn = last_layer_attn[0]
        
        num_heads = last_layer_attn.shape[0]
        head_importance = []
        
        for h in range(num_heads):
            head_attn = last_layer_attn[h]
            src_pos = min(input_length + token_idx, head_attn.shape[0] - 1)
            
            if src_pos >= 0 and image_end <= head_attn.shape[1]:
                attn_to_image = head_attn[src_pos, image_start:image_end]
                max_attn = attn_to_image.max().item()
                entropy = -(attn_to_image * torch.log(attn_to_image + 1e-10)).sum().item()
                importance = max_attn * (1 / (1 + entropy))
                head_importance.append((h, importance, attn_to_image))
        
        head_importance.sort(key=lambda x: x[1], reverse=True)
        top_k = min(4, len(head_importance))
        relevancy_map = torch.zeros(256)
        
        for h, importance, attn in head_importance[:top_k]:
            relevancy_map += attn * importance
        
        if relevancy_map.max() > 0:
            p95 = torch.quantile(relevancy_map, 0.95)
            if p95 > 0:
                relevancy_map = torch.clamp(relevancy_map / p95, 0, 1)
            else:
                relevancy_map = relevancy_map / relevancy_map.max()
        
        return relevancy_map.reshape(16, 16).numpy()
    
    def compute_attention_flow(self, outputs, inputs, token_idx):
        """Attention flow through layers"""
        image_start, image_end = 1, 257
        input_length = inputs['input_ids'].shape[1]
        
        if token_idx >= len(outputs.attentions):
            return np.zeros((16, 16))
        
        token_attentions = outputs.attentions[token_idx]
        last_layer = token_attentions[-1].cpu().float()
        if len(last_layer.shape) == 4:
            last_layer = last_layer[0]
        
        flow = last_layer.mean(dim=0)
        src_pos = min(input_length + token_idx, flow.shape[0] - 1)
        
        if src_pos >= 0 and image_end <= flow.shape[1]:
            image_attention = flow[src_pos, image_start:image_end]
        else:
            image_attention = torch.zeros(256)
        
        layer_contributions = []
        for layer_idx in range(len(token_attentions) - 2, -1, -1):
            layer_attn = token_attentions[layer_idx].cpu().float()
            if len(layer_attn.shape) == 4:
                layer_attn = layer_attn[0]
            
            layer_attn = layer_attn.mean(dim=0)
            if src_pos < layer_attn.shape[0] and image_end <= layer_attn.shape[1]:
                layer_contribution = layer_attn[src_pos, image_start:image_end]
                layer_contributions.append(layer_contribution)
        
        if layer_contributions:
            weights = torch.tensor([0.5 ** i for i in range(len(layer_contributions))])
            weights = weights / weights.sum()
            combined = image_attention * 0.5
            for i, contrib in enumerate(layer_contributions):
                combined = combined + contrib * weights[i] * 0.5
            return combined.reshape(16, 16).numpy()
        
        return image_attention.reshape(16, 16).numpy()


def enhance_visualization_contrast(attention_map, method='percentile'):
    """Enhance contrast for better visibility"""
    if method == 'percentile':
        p5, p95 = np.percentile(attention_map, [5, 95])
        if p95 > p5:
            enhanced = np.clip((attention_map - p5) / (p95 - p5), 0, 1)
        else:
            enhanced = attention_map
    elif method == 'log':
        enhanced = np.log(attention_map + 1e-8)
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
    else:
        enhanced = attention_map
    return enhanced


# Launch function
def launch_app(model, processor, server_name="0.0.0.0", server_port=7860):
    """Launch the Gradio app
    
    Args:
        model: The MedGemma model
        processor: The MedGemma processor
        server_name: IP address to bind to ("0.0.0.0" for all interfaces, "127.0.0.1" for localhost only)
        server_port: Port number (default 7860)
    """
    demo = create_gradio_interface(model, processor)
    demo.launch(share=True, server_name=server_name, server_port=server_port)


# Example usage
if __name__ == "__main__":
    # Assuming model and processor are already loaded
    # launch_app(model, processor)
    print("Enhanced Gradio app ready to launch!")
    print("Call launch_app(model, processor) to start the interface")