import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
from transformers import AutoProcessor, AutoModelForImageTextToText
import gc
# Disable parallelism to avoid conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Fresh start initialized")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Clear everything
gc.collect()
torch.cuda.empty_cache()

print("=== GPU-Safe Setup ===")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# Load your model and processor first
print("\n=== Loading Model with Safe Config ===")
model_id='google/medgemma-4b-it'
# Load processor first
processor = AutoProcessor.from_pretrained(model_id)
print("✓ Processor loaded")



try:
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float16
        device_map="cuda:0",
        attn_implementation="eager",
        # Important: set these to avoid issues
        tie_word_embeddings=False
    )
    
    # Critical: set model to eval mode
    model.eval()
    
    # Ensure attention output is enabled
    model.config.output_attentions = True
    
    print("✓ Model loaded successfully")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Launch the app
from medgemma_attention_visualizer import launch_app
# Launch on all network interfaces (0.0.0.0) so it's accessible from other machines
# Use server_name="127.0.0.1" for localhost only
launch_app(model, processor, server_name="0.0.0.0", server_port=7860)