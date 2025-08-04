"""MedGemma model evaluator for chest X-ray analysis."""

import json
import time
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import pandas as pd
from tqdm import tqdm

from config import ModelConfig
from database import DatabaseManager


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_answer: str
    model_explanation: str
    model_confidence: float
    model_findings: List[str]
    question: str
    study_id: str
    correct_answer: str
    image_path: str
    question_type: str
    attack_type: str
    variation_type: str
    response_time: float
    parse_success: bool = True
    error_message: Optional[str] = None


class MedGemmaEvaluator:
    """Evaluator for MedGemma model on chest X-ray questions."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the evaluator with model configuration."""
        self.config = config
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load MedGemma model and processor."""
        print(f"Loading model: {self.config.model_id}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id, 
            use_fast=True
        )
        print("Model loaded successfully!")
    
    def extract_json(self, response: str) -> Dict:
        """Extract JSON from model response."""
        response = response.strip()
        
        # Strip triple backticks if present
        if response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        # Strip leading "json" keyword
        if response.lower().startswith("json"):
            response = response[4:].strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    
    def _load_image(self, image_path: str, base_folder: Path) -> Image.Image:
        """Load image from path, handling relative and absolute paths."""
        if image_path.startswith("http"):
            # Handle URL images if needed
            raise NotImplementedError("URL images not supported")
        
        # Clean up the path
        clean_path = image_path.replace('../', '')
        
        if clean_path.startswith('/'):
            # Absolute path
            full_path = Path(clean_path)
        else:
            # Relative path
            full_path = base_folder / clean_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        return Image.open(full_path).convert("RGB")
    
    def generate_multiple_choice(
        self,
        prompt: str,
        options: str,
        image_paths: Union[str, List[str]],
        folder_path: Path
    ) -> Dict:
        """Generate answer for multiple choice question."""
        
        # Construct the multiple choice prompt
        mc_prompt = f"""Question: {prompt}

Options:
{options}

Instructions: Analyze the chest X-ray image(s) and select the single best answer from the options provided.
Respond with ONLY a JSON object in the following format:
{{
  "answer": "A/B/C/D",
  "explanation": "Brief one-sentence explanation for your choice",
  "confidence": 0.0-1.0,
  "key_findings": ["finding1", "finding2"]
}}"""
        
        # Process image paths
        if isinstance(image_paths, str):
            if image_paths.startswith('['):
                # String representation of list
                image_paths = ast.literal_eval(image_paths)
            else:
                image_paths = [image_paths]
        elif not isinstance(image_paths, list):
            image_paths = [image_paths]
        
        # Load images
        image_contents = []
        for idx, path in enumerate(image_paths):
            try:
                img = self._load_image(path, folder_path)
                image_contents.append({"type": "image", "image": img})
            except Exception as e:
                print(f"Warning: Could not load image {idx + 1}: {path}")
                print(f"Error: {e}")
        
        if not image_contents:
            raise ValueError("No images could be loaded")
        
        # Build messages
        user_content = image_contents + [{"type": "text", "text": mc_prompt}]
        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (
                        "You are an expert radiologist. Analyze chest X-rays and answer multiple choice questions. "
                        "When multiple views are provided (e.g., PA and lateral), consider both in your analysis. "
                        "Always respond with valid JSON format only. Do not include any text outside the JSON object."
                    )
                }]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=getattr(torch, self.config.torch_dtype))
        
        # Generate response
        input_len = inputs["input_ids"].shape[-1]
        start_time = time.time()
        
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample
            )
            generation = generation[0][input_len:]
        
        response_time = time.time() - start_time
        
        # Decode output
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        
        return {
            'model_response': decoded,
            'prompt': mc_prompt,
            'response_time': response_time,
            'full_response': decoded,
            'num_images': len(image_contents)
        }
    
    def evaluate_dataset(
        self,
        data: pd.DataFrame,
        folder_path: Path,
        attack_type: str = 'baseline',
        save_path: Optional[Path] = None,
        resume_from: Optional[str] = None,
        run_id: Optional[str] = None,
        db_manager: Optional[DatabaseManager] = None
    ) -> pd.DataFrame:
        """Evaluate model on a dataset."""
        
        results = []
        processed_ids = set()
        
        # Resume from previous run if specified
        if resume_from and Path(resume_from).exists():
            previous_results = pd.read_csv(resume_from)
            results = previous_results.to_dict('records')
            processed_ids = set(previous_results['study_id'])
            print(f"Resuming from {len(processed_ids)} processed samples")
        
        # Process each sample
        for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Evaluating {attack_type}"):
            if row['study_id'] in processed_ids:
                continue
            
            try:
                # Generate response
                result = self.generate_multiple_choice(
                    prompt=row['question'],
                    options=row['options'],
                    image_paths=row['image_path'],
                    folder_path=folder_path
                )
                
                # Parse response
                parsed = self.extract_json(result['model_response'])
                
                # Check correctness
                is_correct = parsed['answer'] == row['correct_answer']
                if not is_correct:
                    print(f"✗ Study {row['study_id']}: {row['correct_answer']} → {parsed['answer']}")
                
                # Create result entry
                eval_result = EvaluationResult(
                    model_answer=parsed['answer'],
                    model_explanation=parsed.get('explanation', ''),
                    model_confidence=parsed.get('confidence', 0.0),
                    model_findings=parsed.get('key_findings', []),
                    question=row['question'],
                    study_id=row['study_id'],
                    correct_answer=row['correct_answer'],
                    image_path=str(row['image_path']),
                    question_type=attack_type,
                    attack_type=attack_type,
                    variation_type=attack_type,
                    response_time=result['response_time']
                )
                
            except Exception as e:
                print(f"Error processing study {row['study_id']}: {e}")
                eval_result = EvaluationResult(
                    model_answer='ERROR',
                    model_explanation=str(e),
                    model_confidence=0.0,
                    model_findings=[],
                    question=row['question'],
                    study_id=row['study_id'],
                    correct_answer=row['correct_answer'],
                    image_path=str(row['image_path']),
                    question_type=attack_type,
                    attack_type=attack_type,
                    variation_type=attack_type,
                    response_time=0.0,
                    parse_success=False,
                    error_message=str(e)
                )
            
            result_dict = eval_result.__dict__
            results.append(result_dict)
            processed_ids.add(row['study_id'])
            
            # Save to database if available
            if db_manager and run_id:
                db_manager.insert_result(run_id, result_dict)
            
            # Save intermediate results
            if save_path and len(results) % 10 == 0:
                pd.DataFrame(results).to_csv(save_path, index=False)
        
        # Save final results
        results_df = pd.DataFrame(results)
        if save_path:
            results_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        
        return results_df