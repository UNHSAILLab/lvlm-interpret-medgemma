import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch



from one_pixel_attack_helper import *


to_attack=pd.read_csv('/home/bsada1/ReXGradient-160K/to_attack.csv')

source_folder='/home/bsada1/ReXGradient-160K/'
output_folder='/home/bsada1/ReXGradient-160K/deid_png_onepx_pneumo/'

print("Ready to run full attack generation.")
print(f"Total images to process: {len(to_attack)}")


print(f"Source folder: {source_folder}")
print(f"Output folder: {output_folder}")
print("\nSettings:")
print("  max_iter: 100")
print("  pop_size: 200")
print("  pathology: Pneumonia")

# Estimate time
print(f"\nEstimated time: {len(to_attack) * 1.5:.0f} minutes ({1.5} min/image)")
# Run the attack
one_pixel_results = create_one_pixel_attacks(
    df=to_attack,
    source_folder=source_folder,
    output_folder=output_folder,
    pathology='Pneumonia',
    max_iter=100,
    pop_size=200
)

one_px_log = pd.read_csv('/home/bsada1/ReXGradient-160K/one_pixel_attack_log.csv')

merged_df = pd.merge(
    one_px_log,
    to_attack,
    left_on=['study_id'],
    right_on=['study_id'],
    how='inner' )

print("\nChecking merged_df structure...")
print(f"Shape: {merged_df.shape}")
print(f"Columns: {merged_df.columns.tolist()}")


model_id='google/medgemma-4b-it'
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

def extract_json(s: str) -> dict:
    s = s.strip()
    # strip triple backticks
    if s.startswith("```") and s.endswith("```"):
        s = s[3:-3].strip()
    # strip a leading "json"
    if s.lower().startswith("json"):
        s = s[4:].strip()
    return json.loads(s)

def generate_gemma3_multiple_choice(prompt, options, image_path,folder_path):
    # Construct the multiple choice prompt with JSON instruction
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

    # Process image path(s)
    image_contents = []

    try:
        # Parse the image path
        if isinstance(image_path, str):
            if image_path.startswith('['):
                # It's a string representation of a list
                import ast
                image_paths = ast.literal_eval(image_path)
            else:
                # Single image path
                image_paths = [image_path]
        else:
            image_paths = image_path if isinstance(image_path, list) else [image_path]

        # Load all images
        for idx, path in enumerate(image_paths):
            if path.startswith("http"):
                image_contents.append({"type": "image", "image": path})
            else:
                # Clean up the path
                actual_path = path.replace('../', '')
                if actual_path.startswith('/'):
                    # Absolute path
                    full_path = actual_path
                else:
                    # Relative path - adjust based on your directory structure
                    full_path = f"{folder_path}/{actual_path}"

                try:
                    img = Image.open(full_path).convert("RGB")
                    image_contents.append({"type": "image", "image": img})
                    print(f"  Loaded image {idx + 1}: {actual_path}")
                except Exception as e:
                    print(f"  Warning: Could not load image {idx + 1}: {actual_path}")
                    print(f"  Error: {e}")

        if not image_contents:
            raise Exception("No images could be loaded")

    except Exception as e:
        print(f"Error processing images: {e}")
        return {
            "answer": "ERROR",
            "explanation": f"Image loading error: {str(e)}",
            "confidence": 0.0,
            "key_findings": [],
            "full_response": f"Image loading error: {str(e)}",
            "response_time": 0.0,
            "parse_success": False,
            "num_images": 0
        }

    # Build the messages with all images
    user_content = image_contents + [{"type": "text", "text": mc_prompt}]
    print("Prompt : ",mc_prompt)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an expert radiologist. Analyze chest X-rays and answer multiple choice questions. "
                        "When multiple views are provided (e.g., PA and lateral), consider both in your analysis. "
                        "Always respond with valid JSON format only. Do not include any text outside the JSON object."
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    # Apply the chat template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    # Capture the length of the prompt tokens
    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    start_time = time.time()
    with torch.inference_mode():
      generation = model.generate(**inputs, max_new_tokens=150, do_sample=False)
      generation = generation[0][input_len:]

    response_time = time.time() - start_time

    # Decode the output
    decoded = processor.decode(generation, skip_special_tokens=True)
    result = {"model_response" : decoded }
    result['prompt'] = mc_prompt
    result['response_time'] = response_time
    result['full_response'] = decoded
    result['num_images'] = len(image_contents)

    return result




import ast
import os
import time
import json
from IPython.display import clear_output
#folder_path='/content/drive/MyDrive/Health_Data/RexVQA/deid_png_onepx_pneumo'
#folder_path='/home/bsada1/ReXGradient-160K'
folder_path='/home/bsada1/ReXGradient-160K/deid_png_onepx_pneumo'
all_results = []
process_study_id = []

total = len(q_data)
for idx, row in enumerate(q_data.itertuples(index=False), start=1):
    # Print progress
    print(f"Processing {idx}/{total} - study_id: {row.study_id}")

    # Clear output after every two prints to avoid cluttering the notebook
    if idx % 2 == 0:
        clear_output(wait=True)

    question       = row.question
    options_list   = row.options       # assume this is already a list
    correct_answer = row.correct_answer
    study_id       = row.study_id
    image_path     = ast.literal_eval(row.image_path)[0]
    full_path      = folder_path + image_path.replace('../','')

    if study_id in process_study_id:
        print(f"Skipping {study_id}")
        continue
    process_study_id.append(study_id)

    # Generate and parse results
    result = generate_gemma3_multiple_choice(question, options_list, row.image_path,folder_path)
    #print(result)
    parsed = extract_json(result['model_response'])
    if correct_answer != parsed['answer']:
        print(f"Warning: Correct answer {correct_answer} does not match model answer {parsed['answer']}")
    else:
        print(f"✓ Correct answer {correct_answer} matches model answer {parsed['answer']}")

    # Build result dictionary
    result_data = {
        'model_answer':   parsed['answer'],
        'model_expl':     parsed['explanation'],
        'model_conf':     parsed['confidence'],
        'model_finds':    parsed['key_findings'],
        'question':       question,
        'study_id':       study_id,
        'correct_answer': correct_answer,
        'image_path':     full_path,
        'question_type':  'baseline',
        'attack_type':    'baseline',
        'variation_type': 'baseline'
    }
    all_results.append(result_data)


folder_path='/home/bsada1/ReXGradient-160K/deid_png_onepx_pneumo'
all_results = []
process_study_id = []

total = len(q_data)
for idx, row in enumerate(q_data.itertuples(index=False), start=1):
    # Print progress
    print(f"Processing {idx}/{total} - study_id: {row.study_id}")

    # Clear output after every two prints to avoid cluttering the notebook
    if idx % 2 == 0:
        clear_output(wait=True)

    question       = row.question
    options_list   = row.options       # assume this is already a list
    correct_answer = row.correct_answer
    study_id       = row.study_id
    image_path     = ast.literal_eval(row.image_path)[0]
    full_path      = folder_path + image_path.replace('../','')

    if study_id in process_study_id:
        print(f"Skipping {study_id}")
        continue
    process_study_id.append(study_id)

    # Generate and parse results
    result = generate_gemma3_multiple_choice(question, options_list, row.image_path,folder_path)
    #print(result)
    parsed = extract_json(result['model_response'])
    if correct_answer != parsed['answer']:
        print(f"Warning: Correct answer {correct_answer} does not match model answer {parsed['answer']}")
    else:
        print(f"✓ Correct answer {correct_answer} matches model answer {parsed['answer']}")

    # Build result dictionary
    result_data = {
        'model_answer':   parsed['answer'],
        'model_expl':     parsed['explanation'],
        'model_conf':     parsed['confidence'],
        'model_finds':    parsed['key_findings'],
        'question':       question,
        'study_id':       study_id,
        'correct_answer': correct_answer,
        'image_path':     full_path,
        'question_type':  'onepix',
        'attack_type':    'onepix',
        'variation_type': 'onepix'
    }
    all_results.append(result_data)
