# evaluation/mcq_qwen3_vl_4b.py
# Deterministic generation: do_sample=False, temperature=0
# max_new_tokens=10 for short response (just the letter)
# Tips from https://github.com/QwenLM/Qwen3-VL : 
# - "image_patch_size": 14 for Qwen2.5-VL and 16 for Qwen3-VL. Default set to 14.
# - Note: Since qwen-vl-utils already resizes images/videos, pass do_resize=False to the processor to avoid duplicate resizing.


import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from PIL import Image
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info  # Required for vision input processing as per model card

# python evaluation/mcq_qwen3_vl_4b.py
# CUDA_VISIBLE_DEVICES=1 nohup python evaluation/mcq_qwen3_vl_4b.py >> results/output_mcq_qwen3_vl_4b.log 2>&1 &
# conda activate qwen3vl_venv

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)

# Model configuration
SEED = 0
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"  # As per the model card: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Paths
INPUT_JSON_PATH = "RobustMAD_MCQ.json" #"RobustMAD_MCQ_low.json"      # Input dataset JSON
model_name_short = MODEL_NAME.split("/")[-1].replace("-", "_")

# Suffix determination
if "low" in INPUT_JSON_PATH.lower(): 
    DATASET_SUFFIX = "_low"        
else:
    DATASET_SUFFIX = ""           

#timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#RESULTS_FOLDER = os.path.join("results", f"{model_name_short}_s{SEED}_{timestamp}")
RESULTS_FOLDER = os.path.join("results", f"{model_name_short}_s{SEED}")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

OUTPUT_JSON_PATH = os.path.join(RESULTS_FOLDER, f"MCQ_{model_name_short}_s{SEED}{DATASET_SUFFIX}.json")
ACCURACY_CSV_PATH = os.path.join(RESULTS_FOLDER, f"accuracy_{model_name_short}_s{SEED}{DATASET_SUFFIX}.csv")
PROMPT_LOG_FILE = os.path.join(RESULTS_FOLDER, f"prompt_preview_{model_name_short}.txt")

# Instruction prompt to ensure the model responds with only the letter
INSTRUCTION = (
    "You are an expert industrial product quality inspector. Given the image(s) and the question below, select the single correct option "
    "and return only the corresponding letter (A, B, C, D, or E). No other open-ended output is allowed."
)

# Short-form mapping for robustness categories
ROBUSTNESS_SHORTFORMS = {
    "General Object Understanding": "general",
    "Anomaly Understanding and Localization (Stand-alone)": "standalone",
    "Anomaly Understanding and Localization (Pair-wise comparison)": "pairwise",
    "Unanswerable or Misleading Query Detection": "unanswerable"
}

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def load_json(file_path):
    """Load JSON data from the given file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, output_path):
    """Save data as JSON to the given output path."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_image(image_path):
    """Load an image from the given path, converting to RGB."""
    try:
        return Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"[WARNING] Image not found: {image_path}")
        return None

def format_options(options):
    """Format the options dictionary into a string (e.g., A: text\nB: text)."""
    return "\n".join([f"{key}: {value}" for key, value in options.items()])

def parse_model_response(response):
    """Fool-proof extraction of answer choice A-E from model response."""
    if not response:
        return "Error: Empty response"
    
    text = response.strip().upper()
    
    # Comprehensive patterns
    patterns = [
        r'\b([A-E])\b',
        r'\b([A-E])(?=[\.\,\)\]\}\>\:\s]|$)',
        r'(?:answer|option|choice)[\s\:]*([A-E])\b',
        r'\[([A-E])\]',
        r'\*\*([A-E])\*\*',
        r'^([A-E])\s',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    # Last resort
    fallback = re.search(r'[A-E]', text)
    return fallback.group(0) if fallback else "Error: Invalid response"

def save_and_print_prompt_preview(images, full_prompt):
    """Log a preview of the prompt (images and text) to a file and print it."""
    lines = []
    lines.append("=== Prompt preview ===")
    for i, img in enumerate(images if isinstance(images, list) else [images]):
        if isinstance(img, Image.Image):
            lines.append(f"  Image {i+1}: PIL.Image object [not shown]")
        else:
            lines.append(f"  Image {i+1}: [Invalid or None]")
    text_snippet = full_prompt[:300] + ("..." if len(full_prompt) > 300 else "")
    lines.append(f"  Text: {text_snippet}")
    lines.append("=====================\n")
    with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

# -------------------------------------------------------------------------
# Model inference
# -------------------------------------------------------------------------

def generate_answer(model, processor, images, prompt, retries=3):
    """Generate an answer from the model given images and prompt. Handles retries on failure."""
    if not images or (isinstance(images, list) and any(img is None for img in images)):
        return "Error: Image(s) not found"
    
    # Clear unused GPU memory before inference to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Construct the full prompt text (instruction + question + options)
    full_prompt = (
        f"{INSTRUCTION}\n\n"
        f"Question: {prompt}\n"
    )

    # Prepare messages: user role with content list (images + text)
    content = []
    if isinstance(images, list):
        for img in images:
            content.append({"type": "image", "image": img})
    else:
        content.append({"type": "image", "image": images})
    content.append({"type": "text", "text": full_prompt})

    messages = [{"role": "user", "content": content}]

    save_and_print_prompt_preview(images, full_prompt)

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process vision info (handles image inputs)
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16) #aa video_inputs=None if only images
    #print("video_inputs      :", video_inputs)
    
    start_time = time.time()

    for attempt in range(1, retries + 1):
        try:
            # Tokenize and prepare inputs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                do_resize=False,  
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(DEVICE)

            # Generate (deterministic)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=10,  # Short response: just the letter
                do_sample=False,
                temperature=0.0
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f"[INFO] Time for this question: {mins}m {secs}s")
            # Clear cache after successful inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return response.strip()
        except Exception as e:
            print(f"[ERROR] Model call failed (attempt {attempt}/{retries}): {str(e)}")
            traceback.print_exc()
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                # Clear cache even on failure to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return "Error: Model call failed"

# -------------------------------------------------------------------------
# Accuracy computation and logging
# -------------------------------------------------------------------------

def compute_and_log_accuracy(output_data):
    """Compute accuracies based on the output data and log to CSV in the specified template format."""
    # Create a DataFrame from the output data
    df = pd.DataFrame(output_data)
    
    # Add a column for correctness
    df['correct'] = df.apply(lambda row: row['mslm_answer'] == row['answer'], axis=1)
    
    # Create a combined object-specie_name pair
    df['object_specie_pair'] = df['image_path'].apply(lambda x: "-".join([x.split("/")[1], x.split("/")[3]]))  # e.g., "bottle-good"
    
    # Map robustness categories to short forms
    df['robustness_category'] = df['robustness_category'].map(lambda x: ROBUSTNESS_SHORTFORMS.get(x, x))
    
    # Clean questions by removing specified phrase
    phrase_to_remove = "The first image shows a normal, defect-free reference object. The second image shows the query object for inspection. "
    df['question'] = df['question'].str.replace(phrase_to_remove, "", regex=False)
    
    # Group by robustness_category, object_specie_pair, question
    grouped = df.groupby(['robustness_category', 'object_specie_pair', 'question'])
    
    accuracy_data = []
    category_averages = {}
    
    for (robustness_category, object_specie_pair, question), group in grouped:
        num_questions = len(group)
        correct_count = group['correct'].sum()
        incorrect_count = num_questions - correct_count
        accuracy = correct_count / num_questions if num_questions > 0 else 0
        
        difficulty = group['difficulty'].unique()[0]
        
        accuracy_data.append({
            'robustness_category': robustness_category,
            'difficulty': difficulty,
            'object_specie_pair': object_specie_pair,
            'num_questions': num_questions,
            'question': question,
            'correct_questions': correct_count,
            'incorrect_questions': incorrect_count,
            'accuracy_per_pair': accuracy,
            'average_per_category': None  # To be filled later
        })
        
        # Accumulate for category average
        if robustness_category not in category_averages:
            category_averages[robustness_category] = {'total_correct': 0, 'total_questions': 0}
        category_averages[robustness_category]['total_correct'] += correct_count
        category_averages[robustness_category]['total_questions'] += num_questions
    
    # Step 1: Compute Overall average per robustness category
    for cat, stats in category_averages.items():
        avg_accuracy = stats['total_correct'] / stats['total_questions'] if stats['total_questions'] > 0 else 0
        # Append a summary row for each category
        accuracy_data.append({
            'robustness_category': cat,
            'difficulty': '',
            'object_specie_pair': 'Average',
            'num_questions': stats['total_questions'],
            'question': '',
            'correct_questions': stats['total_correct'],
            'incorrect_questions': stats['total_questions'] - stats['total_correct'],
            'accuracy_per_pair': '',
            'average_per_category': avg_accuracy
        })

    # Step 2: Compute Average per robustness category × difficulty ("general" / "domain")
    # Group by robustness_category AND difficulty for subcategory averages
    difficulty_stats = {}
    for (robustness_category, difficulty), group in df.groupby(['robustness_category', 'difficulty']):
        total_q = len(group)
        correct_q = group['correct'].sum()
        
        if robustness_category not in difficulty_stats:
            difficulty_stats[robustness_category] = {}
        difficulty_stats[robustness_category][difficulty] = {
            'total_questions': total_q,
            'total_correct': correct_q
        }

    # Append rows for each (category, difficulty) combination
    for cat in sorted(difficulty_stats.keys()):
        for diff_level in ['general', 'domain']:  # Ensure consistent order
            if diff_level in difficulty_stats[cat]:
                stats = difficulty_stats[cat][diff_level]
                avg_acc = stats['total_correct'] / stats['total_questions'] if stats['total_questions'] > 0 else 0
                accuracy_data.append({
                    'robustness_category': cat,
                    'difficulty': diff_level,               # Shows "general" or "domain"
                    'object_specie_pair': 'Average_by_Difficulty',
                    'num_questions': stats['total_questions'],
                    'question': '',
                    'correct_questions': stats['total_correct'],
                    'incorrect_questions': stats['total_questions'] - stats['total_correct'],
                    'accuracy_per_pair': '',
                    'average_per_category': avg_acc
                })
            else:
                # Optional: put zero row if a difficulty level has no questions
                accuracy_data.append({
                    'robustness_category': cat,
                    'difficulty': diff_level,
                    'object_specie_pair': 'Average_by_Difficulty',
                    'num_questions': 0,
                    'question': '',
                    'correct_questions': 0,
                    'incorrect_questions': 0,
                    'accuracy_per_pair': '',
                    'average_per_category': 0.0
                })
    
    # Save to CSV
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df.to_csv(ACCURACY_CSV_PATH, index=False)
    print(f"[INFO] Accuracy logged to {ACCURACY_CSV_PATH}")

# -------------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------------

def main():
    # Set seeds for reproducibility
    set_seeds(seed=SEED)

    # Load model and processor
    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load dataset
    dataset = load_json(INPUT_JSON_PATH)['images']  # The JSON has an 'images' list

    output_data = []  # List to collect all processed entries
    total_questions = sum(len(entry['questions']) for entry in dataset)
    progress_bar = tqdm(total=total_questions, desc="Processing questions")

    total_start = time.time()

    for image_entry in dataset:
        image_path = image_entry['image_path']
        difficulty = image_entry['difficulty']
        specie_name = image_entry['specie_name']  # Used for grouping

        print(f"[INFO] Processing image: {image_path}")

        for question_entry in image_entry['questions']:
            question_id = question_entry['question_id']
            robustness_category = question_entry['robustness_category']
            question = question_entry['question']
            options = question_entry['options']
            groundtruth_answer = question_entry['answer']

            print(f"[INFO] Generating answer for QID: {question_id}")

            # Prepare images
            images = None
            reference_image_path = None
            if robustness_category == "Anomaly Understanding and Localization (Pair-wise comparison)":
                reference_image_path = image_entry['reference_image_path']
                ref_img = load_image(reference_image_path)
                query_img = load_image(image_path)
                if ref_img and query_img:
                    images = [ref_img, query_img]  # Reference first, then query
                else:
                    mslm_answer = "Error: Images not found"
            else:
                query_img = load_image(image_path)
                if query_img:
                    images = query_img  # Single image
                else:
                    mslm_answer = "Error: Image not found"

            if images is None:
                raw_response = "Error: Images not found"
                mslm_answer = "Error: Images not found"
                output_data.append({
                    "image_path": image_path,
                    "reference_image_path": reference_image_path,
                    "difficulty": difficulty,
                    "robustness_category": robustness_category,
                    "question_id": question_id,
                    "question": question,
                    "answer": groundtruth_answer,
                    "mslm_answer": mslm_answer,
                    "raw_response": raw_response
                })
                progress_bar.update(1)
                continue

            # Construct prompt: Question + Options (full_prompt includes instruction; passed as 'prompt' arg to generate_answer)
            prompt = (
                f"{question}\n"
                f"Options:\n{format_options(options)}"
            )

            # Generate raw response
            raw_response = generate_answer(model, processor, images, prompt)
            
            # Parse to get the letter
            mslm_answer = parse_model_response(raw_response)

            print(f"[INFO] MSLM answer for QID {question_id}: {mslm_answer}")

            # Collect output entry
            output_data.append({
                "image_path": image_path,
                "reference_image_path": reference_image_path,
                "difficulty": difficulty,
                "robustness_category": robustness_category,
                "question_id": question_id,
                "question": question,
                "answer": groundtruth_answer,
                "mslm_answer": mslm_answer,
                "raw_response": raw_response
            })

            # Incremental save after each question
            save_json(output_data, OUTPUT_JSON_PATH)
            progress_bar.update(1)

    progress_bar.close()

    total_elapsed = time.time() - total_start
    total_mins, total_secs = divmod(int(total_elapsed), 60)
    print(f"[INFO] Total processing time: {total_mins}m {total_secs}s")

    # Save final output JSON
    save_json(output_data, OUTPUT_JSON_PATH)
    print(f"[INFO] Output saved to {OUTPUT_JSON_PATH}")

    # Compute and log accuracy to CSV
    compute_and_log_accuracy(output_data)

if __name__ == "__main__":
    main()