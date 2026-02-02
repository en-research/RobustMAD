# evaluation/mcq_internvl3_5_4b.py
# Note: done in deterministic setting as I cannot find temperature argument in HF card: do_sample=False below
# max_new_tokens=10,  # Short response: just the letter

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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# conda activate internvl35_venv
# python evaluation/mcq_internvl3_5_4b.py
# CUDA_VISIBLE_DEVICES=1 nohup python evaluation/mcq_internvl3_5_4b.py >> results/output_mcq_internvl3_5_4b.log 2>&1 &

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
MODEL_NAME = "OpenGVLab/InternVL3_5-4B"  # Updated to InternVL3.5-4B as per the model card: https://huggingface.co/OpenGVLab/InternVL3_5-4B
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
INPUT_SIZE = 448  # Default input size for InternVL models
MAX_NUM = 12  # Max patches for dynamic preprocessing

# Paths
INPUT_JSON_PATH = "RobustMAD_MCQ.json" #"RobustMAD_MCQ_low.json"     # Input dataset JSON
model_name_short = MODEL_NAME.split("/")[-1].replace("-", "_")

# Suffix determination
if "low" in INPUT_JSON_PATH.lower(): 
    DATASET_SUFFIX = "_low"        
else:
    DATASET_SUFFIX = ""            

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# RESULTS_FOLDER = os.path.join("results", f"{model_name_short}_s{SEED}_{timestamp}")
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

# Image preprocessing functions (adapted from InternVL model card)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """Build transformation pipeline for image preprocessing."""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio for dynamic preprocessing."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamically preprocess image into patches."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def preprocess_image(pil_image, input_size=INPUT_SIZE, max_num=MAX_NUM):
    """Preprocess a single PIL image into tensor patches."""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values).to(DTYPE).to(DEVICE)

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

def generate_answer(model, tokenizer, images, prompt, retries=3):
    """Generate an answer from the model given images and prompt. Handles retries on failure."""
    if not images or (isinstance(images, list) and any(img is None for img in images)):
        return "Error: Image(s) not found"
    
    # Clear unused GPU memory before inference to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pixel_values = None
    num_patches_list = None
    question_prefix = ""
    if isinstance(images, list):
        pixel_values_list = [preprocess_image(img) for img in images if img is not None]
        if not pixel_values_list:
            return "Error: Images not found"
        pixel_values = torch.cat(pixel_values_list, dim=0)
        num_patches_list = [pv.shape[0] for pv in pixel_values_list]
        question_prefix = ''.join(f"Image-{i+1}: <image>\n" for i in range(len(images)))
    else:
        pixel_values = preprocess_image(images)
        num_patches_list = [pixel_values.shape[0]]
        question_prefix = "<image>\n"
    full_prompt = question_prefix + prompt

    save_and_print_prompt_preview(images, full_prompt)

    start_time = time.time()

    for attempt in range(1, retries + 1):
        try:
            generation_config = dict(
                max_new_tokens=10,  # Short response: just the letter
                do_sample=False,    # Deterministic output
            )
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=full_prompt,
                generation_config=generation_config,
                num_patches_list=num_patches_list
            )
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

    # Load model and tokenizer
    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        load_in_8bit=False,  # Added as per model card example
        device_map="auto"    # Updated to use device_map="auto" for better device handling
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)

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

            # Construct prompt: Instruction + Question + Options
            full_prompt = (
                f"{INSTRUCTION}\n\n"
                f"Question: {question}\n"
                f"Options:\n{format_options(options)}"
            )

            # Generate raw response
            raw_response = generate_answer(model, tokenizer, images, full_prompt)
            
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