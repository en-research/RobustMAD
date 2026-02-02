# evaluation/oe_internvl3_5_4b.py
# Note: done in non-deterministic setting: do_sample=True, temperature=0.2
# max_new_tokens=100

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
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# conda activate internvl_venv
# python evaluation/oe_internvl3_5_4b.py

# nohup python evaluation/oe_internvl3_5_4b.py >> results/output_oe_internvl3_5_4b.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python evaluation/oe_internvl3_5_4b.py >> results/output_oe_internvl3_5_4b.log 2>&1 &

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
MODEL_NAME = "OpenGVLab/InternVL3_5-4B"  # Updated to InternVL3_5-4B as per requirements
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
INPUT_SIZE = 448  # Default input size for InternVL models
MAX_NUM = 12  # Max patches for dynamic preprocessing

# Paths
INPUT_JSON_PATH = "RobustMAD_OE.json" #"RobustMAD_OE_low.json"   # Input dataset JSON for open-ended questions
model_name_short = MODEL_NAME.split("/")[-1].replace("-", "_")

# Suffix determination
if "low" in INPUT_JSON_PATH.lower():  # For low-quality images
    DATASET_SUFFIX = "_low"        
else:
    DATASET_SUFFIX = ""            # For high-quality images

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# RESULTS_FOLDER = os.path.join("results", f"OE_{model_name_short}_s{SEED}_{timestamp}")
RESULTS_FOLDER = os.path.join("results", f"OE_{model_name_short}_s{SEED}")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

OUTPUT_JSON_PATH = os.path.join(RESULTS_FOLDER, f"OE_{model_name_short}_s{SEED}{DATASET_SUFFIX}.json")
PROMPT_LOG_FILE = os.path.join(RESULTS_FOLDER, f"prompt_preview_{model_name_short}.txt")

# Instruction prompt to ensure the model responds concisely
INSTRUCTION = (
    "You are an expert industrial product quality inspector. Given the image(s) and the question below, "
    "provide a clear, concise, accurate, and helpful answer."
    "Answer should be in complete sentences. Avoid bullet points, parentheses, slashes, or Unicode dashes in the text. Colon, semi-colon, and commas if needed are fine."
    "Answer should not be more than 50 words."
)

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
                max_new_tokens=100,
                do_sample=True,
                temperature=0.2
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
# Main execution
# -------------------------------------------------------------------------

def main():
    # Set seeds for reproducibility
    set_seeds(seed=SEED)

    # Load model and tokenizer
    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
        load_in_8bit=False,  # Added as per model card example
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"  # Updated to use device_map="auto" as per model card
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
        specie_name = image_entry['specie_name']  # Used for grouping if needed

        print(f"[INFO] Processing image: {image_path}")

        for question_entry in image_entry['questions']:
            question_id = question_entry['question_id']
            robustness_category = question_entry['robustness_category']
            question = question_entry['question']

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
                output_data.append({
                    "image_path": image_path,
                    "reference_image_path": reference_image_path,
                    "difficulty": difficulty,
                    "robustness_category": robustness_category,
                    "question_id": question_id,
                    "question": question,
                    "mslm_answer": mslm_answer
                })
                progress_bar.update(1)
                continue

            # Construct prompt: Instruction + Question
            full_prompt = (
                f"{INSTRUCTION}\n\n"
                f"Question: {question}"
            )

            # Generate response
            mslm_answer = generate_answer(model, tokenizer, images, full_prompt)

            print(f"[INFO] MSLM answer for QID {question_id}: {mslm_answer}")

            # Collect output entry (no groundtruth 'answer', no 'raw_response')
            output_data.append({
                "image_path": image_path,
                "reference_image_path": reference_image_path,
                "difficulty": difficulty,
                "robustness_category": robustness_category,
                "question_id": question_id,
                "question": question,
                "mslm_answer": mslm_answer
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

if __name__ == "__main__":
    main()