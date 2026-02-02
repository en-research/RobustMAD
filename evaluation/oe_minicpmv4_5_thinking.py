# Note: done in non-deterministic setting: sampling=True, temperature=0.2 because OE
# max_new_tokens=3000 (to accommodate potential thinking in the response)

import json
import os
import time
import traceback
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datetime import datetime

# python evaluation/oe_minicpmv4_5_thinking.py
# nohup python evaluation/oe_minicpmv4_5_thinking.py >> results/output_oe_minicpmv4_5_thinking.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python evaluation/oe_minicpmv4_5_thinking.py >> results/output_oe_minicpmv4_5_thinking.log 2>&1 &
#conda activate minicpm_venv

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
MODEL_NAME = "openbmb/MiniCPM-V-4_5"  # Adapted to the requested model: https://huggingface.co/openbmb/MiniCPM-V-4_5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

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

# Instruction prompt to ensure the model responds concisely, with thinking and final answer format
INSTRUCTION = (
    "You are an expert industrial product quality inspector. Given the image(s) and the question below, "
    "provide a clear, concise, accurate, and helpful final answer."
    "Think step by step **silently** and concisely about the content and reasoning required before providing the final answer. Then, output the final answer starting with 'Final answer:'."
    "Final answer should be in complete sentences. Avoid bullet points, parentheses, slashes, or Unicode dashes in the text. Colon, semi-colon, and commas if needed are fine."
    "Final answer should not be more than 50 words." 
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

# -------------------------------------------------------------------------
# Model inference
# -------------------------------------------------------------------------

def parse_response(response):
    """
    Parse the response to extract the final answer.
    First, try to find 'Final answer:' and take the part after the last one.
    If not found, fall back to thinking tag parsing: take after the last '</think>'.
    If neither is found, return the full response as fallback.
    """
    if 'Final answer:' in response:
        # Split on the last 'Final answer:' and take the part after it
        parts = response.rsplit('Final answer:', 1)
        if len(parts) > 1:
            return parts[1].strip()
    elif '</think>' in response:
        # Split on the last '</think>' and take the part after it
        parts = response.rsplit('</think>', 1)
        if len(parts) > 1:
            return parts[1].strip()
    return response.strip()  # Fallback to full response if parsing fails

def generate_answer(model, tokenizer, images, prompt, retries=3):
    """Generate an answer from the model given images and prompt. Handles retries on failure."""
    if not images or (isinstance(images, list) and any(img is None for img in images)):
        return "Error: Image(s) not found", "Error: Image(s) not found"
    
    # Clear unused GPU memory before inference to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Content is a list: images (PIL) + prompt string
    content = images if isinstance(images, list) else [images]
    content.append(prompt)

    save_and_print_prompt_preview(images, prompt)

    msgs = [{'role': 'user', 'content': content}]

    start_time = time.time()

    for attempt in range(1, retries + 1):
        try:
            raw_answer = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,  # Non-deterministic output
                temperature=0.2,
                max_new_tokens=3000  # To accommodate thinking and answer
            )
            # Parse the final answer using the dual method
            parsed_answer = parse_response(raw_answer)
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f"[INFO] Time for this question: {mins}m {secs}s")
            # Clear cache after successful inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return parsed_answer, raw_answer  # Return both parsed and raw for output
        except Exception as e:
            print(f"[ERROR] Model call failed (attempt {attempt}/{retries}): {str(e)}")
            traceback.print_exc()
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                # Clear cache even on failure to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return "Error: Model call failed", "Error: Model call failed"

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
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=DTYPE
    ).to(device=DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

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
                    raw_mslm_answer = "Error: Images not found"
            else:
                query_img = load_image(image_path)
                if query_img:
                    images = [query_img]  # Single image
                else:
                    mslm_answer = "Error: Image not found"
                    raw_mslm_answer = "Error: Image not found"

            if images is None:
                output_data.append({
                    "image_path": image_path,
                    "reference_image_path": reference_image_path,
                    "difficulty": difficulty,
                    "robustness_category": robustness_category,
                    "question_id": question_id,
                    "question": question,
                    "mslm_answer": mslm_answer,
                    "raw_mslm_answer": raw_mslm_answer
                })
                progress_bar.update(1)
                continue

            # Construct prompt: Instruction + Question
            full_prompt = (
                f"{INSTRUCTION}\n\n"
                f"Question: {question}"
            )

            # Generate response
            mslm_answer, raw_mslm_answer = generate_answer(model, tokenizer, images, full_prompt)

            print(f"[INFO] MSLM answer for QID {question_id}: {mslm_answer}")
            print(f"[INFO] Raw MSLM answer for QID {question_id}: {raw_mslm_answer}")

            # Collect output entry (no groundtruth 'answer')
            output_data.append({
                "image_path": image_path,
                "reference_image_path": reference_image_path,
                "difficulty": difficulty,
                "robustness_category": robustness_category,
                "question_id": question_id,
                "question": question,
                "mslm_answer": mslm_answer,
                "raw_mslm_answer": raw_mslm_answer
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