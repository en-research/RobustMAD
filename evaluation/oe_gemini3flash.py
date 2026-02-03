# evaluation/oe_gemini3flash.py
# Note: Google recommends temperature =1.0 for Gemini3 https://ai.google.dev/gemini-api/docs/models?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content=#gemini-3-flash-preview, so just leave it as default, nothing set

import json
import os
import time
import traceback
import random
from tqdm import tqdm
from datetime import datetime
from google import genai as genai  # Corrected import alias to match usage
from google.genai import types

# conda activate gemini_venv
# python evaluation/oe_gemini3flash.py
# nohup python evaluation/oe_gemini3flash.py >> results/output_oe_gemini3flash.log 2>&1 &

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set random seeds for reproducibility across random."""
    random.seed(seed)

# Model configuration
SEED = 0
MODEL_NAME = "gemini-3-flash-preview"  # Based on Gemini 3 Flash preview model name from documentation
api_key = ""  # Set your Google API key here

# Paths
INPUT_JSON_PATH = "RobustMAD_OE.json" #"RobustMAD_OE_low.json"   # Input dataset JSON for open-ended questions
model_name_short = "gemini3flash"

# Suffix determination
if "low" in INPUT_JSON_PATH.lower():  # For low-quality images
    DATASET_SUFFIX = "_low"        
else:
    DATASET_SUFFIX = ""            # For high-quality images

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
    """Load an image as bytes and return (bytes, mime_type)."""
    try:
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        with open(image_path, 'rb') as f:
            return f.read(), mime
    except FileNotFoundError:
        print(f"[WARNING] Image not found: {image_path}")
        return None, None

def save_and_print_prompt_preview(text, image_info):
    """Log the prompt preview to a file and print it."""
    lines = []
    lines.append("=== Prompt preview ===")
    text_snippet = text[:300] + ("..." if len(text) > 300 else "")
    lines.append(f"  Text: {text_snippet}")
    if isinstance(image_info, list):
        for i, (mime) in enumerate(image_info, 1):
            lines.append(f"  Image {i}: {mime} bytes [not shown]")
    else:
        mime = image_info
        lines.append(f"  Image: {mime} bytes [not shown]")
    lines.append("=====================\n")
    with open(PROMPT_LOG_FILE, "a", encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

# -------------------------------------------------------------------------
# Model inference
# -------------------------------------------------------------------------

def generate_answer(client, images, prompt, retries=3):
    """Generate an answer from the model given images and prompt. Handles retries on failure."""
    if images is None:
        return "Error: Image(s) not found"

    # Prepare contents
    if isinstance(images, tuple):
        # Single image
        img_bytes, mime = images
        if img_bytes is None:
            return "Error: Image not found"
        parts = [
            types.Part(text=prompt),
            types.Part(inline_data=types.Blob(mime_type=mime, data=img_bytes))
        ]
        contents = [types.Content(parts=parts)]
        image_info = mime
    else:
        # Pairwise images: list of tuples
        if any(b is None for b, m in images):
            return "Error: Images not found"
        parts = [types.Part(text="First image: reference (normal). Second image: query.\n" + prompt)]
        for img_bytes, mime in images:
            parts.append(types.Part(inline_data=types.Blob(mime_type=mime, data=img_bytes)))
        contents = [types.Content(parts=parts)]
        image_info = [m for b, m in images]

    # Log prompt preview
    save_and_print_prompt_preview(prompt, image_info)

    start_time = time.time()

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="medium")
                )
            )
            answer = response.text.strip()
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f"[INFO] Time for this question: {mins}m {secs}s")
            return answer
        except Exception as e:
            print(f"[ERROR] Model call failed (attempt {attempt}/{retries}): {str(e)}")
            traceback.print_exc()
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return "Error: Model call failed"

# -------------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------------

def main():
    # Set seeds for reproducibility
    set_seeds(seed=SEED)

    # Create client
    print(f"[INFO] Creating client for model: {MODEL_NAME}")
    client = genai.Client(api_key=api_key)

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
                reference_image_path = image_entry.get('reference_image_path')
                ref_data = load_image(reference_image_path)
                query_data = load_image(image_path)
                if ref_data[0] is not None and query_data[0] is not None:
                    images = [ref_data, query_data]  # List of (bytes, mime)
                else:
                    mslm_answer = "Error: Images not found"
            else:
                query_data = load_image(image_path)
                if query_data[0] is not None:
                    images = query_data  # Tuple (bytes, mime)
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
            mslm_answer = generate_answer(client, images, full_prompt)

            print(f"[INFO] MSLM answer for QID {question_id}: {mslm_answer}")

            # Collect output entry
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