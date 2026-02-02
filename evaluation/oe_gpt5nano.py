# Note: done in non-deterministic setting
# No max_tokens or temperature for gpt-5-nano

import json
import os
import time
import traceback
#from pathlib import Path
#from PIL import Image
import random
import base64
from tqdm import tqdm
#from datetime import datetime
import openai

# python evaluation/oe_gpt5nano.py
# nohup python evaluation/oe_gpt5nano.py >> results/output_oe_gpt5nano.log 2>&1 &

# Set seeds for reproducibility (limited effect on API calls)
def set_seeds(seed=42):
    """Set random seeds for reproducibility across random."""
    random.seed(seed)

# Model configuration
SEED = 0
MODEL_NAME = "gpt-5-nano"  # Adjusted to gpt-5-nano as per request
api_key = ""  
client = openai.OpenAI(api_key=api_key)

# Paths
INPUT_JSON_PATH = "RobustMAD_OE.json" #"RobustMAD_OE_low.json"    # Input dataset JSON for open-ended questions
model_name_short = MODEL_NAME.replace("-", "_")

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

def load_image_base64(path):
    """Load an image and return its base64 encoded string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"[WARNING] Image not found: {path}")
        return None

def get_image_mime(path):
    """Get the MIME type based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    return "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

def save_and_print_prompt_preview(content):
    """Log a preview of the prompt (text and images) to a file and print it."""
    lines = []
    lines.append("=== Prompt preview ===")
    for item in content:
        if item["type"] == "text":
            text_snippet = item["text"][:300] + ("..." if len(item["text"]) > 300 else "")
            lines.append(f"  Text: {text_snippet}")
        elif item["type"] == "image_url":
            lines.append(f"  Image: data URL [not shown]")
    lines.append("=====================\n")
    with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

# -------------------------------------------------------------------------
# Model inference
# -------------------------------------------------------------------------

def generate_answer(image_path, reference_image_path, prompt, robustness_category, retries=3):
    """Generate an answer from the model given image paths and prompt. Handles retries on failure."""
    # Load base64 for images
    query_b64 = load_image_base64(image_path)
    if query_b64 is None:
        return "Error: Query image not found"
    
    query_mime = get_image_mime(image_path)
    
    content = [{"type": "text", "text": prompt}]
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{query_mime};base64,{query_b64}"}
    })
    
    if "Pair-wise comparison" in robustness_category and reference_image_path:
        ref_b64 = load_image_base64(reference_image_path)
        if ref_b64 is None:
            return "Error: Reference image not found"
        ref_mime = get_image_mime(reference_image_path)
        content.insert(1, {
            "type": "image_url",
            "image_url": {"url": f"data:{ref_mime};base64,{ref_b64}"}
        })
        content[0]["text"] = "First image: reference (normal). Second image: query.\n" + content[0]["text"]

    save_and_print_prompt_preview(content)

    messages = [{"role": "user", "content": content}]

    start_time = time.time()

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                verbosity="medium",
                reasoning_effort="medium"
            )
            answer = response.choices[0].message.content.strip()
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f"[INFO] Time for this question: {mins}m {secs}s")
            model_used = getattr(response, "model", "unknown_model")
            print(f"[INFO] Model used: {model_used}")
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

            # Prepare reference if applicable
            reference_image_path = None
            if robustness_category == "Anomaly Understanding and Localization (Pair-wise comparison)":
                reference_image_path = image_entry.get('reference_image_path')

            # Construct prompt: Instruction + Question
            full_prompt = (
                f"{INSTRUCTION}\n\n"
                f"Question: {question}"
            )

            # Generate response
            mslm_answer = generate_answer(image_path, reference_image_path, full_prompt, robustness_category)

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