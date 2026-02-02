# Note: done in deterministic setting if possible, but no temperature or max_tokens for gpt-5-nano

import json
import os
import time
import traceback
import random
import base64
from tqdm import tqdm
import openai
import pandas as pd
import re
from datetime import datetime

# python evaluation/mcq_gpt5nano.py
# nohup python evaluation/mcq_gpt5nano.py >> results/output_mcq_gpt5nano.log 2>&1 &

# Set seeds for reproducibility (limited effect on API calls)
def set_seeds(seed=42):
    """Set random seeds for reproducibility across random."""
    random.seed(seed)

# Model configuration
SEED = 0
MODEL_NAME = "gpt-5-nano"  # Adjusted to gpt-5-nano
api_key = ""  
client = openai.OpenAI(api_key=api_key)

# Paths
INPUT_JSON_PATH = "RobustMAD_MCQ.json" #"RobustMAD_MCQ_low.json"    # Input dataset JSON for multiple-choice questions
model_name_short = MODEL_NAME.replace("-", "_")

# Suffix determination
if "low" in INPUT_JSON_PATH.lower():  # For low-quality images
    DATASET_SUFFIX = "_low"        
else:
    DATASET_SUFFIX = ""            # For high-quality images

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
    with open(PROMPT_LOG_FILE, "a", encoding='utf-8') as f:
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
                verbosity="low",
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
# Accuracy computation and logging
# -------------------------------------------------------------------------

def compute_and_log_accuracy(output_data):
    """Compute accuracies based on the output data and log to CSV in the specified template format."""
    # Create a DataFrame from the output data
    df = pd.DataFrame(output_data)
    
    # Add a column for correctness
    df['correct'] = df.apply(lambda row: row['mslm_answer'] == row['answer'], axis=1)
    
    # Create a combined object-specie_name pair (FIXED: match phi/gemini indexing)
    df['object_specie_pair'] = df['image_path'].apply(lambda x: "-".join([x.split("/")[1], x.split("/")[3]]))  # e.g., "bottle-good"
    
    # Map robustness categories to short forms
    df['robustness_category'] = df['robustness_category'].map(lambda x: ROBUSTNESS_SHORTFORMS.get(x, x))
    
    # Clean questions by removing specified phrase (for pairwise)
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
                    'difficulty': diff_level,  # Shows "general" or "domain"
                    'object_specie_pair': 'Average_by_Difficulty',
                    'num_questions': stats['total_questions'],
                    'question': '',
                    'correct_questions': stats['total_correct'],
                    'incorrect_questions': stats['total_questions'] - stats['total_correct'],
                    'accuracy_per_pair': '',
                    'average_per_category': avg_acc
                })
            else:
                # Optional: Put zero row if a difficulty level has no questions
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

    if os.path.exists(OUTPUT_JSON_PATH):
        print(f"[INFO] Output JSON already exists: {OUTPUT_JSON_PATH}. Skipping generation and computing accuracy directly.")
        output_data = load_json(OUTPUT_JSON_PATH)
        compute_and_log_accuracy(output_data)
        return

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
            options = question_entry['options']
            groundtruth_answer = question_entry['answer']

            print(f"[INFO] Generating answer for QID: {question_id}")

            # Prepare reference if applicable
            reference_image_path = None
            if robustness_category == "Anomaly Understanding and Localization (Pair-wise comparison)":
                reference_image_path = image_entry.get('reference_image_path')

            # Construct prompt: Instruction + Question + Options
            full_prompt = (
                f"{INSTRUCTION}\n\n"
                f"Question: {question}\n"
                f"Options:\n{format_options(options)}"
            )

            # Generate raw response
            raw_response = generate_answer(image_path, reference_image_path, full_prompt, robustness_category)

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