"""
Enhanced parsing to extract the most likely answer letter A-E from the response since Gemini doesnt follow instructions to output answer as letter sometimes.
Collects all candidate matches from various patterns and selects the last one by position.

Will replace input MCQ_gemini3flash_s0(_low).json with correctly parsed mslm_answer

"""

# python results/gemini3flash_s0/robust_parsing_gemini3flash.py results/gemini3flash_s0/MCQ_gemini3flash_s0.json
# python results/gemini3flash_s0/robust_parsing_gemini3flash.py results/gemini3flash_s0/MCQ_gemini3flash_s0_low.json

import json
import re
import os
import argparse

# Improved parsing function
def improved_parse_model_response(response):
    """
    Enhanced parsing to extract the most likely answer letter A-E from the response.
    Collects all candidate matches from various patterns and fallback selects the last one by position.
    """
    if not response:
        return "Error: Empty response"
    
    text = response.strip().upper()
    
    # Extended list of patterns to match potential answer indicators
    patterns = [
        r'\b([A-E])\b',  # Standalone A-E
        r'\b([A-E])(?=[\.\,\)\]\}\>\:\s]|$)',  # A-E followed by punctuation or end
        r'(?:ANSWER|OPTION|CHOICE|ASSESSMENT)[\s\:]*([A-E])\b',  # After keywords
        r'\[([A-E])\]',  # In brackets
        r'\*\*([A-E])\*\*',  # In bold markdown
        r'^([A-E])\s',  # Starting with A-E
        r'FINAL\s+ASSESSMENT:\s*([A-E])',  # Specific to "Final assessment: D"
        r'([A-E])[\.\s]*$'  # A-E at the very end, optionally followed by dot/space
    ]
    
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            # Append (start position, captured group)
            candidates.append((match.start(1), match.group(1)))
    
    if candidates:
        # Sort by position ascending and take the last (farthest in text)
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    
    # Improved fallback: Find all standalone A-E and take the last one
    standalone_matches = re.findall(r'\b([A-E])\b', text)
    if standalone_matches:
        return standalone_matches[-1]
    
    # Ultimate fallback: Last occurrence of A-E in the text
    for i in range(len(text) - 1, -1, -1):
        if text[i] in 'ABCDE':
            return text[i]
    
    return "Error: Invalid response"

# Script to update the JSON
def update_json(json_path):
    """
    Load the JSON, update 'mslm_answer' for each entry using improved parsing,
    and save back to the same file.
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] File not found: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    updated_count = 0
    for entry in data:
        raw_response = entry.get('raw_response', '')
        old_mslm = entry.get('mslm_answer', '')
        new_mslm = improved_parse_model_response(raw_response)
        if new_mslm != old_mslm:
            print(f"[INFO] Updating QID {entry.get('question_id', 'unknown')}: {old_mslm} -> {new_mslm}")
            entry['mslm_answer'] = new_mslm
            updated_count += 1
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"[INFO] Updated {updated_count} entries in {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix mslm_answer in JSON file.")
    parser.add_argument("json_path", type=str, help="Path to the JSON file (e.g., MCQ_gemini3flash_s0.json or MCQ_gemini3flash_s0_low.json)")
    args = parser.parse_args()
    
    update_json(args.json_path)