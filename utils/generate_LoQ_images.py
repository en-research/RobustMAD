
import cv2
import numpy as np
import os
import json
import random
from collections import Counter


# python utils/generate_LoQ_anomaly_dataset_Step2.py
# python utils/generate_LoQ_images.py

# Set random seed for reproducibility (optional; uncomment if needed)
random.seed(42)

# Set parameters
# - alpha: contrast factor (0.5 to 1.0 for mild decrease; lower values decrease contrast more)
# - beta: brightness offset (-50 to 0 for mild darkening; more negative for darker)
# - kernel_size: size of the blur kernel (3 to 7 for mild blur; larger for stronger blur)

applied_alpha = 0.6
applied_beta = -10
applied_kernel_size = 12

# Load the JSON file containing the image pairs
with open('selected_images_vetted.json', 'r') as f:
    data = json.load(f)

# Extract the list of image pairs
image_pairs = data['images']

# Number of pairs
num_pairs = len(image_pairs)

# Define the two perturbation types
perturbations = ['light', 'blur']

# Create a list of perturbations to assign to pairs, ensuring approximately half-half distribution
# If num_pairs is odd, one perturbation will have one more pair than the other
half_count = num_pairs // 2
# Assign more to 'light' if odd
assigned_perturbations = (
    ['light'] * (half_count + (num_pairs % 2)) + 
    ['blur'] * half_count
)
# Shuffle to randomize assignment
random.shuffle(assigned_perturbations)

# Function to apply low-light perturbation (mild reduction in brightness and contrast)
# Parameters:
# - alpha: contrast factor (0.5 to 1.0 for mild decrease; lower values decrease contrast more)
# - beta: brightness offset (-50 to 0 for mild darkening; more negative for darker)
def apply_low_light(image, alpha=0.8, beta=-30):
    # Apply contrast and brightness adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# Function to apply motion blur perturbation (mild directional blur)
# Parameters:
# - kernel_size: size of the blur kernel (3 to 7 for mild blur; larger for stronger blur)
# - direction: 'horizontal' or 'vertical' for motion direction
def apply_motion_blur(image, kernel_size=5, direction='horizontal'):
    # Create a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    if direction == 'horizontal':
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    elif direction == 'vertical':
        kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")
    kernel /= kernel_size
    # Apply the filter
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

# List to store the new low-quality image metadata
low_images = []

# Counter for summarization: (category, perturbation) -> count
category_pert_counts = Counter()

# List to collect new query-reference path pairs for summary
path_pairs = []

# Process each selected pair
for i, pair in enumerate(image_pairs):
    pert = assigned_perturbations[i]
    print(f"Processing pair {i+1} with perturbation: {pert}")
    
    # Process the query image
    query_path = pair['image_path']
    if not os.path.exists(query_path):
        print(f"Warning: Query image not found at {query_path}. Skipping.")
        continue
    query_dir = os.path.dirname(query_path)
    new_query_dir = os.path.join(query_dir, pert)
    os.makedirs(new_query_dir, exist_ok=True)
    
    # Create new image_id with suffix before extension
    query_base, query_ext = os.path.splitext(os.path.basename(query_path))
    new_query_id = f"{query_base}_{pert}{query_ext}"
    new_query_path = os.path.join(new_query_dir, new_query_id)
    
    query_img = cv2.imread(query_path)
    if query_img is None:
        print(f"Error: Failed to load query image at {query_path}. Skipping.")
        continue
    if pert == 'light':
        low_quality_query = apply_low_light(query_img, alpha=applied_alpha, beta=applied_beta)  # Tweak alpha/beta here
    elif pert == 'blur':
        low_quality_query = apply_motion_blur(query_img, kernel_size=applied_kernel_size, direction='horizontal')  # Tweak kernel_size/direction here
    cv2.imwrite(new_query_path, low_quality_query)
    print(f"Saved low-quality query image to: {new_query_path}")
    
    # Process the reference image
    ref_path = pair['reference_image_path']
    if not os.path.exists(ref_path):
        print(f"Warning: Reference image not found at {ref_path}. Skipping.")
        continue
    ref_dir = os.path.dirname(ref_path)
    new_ref_dir = os.path.join(ref_dir, pert)
    os.makedirs(new_ref_dir, exist_ok=True)
    
    # Create new reference_image_id with suffix before extension
    ref_base, ref_ext = os.path.splitext(os.path.basename(ref_path))
    new_ref_id = f"{ref_base}_{pert}{ref_ext}"
    new_ref_path = os.path.join(new_ref_dir, new_ref_id)
    
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"Error: Failed to load reference image at {ref_path}. Skipping.")
        continue
    if pert == 'light':
        low_quality_ref = apply_low_light(ref_img, alpha=applied_alpha, beta=applied_beta)  # Tweak alpha/beta here
    elif pert == 'blur':
        low_quality_ref = apply_motion_blur(ref_img, kernel_size=applied_kernel_size, direction='horizontal')  # Tweak kernel_size/direction here
    cv2.imwrite(new_ref_path, low_quality_ref)
    print(f"Saved low-quality reference image to: {new_ref_path}")
    
    # Create new metadata entry
    new_entry = pair.copy()
    new_entry['image_id'] = new_query_id
    new_entry['image_path'] = new_query_path
    new_entry['reference_image_id'] = new_ref_id
    new_entry['reference_image_path'] = new_ref_path
    new_entry['perturbation_type'] = pert
    # Text paths remain the same (point to original high-quality txt files)
    low_images.append(new_entry)
    
    # Update counters and path pairs
    category_pert_counts[(pair['category'], pert)] += 1
    path_pairs.append(f"Query: {new_query_path} | Reference: {new_ref_path}")

# Save the new low-quality metadata to JSON
low_data = {"images": low_images}
with open('selected_images_low.json', 'w') as f:
    json.dump(low_data, f, indent=4)

# Append summary to existing summary.txt
with open('summary.txt', 'a') as f:
    f.write("\nLow-Quality Images Summary:\n")
    
    # Total counts
    total_light = sum(count for (cat, p), count in category_pert_counts.items() if p == 'light')
    total_blur = sum(count for (cat, p), count in category_pert_counts.items() if p == 'blur')
    f.write(f"Total low-light images: {total_light}\n")
    f.write(f"Total motion-blurred images: {total_blur}\n")
    
    # Per category
    f.write("Per category:\n")
    for cat in sorted(set(cat for (cat, p) in category_pert_counts)):
        light_count = category_pert_counts.get((cat, 'light'), 0)
        blur_count = category_pert_counts.get((cat, 'blur'), 0)
        f.write(f"  {cat}: {light_count} light, {blur_count} blur\n")
    
    # Query-reference path pairs
    f.write("\nLow-Quality Query and Reference Image Path Pairs:\n")
    for path_pair in path_pairs:
        f.write(f"{path_pair}\n")

print("Processing complete. Low-quality JSON and summary updated.")














#Trial code on 2 random query-reference image pairs to tune suitable parameters for lightinng and motion blur

# import cv2
# import numpy as np
# import os
# import json
# import random

# # Set random seed for reproducibility
# random.seed(42)  # Ensures consistent random selection across runs

# # Load the JSON file containing the image pairs
# # Assumption: The JSON file 'selected_images.json' is in the current working directory
# with open('selected_images.json', 'r') as f:
#     data = json.load(f)

# # Extract the list of image pairs
# image_pairs = data['images']

# # Number of pairs to select (can be changed as needed)
# num_pairs = 2
# # Select random pairs for prototyping/testing
# selected_pairs = random.sample(image_pairs, num_pairs)

# # Define the two perturbation types
# perturbations = ['light', 'blur']

# # Parameters:
# # - alpha: contrast factor (0.5 to 1.0 for mild decrease; lower values decrease contrast more)
# # - beta: brightness offset (-50 to 0 for mild darkening; more negative for darker)

# # - kernel_size: size of the blur kernel (3 to 7 for mild blur; larger for stronger blur)
# # - direction: 'horizontal' or 'vertical' for motion direction

# applied_alpha = 0.6
# applied_beta = -20
# applied_kernel_size = 12

# # Create a list of perturbations to assign to pairs, ensuring half-half distribution
# # If num_pairs is odd, one perturbation will have one more pair than the other
# half_count = num_pairs // 2
# # Extend perturbations list to match num_pairs, balancing 'light' and 'blur'
# assigned_perturbations = (
#     ['light'] * (half_count + (num_pairs % 2)) +  # Extra 'light' if odd
#     ['blur'] * half_count
# )
# # Shuffle the perturbations to randomize which pairs get which perturbation
# random.shuffle(assigned_perturbations)

# # Function to apply low-light perturbation (mild reduction in brightness and contrast)
# # Parameters:
# # - alpha: contrast factor (0.5 to 1.0 for mild decrease; lower values decrease contrast more)
# # - beta: brightness offset (-50 to 0 for mild darkening; more negative for darker)
# def apply_low_light(image, alpha=0.8, beta=-30):
#     # Apply contrast and brightness adjustment
#     adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
#     return adjusted

# # Function to apply motion blur perturbation (mild directional blur)
# # Parameters:
# # - kernel_size: size of the blur kernel (3 to 7 for mild blur; larger for stronger blur)
# # - direction: 'horizontal' or 'vertical' for motion direction
# def apply_motion_blur(image, kernel_size=5, direction='horizontal'):
#     # Create a motion blur kernel
#     kernel = np.zeros((kernel_size, kernel_size))
#     if direction == 'horizontal':
#         kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
#     elif direction == 'vertical':
#         kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
#     else:
#         raise ValueError("Direction must be 'horizontal' or 'vertical'")
#     kernel /= kernel_size
#     # Apply the filter
#     blurred = cv2.filter2D(image, -1, kernel)
#     return blurred

# # Process each selected pair
# for i, pair in enumerate(selected_pairs):
#     pert = assigned_perturbations[i]  # Use the pre-assigned perturbation
#     print(f"Processing pair {i+1} with perturbation: {pert}")
    
#     # Process the query image
#     query_path = pair['image_path']
#     if not os.path.exists(query_path):
#         print(f"Warning: Query image not found at {query_path}. Skipping.")
#         continue
#     query_dir = os.path.dirname(query_path)
#     new_query_dir = os.path.join(query_dir, pert)
#     os.makedirs(new_query_dir, exist_ok=True)
#     new_query_path = os.path.join(new_query_dir, os.path.basename(query_path))
    
#     query_img = cv2.imread(query_path)
#     if query_img is None:
#         print(f"Error: Failed to load query image at {query_path}. Skipping.")
#         continue
#     if pert == 'light':
#         low_quality_query = apply_low_light(query_img, alpha=applied_alpha, beta=applied_beta)  # Tweak alpha/beta here
#     elif pert == 'blur':
#         low_quality_query = apply_motion_blur(query_img, kernel_size=applied_kernel_size, direction='horizontal')  # Tweak kernel_size/direction here
#     cv2.imwrite(new_query_path, low_quality_query)
#     print(f"Saved low-quality query image to: {new_query_path}")
    
#     # Process the reference image
#     ref_path = pair['reference_image_path']
#     if not os.path.exists(ref_path):
#         print(f"Warning: Reference image not found at {ref_path}. Skipping.")
#         continue
#     ref_dir = os.path.dirname(ref_path)
#     new_ref_dir = os.path.join(ref_dir, pert)
#     os.makedirs(new_ref_dir, exist_ok=True)
#     new_ref_path = os.path.join(new_ref_dir, os.path.basename(ref_path))
    
#     ref_img = cv2.imread(ref_path)
#     if ref_img is None:
#         print(f"Error: Failed to load reference image at {ref_path}. Skipping.")
#         continue
#     if pert == 'light':
#         low_quality_ref = apply_low_light(ref_img, alpha=applied_alpha, beta=applied_beta)  # Tweak alpha/beta here
#     elif pert == 'blur':
#         low_quality_ref = apply_motion_blur(ref_img, kernel_size=applied_kernel_size, direction='horizontal')  # Tweak kernel_size/direction here
#     cv2.imwrite(new_ref_path, low_quality_ref)
#     print(f"Saved low-quality reference image to: {new_ref_path}")

# print("Processing complete.")