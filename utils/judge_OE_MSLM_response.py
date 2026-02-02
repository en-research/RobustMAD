import json
import openai
import base64
import os
import time
from collections import defaultdict, Counter
import glob
import sys
from tqdm import tqdm 
import concurrent.futures 

# Making batch API calls (instead of single API calls) to OpenAI for judging model responses

# python script to evaluate model responses using OpenAI as judge
# Run per model (normal): nohup python utils/judge_OE_MSLM_response.py <model_name> >> results/<model_name>/output_oe_judge_batchAPI.log 2>&1 &
# Run per model (low): nohup python utils/judge_OE_MSLM_response.py <model_name> --low >> results/<model_name>/output_oe_judge_batchAPI.log 2>&1 &

# Example (normal): nohup python utils/judge_OE_MSLM_response.py OE_Qwen3_VL_4B_Instruct_s0 >> results/OE_Qwen3_VL_4B_Instruct_s0/output_oe_judge_batchAPI.log 2>&1 &
# Example (low): nohup python utils/judge_OE_MSLM_response.py OE_Qwen3_VL_4B_Instruct_s0 --low >> results/OE_Qwen3_VL_4B_Instruct_s0/output_oe_judge_batchAPI.log 2>&1 &


if len(sys.argv) < 2:
    print("Usage: python script.py <model_name> [--low]")
    sys.exit(1)

model_name = sys.argv[1]
suffix = "_low" if len(sys.argv) > 2 and sys.argv[2] == "--low" else ""

# Load API key from environment or set manually (do NOT commit your key to code)
api_key = ""
client = openai.OpenAI(api_key=api_key)

# Choose model for judging (use a strong vision model)
chosen_model = 'gpt-5'

# -------------------------------
# Client-side batching config
# -------------------------------
MAX_CONCURRENT_REQUESTS = 8  # Safe default for image-based VQA judging

# Load ground truth JSON
gt_file = f"RobustMAD_OE{suffix}.json"
with open(gt_file, "r") as f:
    gt_data = json.load(f)
gt_images = gt_data["images"]

# Create mappings: question_id to gt_info
gt_question_map = {}
for img in gt_images:
    for q in img["questions"]:
        gt_question_map[q["question_id"]] = {
            "gt_answer": q["answer"],
            "robustness_category": q["robustness_category"],
            "image_path": img["image_path"],
            "reference_image_path": img.get("reference_image_path", None),
            "anomaly": img["anomaly"],
            "specie_name": img["specie_name"],
            "category": img["category"],
            "dataset": img["dataset"],
            "image_id": img["image_id"]
        }

# ===============================
# JUDGE GUIDELINES 
# ===============================
judge_guidelines = """
### Role and Objective ###
You are an expert visual question answer (VQA) judge and an industrial product inspector evaluating generated answers from various candidate models performing product inspection and anomaly detection based on images. 
Critically evaluate the GENERATED_ANSWER. Score the GENERATED_ANSWER against the QUESTION, IMAGE(s), GROUND_TRUTH_ANSWER, and these GUIDELINES. 
Ensure the GENERATED_ANSWER semantically aligns closely with the GROUND_TRUTH_ANSWER in terms of key facts, details, and conclusions, while adhering to the style.
Penalize heavily for important deviations from GROUND_TRUTH_ANSWER, especially in factual details like defect presence, location, or object description. 
However, when comparing GENERATED_ANSWER with GROUND_TRUTH_ANSWER, DO NOT be overly rigid in terms word-for-word matching, but rather judge based on key semantic content (see few-shot human judging examples given below later).
For 'Unanswerable or Misleading Query Detection' robustness category, if GENERATED_ANSWER at least states that referred object/attribute in question is not present based on image AND mentions name of actual object shown, it must be given at least a passing overall score of 3 (and NOT lower like 2 or 1), even if GENERATED_ANSWER does not proceed to describe defect of actual object.  If GENERATED_ANSWER is more precise and helpful by ALSO mentioning glaring defects in actual object, the comprehensiveness and overall score can be increased to 4 (defect only mentioned) or 5 (if defect location is also mentioned). See Example 14 and 15 in few-shot examples below for scoring distinction.

### Core Guidelines to follow for judging GENERATED_ANSWER (Ensure Adherence in Evaluation) ###
- Answer must be clear, concise, accurate, and informative. Limit to 50 words max.
- Penalize ambiguous colors like "warm tones" or just "dark"; use specific colors, e.g., "dark brown".
- Generally, all answers should mention the name of object/product in image when answering the question for answer precision and helpfulness (e.g., "leather has no visible tears" is preferred over "the object has no visible tears").
- Generally, defect description should include its location for comprehesiveness, answer precision, and helpfulness.
- For locations, specific phrases like "bottom-right", "center-left", "central region" (imagine a 3x3 grid) are required over vague terms like "2 o'clock".
- For 'Unanswerable or Misleading Query Detection' robustness category questions, a good answer should state if referred object/attribute is not present, and then describe actual object is shown (with its defects if any) for for comprehesiveness, answer precision, and helpfulness.
- For technical objects like ultrasonic distance sensors or infrared sensors, do not make up component names/functions unless evident.
- If 'Anomaly Understanding and Localization (Pair-wise comparison)' robustness category questions, answer should compare query image to reference image.

### Category-Specific Ground Rules to also follow for judging GENERATED_ANSWER ###
Use these rules based on the provided CATEGORY to evaluate object descriptions and attributes:
- For CATEGORY: "cable" , green is also acceptable for the green/green-yellow conductor in GT, and gray is also acceptable color for the brown conductor in GT depending on image
- For CATEGORY: "capsule" (note: no 's'), acceptable materials of capsule shell are gelatin or hydroxypropyl methylcellulose (HPMC), acceptable colors in lieu of red are orange, peach, reddish brown, or similar
- For CATEGORY: "capsules", mentions of brown in color are acceptable if present in the image.
- For CATEGORY: "metal_nut", the object is a T-nut. Precise descriptions such as "T-nut/ T nut / Metal nut" are acceptable descriptions, but any other specific nuts (e.g., hexagonal nut, wing nut) other than T-nut are incorrect.
- For CATEGORY: "carpet", allow some leeway in descriptions; answers stating "woven fabric" or similar and a logical intended function of "woven fabric" are acceptable based on image, even if slightly different from GT.
- For CATEGORY: "leather", penalize score only slightly, not too heavily (e.g., give 4 vs 5) if leather is claimed to be synthetic unlike GT
- For CATEGORY: "pcb1" or "pcb2", the object is an HC-SR04 ultrasonic distance sensor module and should be precisely stated as at least a ultrasonic distance sensor module.
- For CATEGORY: "pcb3", the object is an infrared sensor module and should be precisely stated as at least an infrared sensor module.
- Appropriately penalize descriptions that contradict above rules.

### Detailed Templates for Each Robustness Category and Question Type ###
Use these as benchmarks (aligned with GROUND_TRUTH_ANSWER) to evaluate how well the GENERATED_ANSWER matches expected structure and content:

- ** Robustness Category 1: General Object Understanding**:
  - Q: What is the object shown, and what are its key features (e.g. color, material, and shape)? 
    Answer: The object is a top view of a cylindrical bottle, with dark brown glass material and amber-colored threaded rim.
  - Q: List the main attributes of the product in the image (e.g., color, texture, and material). 
    Answer: <Mention object in image> is <describe key attributes>.
  - Q: What is the object in the image, and what is its intended function? 
    Answer: The object is a glass bottle. Its intended function is to hold liquids.
  - Q: Explain the intended function or use of the object shown when it is in ideal condition. 
    Answer: In ideal condition, the glass bottle holds liquids securely without leaking.

- ** Robustness Category 2: Anomaly Understanding and Localization (Stand-alone)**:
  - Q: Does the object in the image have any defects? If so, describe the defect or anomaly and specify its location (e.g., upper-left, center, bottom-right). 
    Answer: Yes, <mention object in image: e.g., bottle> has <mention anomaly: e.g. a crack> in <mention location: e.g., the bottom-right of the rim>.
    or 
    <Object in image> has no visible defects. <Give some brief, pertinent reason/visual attributes for this conclusion.>
  - Q: Assess whether the product has any major defects. If so, how might the defect affect the object’s normal function or safety? 
    Answer: Yes, the <object in image> has <mention anomaly, e.g. a large crack at bottom-right>, which <mention impact: e.g. may cause leakage and pose broken glass safety hazard.>
    or 
    <Object in image> has no major defects. <Give some brief, pertinent reason/visual attributes for this conclusion.>
  - Q: Evaluate whether the product has any visible defects. If so, what distinguishes the damaged part from the rest of the object? 
    Answer: Yes, the <mention defect, e.g., crack> on <object in image> appears as a <mention defect: e.g., jagged line> in <mention defect location: e.g.,the bottom-left> differing from the smooth surface in the rest of region.
    or 
    <Object in image> has no visible defects. <Give some brief, pertinent reason/visual attributes for this conclusion.>

- ** Robustness Category 3: Anomaly Understanding and Localization (Pair-wise comparison)**:
  - Q: The first image shows a normal, defect-free reference object. The second image shows the query object for inspection. Compare the query to the reference, and assess whether the query object is normal or defective. If defective, describe any major anomalies with location (e.g., upper-left, center, bottom-right). 
    Answer: Yes, the query <mention object in image> has a defect. The query <mention object in image> shows <mention anomaly and location, e.g. a crack in the bottom-right> unlike the smooth reference <mention object in image>.
  - Q: The first image shows a normal, defect-free reference object. The second image shows the query object for inspection. Compare the query object to the reference and assess whether it has any major defects. If so, how might the defect affect the object’s normal function or safety? 
    Answer: Query <mention object in image> has a <mention anomaly and location, e.g. a jagged crack in the central region> unlike reference. This may cause leakage and safety risks <mention specific to object>.
  - Q: The first image shows a normal, defect-free reference object. The second image shows the query object for inspection. Compare the query object to the appearance and key attributes of the normal reference, and assess whether the query object satisfies the normal product quality standards. 
    Answer: Query <mention object in image> matches reference in shape and color but has <mention anomaly and location, e.g. a jagged crack in the top-left>. Query <mention object in image> does not meet quality standards.

- ** Robustness Category 4: Unanswerable or Misleading Query Detection**:
  - Q: Does the chair in the image have any defects? If so, describe the defect or anomaly and specify its location (e.g., upper-left, center, bottom-right). 
    Answer: The image does not show a chair. It shows <mention actual object in image, e.g., a top view of glass bottle> with <no defects or briefly mention anomaly with location>.
  - Q: What is wrong with the pear in the image? 
    Answer: The image does not show a pear. It shows <mention actual object in image, e.g., a top view of glass bottle> that appears normal <no defects or briefly mention anomaly with location>.
  - Q: What is the diameter of the bottle cap on the left of the image? 
    Answer: The image does not show a bottle cap. It shows <mention actual object in image with anomaly and location if any>. The diameter cannot be determined.
  - Q: What is the voltage rating printed on the device? 
    Answer: <First, check if voltage rating information is available in the image. If not available state: The <mention actual object in image and anomaly and location if any> does not show any printed voltage rating.
  - Q: List the main attributes of the cardboard box in the image (e.g., color, physical condition, and material). 
    Answer: The image does not show a cardboard box. It shows a <mention actual object in image, e.g., a top view of glass bottle> with <mention actual attributes, e.g., dark brown, smooth, glass material, anomaly and location if any>.
  - Q: Confirm that there are no major defects in the object shown. <This is a trick question evaluating expectational bias.>
    Answer: <First, check whether <mention object in image, e.g., a top view of glass bottle> has any defects or not, and answer accordingly. Do not blindly say that it is defect-free without checking. 
  - Q: Product appears to be normal and defect-free. Please confirm that it meets the normal product quality standards. <This is a trick question evaluating expectational bias.>
    Answer: <First, check whether <mention object in image, e.g., a top view of glass bottle> has any defects or not, and answer accordingly. Do not blindly say that it meets normal product quality standards without checking. Give brief reason for conclusion.>
  - Q: Explain the intended function or use of the hammer shown. 
    Answer: The image does not show a hammer, thus it's intended function cannot be determined. It shows a glass bottle with a large chip <i.e.,actual object and defect if present>  whose function is to hold liquids <specify function of actual object>.

### Multi-Dimensional Quality Assessment metric to judge GENERATED_ANSWER ###
Evaluate each dimension on a 1-5 (integer) scale:

a) Technical Accuracy (1-5): Measures how accurately the answer reflects important verifiable image details (e.g., object, defect location, etc.), without factual errors or hallucinated content. Must align with GROUND_TRUTH_ANSWER.
- 1: Completely incorrect with errors (e.g., misidentifying a PCB capacitor as a resistor) or hallucinations (e.g., inventing a defect in a defect-free object).
- 5: Perfectly matches key image details and GROUND_TRUTH_ANSWER, no errors or hallucinations.

b) Comprehensiveness (1-5): Measures whether the answer covers all essential components required by the question (e.g., defect type, location, impact for anomalies; attributes for objects). Should cover key points in GROUND_TRUTH_ANSWER.
- 1: Misses most required components of question (e.g., omits defect location or impact).
- 5: Fully includes all required components of question with appropriate and helpful key details, excludes unnecessary minor details.

c) Relevance (1-5): Measures how directly and appropriately the answer addresses the intent of the question, while grounded in the given image content.
- 1: Completely off-topic or misaligned with question/image (e.g., discusses irrelevant details to question or refers to non-existent objects/attributes).
- 5: Precisely addresses question intent with relevant information from image content.

d) Style and Clarity (1-5): Measures presentation clarity and adherence to formatting rules (<50 words, no parentheses or slashes) for concise and readable inspection reports.
- 1: Unclear, poorly structured, or violates formatting (e.g., >50 words, uses slashes).
- 5: Clear, concise, well-structured, follows all formatting rules.

### Evaluation Process ###
Step 0: Sanity check the GROUND_TRUTH_ANSWER against the image(s). Assume GT is correct unless there is a clear and major discrepancy (e.g., GT describes a defect not visible in the image, or misidentifies the object entirely). If such a discrepancy is found, include "gt_sanity_issue": "Brief description of the issue" in the output JSON. Otherwise, do not include this key. Proceed with judging assuming GT is correct.
Step 1: Score each dimension (1-5). Explain briefly in the explanation field. Be stringent about technical accuracy!
Step 2: Overall Score (1-5), based on the above dimensions, reflecting overall quality. Penalise overall score heavily if technical accuracy is low!
Step 3: Binary is_accurate: 1 if no factual errors/hallucinations (Technical Accuracy >=3, i.e., passable), else 0.

Additional important note on scoring:
- Penalise overall score heavily (<4) if technical accuracy or relevance is low (<4), see few-shot human examples below
- The difference between a 4 and 5 for overall score is mainly based on comprehensiveness and precision of answer relative to GT 
- Penalise relevance score when model makes up non-existent components/defects and does not answer directly to question, especially for unanswerable robustness category
- For 'Unanswerable or Misleading Query Detection' robustness category, if GENERATED_ANSWER at least states that referred object/attribute in question is not present based on image AND mentions name of actual object shown, it must be given at least a passing overall score of 3 (and NOT lower like 2 or 1), even if GENERATED_ANSWER does not proceed to describe defect of actual object.  If GENERATED_ANSWER is more precise and helpful by ALSO mentioning glaring defects in actual object, the comprehensiveness and overall score can be increased to 4 (defect only mentioned) or 5 (if defect location is also mentioned). See Example 14 and 15 in few-shot examples below for scoring distinction.

Output ONLY the exact JSON object with all keys and strings in double quotes. No extra text:
{{
  "dimension_scores": {{"technical_accuracy": x, "comprehensiveness": x, "relevance": x, "style_and_clarity": x}},
  "overall_score": x,
  "is_accurate": x,
  "explanation": "..."
}}
Optionally include "gt_sanity_issue": "..." if a major discrepancy is found.

###  15 Few-Shot Examples with Human Feedback on Judging and Scoring (Learn from These Evaluations) ###

** Example 1 (overall_score: 5): **
ROBUSTNESS_CATEGORY: Anomaly Understanding and Localization (Stand-alone)
QUESTION: Does the object in the image have any defects? If so, describe the defect or anomaly and specify its location (e.g., upper-left, center, bottom-right)
GENERATED_ANSWER: Yes, the object has a defect. The inner rim shows visible cracks and irregularities, particularly along the upper-left and bottom-right areas, indicating structural weakness or manufacturing flaw.
GROUND_TRUTH_ANSWER: Yes, the bottle rim has a large jagged chip spanning the bottom half of rim.

dimension_scores: {"technical_accuracy": 5, "comprehensiveness": 5, "relevance": 5, "style_and_clarity": 5}
overall_score: 5
is_accurate: 1
Human judgement and scoring given: Model correctly identifies defect (crack versus GT's large chip are semantically and practically very similar). For defect location, although word-for-word description differs from GT, in this particular case, the location description is similar to GT's semantic meaning after double checking with image due to symmetry of bottle."


** Example 2 (overall_score: 5): **
ROBUSTNESS_CATEGORY: General Object Understanding
QUESTION: What is the object shown, and what are its key features (e.g. color, material, and shape)?
GENERATED_ANSWER: The object is a rectangular, blue PCB infrared sensor module with a blue trimpot, LM393 chip, clear IR LED and black IR receiver, and a three pin header. Materials include fibreglass, plastic and metal components.
GROUND_TRUTH_ANSWER: The object is an infrared (IR) sensor module. It has a blue printed circuit board (PCB), made of fibreglass, plastic and metal components, a clear IR light-emitting diode (LED), a black IR receiver, an LM393 comparator chip, a blue adjustment potentiometer, and a three-pin header, in a slim rectangular shape.

dimension_scores: {"technical_accuracy": 5, "comprehensiveness": 5, "relevance": 5, "style_and_clarity": 5}
overall_score: 5
is_accurate: 1
Human judgement and scoring given: Aligns well with GT; Sufficiently detailed and answer only mentions clearly verifiable component details from image. Precisely mentions object is infrared sensor module instead of just saying "small electronic sensor module'."

** Example 3 (overall_score: 4): **
ROBUSTNESS_CATEGORY: General Object Understanding
QUESTION: What is the object shown, and what are its key features (e.g. color, material, and shape)?
GENERATED_ANSWER: The object is a dark amber glass bottle viewed from above. It has a circular shape with a smooth, reflective rim and a dark interior, indicating it may contain liquid. The material appears to be glass, with a glossy finish.
GROUND_TRUTH_ANSWER: The object is a top view of a glass bottle opening. It is cylindrical and circular, made of dark brown amber glass with a smooth, even rim and visible threading.

dimension_scores: {"technical_accuracy": 4, "comprehensiveness": 5, "relevance": 5, "style_and_clarity": 5}
overall_score: 4
is_accurate: 1
Human judgement and scoring given: Model's answer covers key requirements of question and aligns well with GT even though there are minor differences like threading not mentioned; Sufficiently detailed. One minor issue is that answer speculates about a liquid inside, thus technical accuracy score is 4."

** Example 4 (overall_score: 4): **
ROBUSTNESS_CATEGORY: General Object Understanding
QUESTION: List the main attributes of the product in the image (e.g., color, texture, and material).
GENERATED_ANSWER: The product is a dark, glossy glass bottle with a smooth, reflective surface. Its interior shows a deep black hue, and the rim exhibits subtle amber reflections, indicating high-quality, transparent material with a polished finish.
GROUND_TRUTH_ANSWER: The bottle in the image is dark brown glass with an amber rim. It is circular, smooth and glossy, with concentric threading around the opening.

dimension_scores: {"technical_accuracy": 5, "comprehensiveness": 4, "relevance": 5, "style_and_clarity": 3}
overall_score: 4
is_accurate: 1
Human judgement and scoring given: Model's answer covers key requirements of question and aligns well with GT even though there are minor differences. Key shortcoming is that bottle color is vaguely mentioned as "dark" and contains overly subjective descriptions like "high-quality". Thus, comprehensiveness, style and overall scores are slightly penalised, but answer is still technically correct."

** Example 5 (overall_score: 3): **
ROBUSTNESS_CATEGORY: Anomaly Understanding and Localization (Pair-wise comparison)
QUESTION: The first image shows a normal, defect-free reference object. The second image shows the query object for inspection. Compare the query object to the reference and assess whether it has any major defects. If so, how might the defect affect the object's normal function or safety?
GENERATED_ANSWER: The query object shows minor surface imperfections but no major defects. These do not impact normal function or safety, as they are superficial and do not compromise structural integrity or performance.
GROUND_TRUTH_ANSWER: Query bottle opening appears normal. It matches the reference with a smooth circular rim and uniform dark interior. No cracks, chips, or deformations on the rim or threads. It meets quality standards and should seal and hold liquids safely.

dimension_scores: {"technical_accuracy": 4, "comprehensiveness": 3, "relevance": 4, "style_and_clarity": 3}
overall_score: 3
is_accurate: 1
Human judgement and scoring given: Model's answer correctly concludes there are no major defects like GT answer despite the minor surface imperfections. However, there is no explicit mention of actual object name or mentions of reference image for helpfulness. Thus, comprehensiveness score and overall score are penalised although answer is technically accurate."


** Example 6 (overall_score: 3): **
ROBUSTNESS_CATEGORY: General Object Understanding
QUESTION: What is the object shown, and what are its key features (e.g. color, material, and shape)?
GENERATED_ANSWER: The object is a small electronic sensor module. It has a rectangular blue PCB made of fiberglass, with metal 3 pin header, clear LED at the right edge, black sensor bulb beside it, a blue trim pot, SMD IC, resistors, and labeled white silkscreen.
GROUND_TRUTH_ANSWER: The object is an infrared (IR) sensor module. It has a blue printed circuit board (PCB), made of fibreglass, plastic and metal components, a clear IR light-emitting diode (LED), a black IR receiver, an LM393 comparator chip, a blue adjustment potentiometer, and a three-pin header, in a slim rectangular shape.

dimension_scores: {"technical_accuracy": 3, "comprehensiveness": 4, "relevance": 3, "style_and_clarity": 3}
overall_score: 3
is_accurate: 1
Human judgement and scoring given: Question asks for key features but answer goes into too many minor components and is not precise in stating object is actually an infrared sensor module. Some component details are inaccurate (e.g., "sensor bulb" instead of "IR receiver"). Therefore, technical accuracy, relevance, style_and_clarity, and overall scores are penalized to 3.

** Example 7 (overall_score: 2): **
ROBUSTNESS_CATEGORY: Anomaly Understanding and Localization (Stand-alone)
QUESTION: Does the object in the image have any defects? If so, describe the defect or anomaly and specify its location (e.g., upper-left, center, bottom-right).
GENERATED_ANSWER: Yes, the object has a defect. There is a visible crack or fracture running diagonally across its surface from one side to the other near the center area.
GROUND_TRUTH_ANSWER: Yes. The metal nut has two broad scratches at top-right and bottom-left, disrupting the concentric machined pattern.

dimension_scores: {"technical_accuracy": 1, "comprehensiveness": 1, "relevance": 3, "style_and_clarity": 3}
overall_score: 2
is_accurate: 0
Human judgement and scoring given: Model's answer is imprecise and inaccurate. The actual object name (metal nut) is critical but it is not given, the presence of two scratches and their locations are missing and/or incorrect. Therefore, technical accuracy, comprehensiveness, and overall scores are penalized heavily. is_accurate flag is also 0 because object is unidentified and defect location incorrect.

** Example 8 (overall_score: 3): **
ROBUSTNESS_CATEGORY: Unanswerable or Misleading Query Detection
QUESTION: List the main attributes of the cardboard box in the image (e.g., color, physical condition, and material).
GENERATED_ANSWER:  The image does not show a cardboard box. Instead, it depicts a metallic object with a central hole and four protruding arms. The main attributes of this item are its metallic material, circular shape, and the presence of sharp edges on the extending parts."
GROUND_TRUTH_ANSWER: The image does not show a cardboard box. It shows a metal nut that is light gray metallic with a central circular hole, smooth reflective surface with visible linear scratches on the top-center and top-right.

dimension_scores: {"technical_accuracy": 3, "comprehensiveness": 3, "relevance": 3, "style_and_clarity": 4}
overall_score: 3
is_accurate: 1
Human judgement and scoring given: Model correctly states that actual object is not a cardboardbox, but fails to precisely identify object as a metal nut (says metallic object instead) and fails to mention glaring defects in the attributes. Therefore, technical accuracy, relevance, comprehensiveness and overall scores are penalized to 3.

** Example 9 (overall_score: 1): **
ROBUSTNESS_CATEGORY: Unanswerable or Misleading Query Detection
QUESTION: Product appears to be normal and defect-free. Please confirm that it meets the normal product quality standards.
GENERATED_ANSWER: The product appears to be normal and defect-free. It meets the normal product quality standards with no visible defects or irregularities in its structure and finish.
GROUND_TRUTH_ANSWER: No. The metal nut has visible linear scratches on the top-center and top-right, disrupting the smooth finish. This surface damage indicates poor finishing quality. It does not meet normal product quality standards.

dimension_scores: {"technical_accuracy": 1, "comprehensiveness": 1 "relevance": 1, "style_and_clarity": 3}
overall_score: 1
is_accurate: 0
Human judgement and scoring given: Model's answer is clearly incorrect as metal nut has major defects as mention in GT and does not meet product quality standards. Therefore, technical accuracy, relevance, comprehensiveness, and overall scores are penalized to 1. is_accurate flag is also 0.

** Example 10 (overall_score: 3): **
ROBUSTNESS_CATEGORY: Anomaly Understanding and Localization (Pair-wise comparison)
QUESTION: The first image shows a normal, defect-free reference object. The second image shows the query object for inspection. Compare the query object to the appearance and key attributes of the normal reference, and assess whether the query object satisfies the normal product quality standards.
GENERATED_ANSWER: The query object matches the reference in appearance and key attributes. No visible defects are present. It satisfies normal product quality standards.
GROUND_TRUTH_ANSWER: Query bottle top view matches the normal reference. Circular opening, uniform dark brown rim, and even threading are intact. No cracks, chips, or irregularities visible. The query bottle meets normal product quality standards.

dimension_scores: {"technical_accuracy": 3, "comprehensiveness": 2, "relevance": 3, "style_and_clarity": 2}
overall_score: 3
is_accurate: 1
Human judgement and scoring given: Although model correctly concludes query is defect-free, it does not mention actual object name (bottle) or give brief reason for query and reference matching. Thus answer is vague and imprecise. Therefore, technical accuracy, relevance, comprehensiveness, style_and_clarity and overall scores are penalized.

** Example 11 (overall_score: 2): **
CATEGORY: metal_nut
ROBUSTNESS_CATEGORY: General Object Understanding
QUESTION: What is the object in the image, and what is its intended function?
GENERATED_ANSWER: The object is a hexagonal flange nut or wing nut, designed to secure fasteners by providing a stable, flat surface for tightening against a surface, often used in industrial assemblies for enhanced grip and load distribution.
GROUND_TRUTH_ANSWER: The object is a metal nut, silver with four lobes and a central hole. Its intended function is to thread onto a bolt to fasten and secure parts.

dimension_scores: {"technical_accuracy": 2, "comprehensiveness": 3, "relevance": 3, "style_and_clarity": 3}
overall_score: 2
is_accurate: 0
Human judgement and scoring given: For object category = metal_nut, the metal nut is a T-nut to be specific. GT's "metal nut" is still acceptable. However model's is technically incorrect to state it is a hexagonal flange nut or wing nut. Therefore, technical accuracy and overall scores are penalized to 2 (not 1 because function is still partially correct). The is_accurate flag is 0.

** Example 12 (overall_score: 1): **
ROBUSTNESS_CATEGORY: Anomaly Understanding and Localization (Pair-wise comparison)",
QUESTION: The first image shows a normal, defect-free reference object. The second image shows the query object for inspection. Compare the query object to the reference and assess whether it has any major defects. If so, how might the defect affect the object's normal function or safety?
GENERATED_ANSWER: The query object appears to have a significant crack, which could compromise its structural integrity and potentially lead to failure under stress or impact, posing a safety risk.
GROUND_TRUTH_ANSWER: Query bottle opening appears normal. It matches the reference with a smooth circular rim and uniform dark interior. No cracks, chips, or deformations on the rim or threads. It meets quality standards and should seal and hold liquids safely.

dimension_scores: {"technical_accuracy": 1, "comprehensiveness": 1, "relevance": 1, "style_and_clarity": 3}
overall_score: 1
is_accurate: 0
Human judgement and scoring given: Model does not mention actual object name and hallucinates a defect that is not present in image. Therefore, technical accuracy, relevance,  overall scores are heavily penalized to 1.

** Example 13 (overall_score: 4): **
CATEGORY: carpet
ROBUSTNESS_CATEGORY: Anomaly Understanding and Localization (Stand-alone)
QUESTION: Assess whether the product has any major defects. If so, how might the defect affect the object's normal function or safety?
GENERATED_ANSWER: The image shows a woven fabric, possibly a type of textile. There are no visible defects such as tears, holes, or frayed threads that would compromise the fabric's integrity or safety. The object appears to be in good condition, suitable for its intended use.
GROUND_TRUTH_ANSWER: The carpet has no major defects. The weave is uniform across the surface with no pulls, holes, frays, or stains. It should function normally with no impact on performance or safety.

dimension_scores: {"technical_accuracy": 4, "comprehensiveness": 5, "relevance": 5, "style_and_clarity": 5}
overall_score: 4
is_accurate: 1
Human judgement and scoring given: Only for object category = carpet, there is some leeway based on the image and piece of woven fabric is acceptable. Technical accuracy score is 4 instead of 5.

** Example 14 (overall_score: 3): **

ROBUSTNESS_CATEGORY: Unanswerable or Misleading Query Detection
QUESTION: Does the chair in the image have any defects? If so, describe the defect or anomaly and specify its location (e.g., upper-left, center, bottom-right).
GENERATED_ANSWER: There is no chair in the image. The image shows four circular tealight candles on a dark surface. No defects or anomalies are present as there is no chair to inspect.
GROUND_TRUTH_ANSWER: The image does not show a chair. It shows four round white candles. The bottom-left candle has a short, off-center wick, unlike the centered wicks on the others. This defect may cause uneven or poor burning.

dimension_scores: {"technical_accuracy": 4, "comprehensiveness": 3, "relevance": 4, "style_and_clarity": 5}
overall_score: 3
is_accurate: 1
Human judgement and scoring given: Model aptly addresses question, states there is no chair and mentions precise actual object name (tealight candles). However, it does not proceed to describe defect and defect location of actual object, making answer not helpful in answering a mistaken query. Therefore, comprehensiveness, and overall score is 3 (passable).



** Example 15 (overall_score: 4): **

ROBUSTNESS_CATEGORY: Unanswerable or Misleading Query Detection
QUESTION: What is wrong with the pear in the image?
GENERATED_ANSWER: There is no pear in the image. The image shows a wooden surface with two red marks, likely stains or smudges, which are not related to a pear.
GROUND_TRUTH_ANSWER: The image does not show a pear. It shows a wooden surface with two red discoloration stains at center-left and bottom-right.

dimension_scores: {"technical_accuracy": 4, "comprehensiveness": 4, "relevance": 5, "style_and_clarity": 5}
overall_score: 4
is_accurate: 1
Human judgement and scoring given: Model aptly addresses question, states there is no pear and mentions precise object name with glaring defects. However, defect location is not given, making answer not helpful in answering a mistaken query. Therefore, technical accuracy, comprehensiveness, and overall score is 4 instead of 5.
"""


def load_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_image_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

def judge_answer(img_info, question_info, cat, gt_answer, category):
    start_time = time.time()
    query_path = img_info["image_path"]

    prompt = judge_guidelines
    prompt += f"\nROBUSTNESS_CATEGORY: {cat}\n"
    prompt += f"\nCATEGORY: {category}\n"
    prompt += f"\nQUESTION: {question_info['question']}\n"
    prompt += f"\nGENERATED_ANSWER: {question_info['answer']}\n"
    prompt += f"\nGROUND_TRUTH_ANSWER: {gt_answer}\n"

    content = [{"type": "text", "text": prompt}]
    query_b64 = load_image_base64(query_path)
    query_mime = get_image_mime(query_path)
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{query_mime};base64,{query_b64}"}
    })

    if "Pair-wise comparison" in cat and img_info["reference_image_path"]:
        ref_path = img_info["reference_image_path"]
        ref_b64 = load_image_base64(ref_path)
        ref_mime = get_image_mime(ref_path)
        content.insert(1, {
            "type": "image_url",
            "image_url": {"url": f"data:{ref_mime};base64,{ref_b64}"}
        })

    response = None
    for attempt in range(3):
        try:
            messages = [{"role": "user", "content": content}]
            response = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                reasoning_effort="medium",
                verbosity="low"
            )
            break
        except Exception as e:
            print(f"[ERROR] API call attempt {attempt + 1} failed: {str(e)}")
            if attempt == 2:
                raise RuntimeError("API call failed after 3 attempts.")

    judge_output = json.loads(response.choices[0].message.content.strip())
    return judge_output

def process_single_entry(entry):
    qid = entry.get("question_id")
    if qid not in gt_question_map:
        print(f"[WARN] Question ID {qid} not found in GT. Skipping.")
        return None, qid, None

    gt_info = gt_question_map[qid]
    img_info = {
        "image_path": gt_info["image_path"],
        "reference_image_path": gt_info["reference_image_path"]
    }
    question_info = {
        "question": entry["question"],
        "answer": entry["mslm_answer"],
        "question_id": qid
    }

    judge_result = judge_answer(
        img_info,
        question_info,
        gt_info["robustness_category"],
        gt_info["gt_answer"],
        gt_info["category"]
    )

    question_entry = entry.copy()
    question_entry.update({
        "category": gt_info["category"],
        "anomaly": gt_info["anomaly"],
        "specie_name": gt_info["specie_name"],
        "dataset": gt_info["dataset"],
        "image_id": gt_info["image_id"],
        "judge": judge_result
    })

    return question_entry, None, judge_result

# -------------------------------
# MAIN EXECUTION
# -------------------------------
results_dir = "results"
model_dir = os.path.join(results_dir, model_name)
input_json = os.path.join(model_dir, f"{model_name}{suffix}.json")

with open(input_json, "r") as f:
    model_data = json.load(f)

processed_questions = []
skipped_questions = []

temp_output_file = os.path.join(
    model_dir, f"{model_name}{suffix}_scored.temp_batch_api.json"
)

if os.path.exists(temp_output_file):
    with open(temp_output_file, "r") as f:
        data = json.load(f)
    processed_questions = data["questions"]
    print(f"[RESUME] Loaded {len(processed_questions)} processed questions from temp file.")

done_qids = {q["question_id"] for q in processed_questions}
pending_entries = [entry for entry in model_data if entry.get("question_id") not in done_qids]

total_questions = len(model_data)
completed_count = len(processed_questions)

with concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_REQUESTS
) as executor:

    futures = [
        executor.submit(process_single_entry, entry)
        for entry in pending_entries
    ]

    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        desc="Processing questions"
    ):
        question_entry, skipped_qid, _ = future.result()

        completed_count += 1
        print(f"[PROGRESS] Completed {completed_count}/{total_questions}")

        if skipped_qid is not None:
            skipped_questions.append(skipped_qid)
            continue

        if question_entry is None:
            continue

        processed_questions.append(question_entry)

        # <<< ADDED: deterministic ordering ONLY for temp JSON >>>
        processed_questions_sorted = sorted(
            processed_questions,
            key=lambda x: x.get("question_id", -1)
        )

        with open(temp_output_file, "w") as f:
            json.dump(
                {
                    "questions": processed_questions_sorted,
                    "meta_metrics": {}
                },
                f,
                indent=4
            )

# -------------------------------
# FINAL METRICS & OUTPUT (UNCHANGED)
# -------------------------------
score_counter = Counter()
for q in processed_questions:
    score_counter[q["judge"]["overall_score"]] += 1

meta_metrics = {
    "score_distribution": dict(score_counter),
    "total_questions_processed": len(processed_questions),
    "skipped_questions": skipped_questions
}

output_file = os.path.join(
    model_dir, f"{model_name}{suffix}_scored_batched_api.json"
)

with open(output_file, "w") as f:
    json.dump(
        {
            "questions": processed_questions,
            "meta_metrics": meta_metrics
        },
        f,
        indent=4
    )

print(f"Output saved to {output_file}")
print("Processing complete.")