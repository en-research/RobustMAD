# RobustMAD: Evaluating Real-World Robustness of Multimodal Small Language Models for Deployable Anomaly Detection Assistants
by Anonymous Authors

## Abstract
Multimodal industrial anomaly inspection assistants are a critical component of nextgeneration smart factories, enabling interactive vision–language–based querying. However, multimodal large language models remain impractical for on-site deployment due to prohibitive computational demands and privacy risks from cloud-based inference. Compact multimodal small language models (MSLMs) offer a deployable alternative, yet progress is constrained by the lack of comprehensive robustness analyses and meaningfully challenging benchmarks that reflect real-world industrial conditions. To address this gap, we develop RobustMAD, a practically realistic benchmark, explicitly designed to evaluate real-world robustness through diverse open-ended queries spanning object understanding, anomaly detection, unanswerable problems, and visual quality degradations. Contrary to conventional assumptions, top-performing MSLMs exhibit promising capabilities, surprisingly outperforming even the larger GPT-5 Nano. However, they are still far below industrial standards, and RobustMAD exposes critical robustness gaps posing significant operational risks. In
particular, three recurring failure modes emerge: (i) fragile multimodal grounding under fine-grained distinctions or degraded visual conditions, (ii) insufficiently comprehensive responses, and (iii) weak logical grounding on unanswerable or ill-posed queries, leading to hallucinated outputs. Grounded in these insights, we provide actionable guidance for the design of next-generation multimodal industrial inspection assistants that leverage their promising competence.

## RobustMAD Benchmark Overview
We present Robust Multimodal Anomaly Detection (RobustMAD), the first practically realistic benchmark for evaluating the real-world robustness of MSLMs in industrial anomaly inspection—a field that can significantly benefit from on-device MSLM deployment. RobustMAD is explicitly designed to capture core challenges encountered in practice, including domain-intensive reasoning, non-standardized or ill-posed user queries, open-ended inspection reporting, and realistic visual quality variations. Our benchmark comprehensively encompasses two major types of robustness: **knowledge-based robustness**, which evaluates model reasoning under diverse, imperfect, and domain-knowledge–intensive queries, and **visual quality robustness**, which assesses model sensitivity to image-quality perturbations commonly occurring in dynamic assembly lines, such as motion blur and low lighting.

<div style="text-align: center;">
  <img src="figures/Fig_Benchmark_Dataset_Overview.jpg" alt="Overview" width="100%" />
</div>

## Evaluation


## Examples of persistant Robustness Gaps in MSLMs

## Acknowledgement
MVTec AD , VisA datasets
