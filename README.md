# SENG533-LLM-Performance-Evaluation
This project compares system performance of a large language model across full precision, 8-bit, and 4-bit quantized versions. It includes visualizations for GPU/CPU utilization, memory usage, power consumption, throughput, and confidence intervals.

---

## Model

- **Model:** [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

---

## Dataset

- **Dataset:** [`SVAMP`](https://huggingface.co/datasets/ChilleD/SVAMP)

A dataset of elementary-level math word problems used to evaluate model inference performance.

---

## Environment Setup

> All scripts were run in **WSL2 Ubuntu**. This was required because [VLLM](https://github.com/vllm-project/vllm) is not compatible with native Windows.

### Requirements

- WSL2 (Ubuntu)
- Python modules listed in requirements.txt
- CUDA-compatible GPU + NVIDIA Drivers
- `nvidia-smi` installed
- [`uv`](https://github.com/astral-sh/uv) for managing virtual environments (Python 3.12.3)

## Procedure
To perform the evaluation of our LLM model, we followed the following procedure:

1. Quantize and save the 4-bit and 8-bit models by running `utils/save_models.py` (models saved locally and untracked by git due to file size)
2. Run each model individually through 500 prompts from the `SVAMP` dataset using `data_initialization.py, saving results as .json files
3. Generate violin plots for the following metrics by running `evaluation/monitoring_data_evaluation.py`:
    - GPU Utilization (%)
    - GPU Memory used (mb)
    - GPU Power usage (mW)
    - CPU Utilization (%)
    - CPU Memory used (mb)
   
   Also generates a bar chart for throughput (prompts/sec)
4. Generate radar charts for each model (min-max normalization) by running `evaluation/radar_charts.py`
5. Perform 95% confidence interval analysis for each metric of each model by running `evaluation/confidence_intervals.py`

