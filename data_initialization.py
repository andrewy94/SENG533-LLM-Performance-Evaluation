"""
Script for gathering data from a selected LLM model running the SVAMP dataset. 

Collects GPU utilization, average GPU memory used, average GPU power usage, 
CPU utilization, average CPU memory used,

Collects LLM responses and uses regex to isolate final answer
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Thread
from datasets import load_dataset
from utils.monitors import get_gpu_info, get_cpu_info, save_to_json
import os
import re

default_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
four_bit_model = "/root/SENG533-LLM-Performance-Evaluation/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-quantized-4bit"
eight_bit_model = "/root/SENG533-LLM-Performance-Evaluation/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-quantized-8bit"

# load model and tokenizer
model_name = default_model # manually change 
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# check if CUDA is available and ensure model is running on CUDA not CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if model_name != eight_bit_model:
    model.to(device)

# Create graphs directory if it doesn't exist
os.makedirs("data_collection", exist_ok=True)

# system prompt
system = "Please reason step by step, and put your final answer within \\boxed{}:"

# load SVAMP train dataset
dataset = load_dataset("ChilleD/SVAMP", split="train")

# number of prompts per runthrough of data collection
num_prompts = 500

# inference is running flag
inference_running = False

answers_data = []

def extract_boxed_answer(generated_text):
    # regex to find the contents within //boxed{}
    matches = re.findall(r'\\boxed{([^}]*)}', generated_text)
    if matches:
        # return final answer only
        return matches[-1] 
    return None 

def monitor_system_inference(inference_start_time, get_prompt_id):
    global inference_running
    monitoring_data = []
    
    while inference_running:
        gpu_info = get_gpu_info()
        cpu_info = get_cpu_info()
        timestamp = time.time() - inference_start_time
        prompt_id = get_prompt_id()

        data_entry = {
            "timestamp": timestamp,
            "prompt_id": prompt_id,
            "gpu": gpu_info,
            "cpu": cpu_info
        }
        monitoring_data.append(data_entry)

        print(f"[{timestamp:.4f}s] Monitoring for Prompt ID {prompt_id}...")

    # calculate averages
    total_samples = len(monitoring_data)
    if total_samples > 0:
        gpu_util_avg = sum(d['gpu']['gpu_utilization'] for d in monitoring_data) / total_samples
        gpu_mem_avg = sum(d['gpu']['gpu_memory_used'] for d in monitoring_data) / total_samples
        gpu_power_avg = sum(d['gpu']['gpu_power_usage_mW'] for d in monitoring_data) / (total_samples * 1000)  # Convert mW to W
        cpu_util_avg = sum(d['cpu']['cpu_utilization'] for d in monitoring_data) / total_samples
        cpu_mem_avg = sum(d['cpu']['cpu_memory_used'] for d in monitoring_data) / total_samples
    else:
        gpu_util_avg = gpu_mem_avg = gpu_power_avg = cpu_util_avg = cpu_mem_avg = 0

    inference_time = time.time() - inference_start_time

    print("\n=== Inference Performance Metrics ===")
    print(f"Total Inference Time: {inference_time:.3f} seconds")
    print(f"Sampling Frequency: {total_samples/inference_time:.1f} Hz")
    print("\nAverage Resource Utilization:")
    print(f"GPU Utilization: {gpu_util_avg:.1f}%")
    print(f"GPU Memory Used: {gpu_mem_avg:.1f} MB")
    print(f"GPU Power Draw: {gpu_power_avg:.2f} W")
    print(f"CPU Utilization: {cpu_util_avg:.1f}%")
    print(f"CPU Memory Used: {cpu_mem_avg:.2f} GB")

    # ave monitoring data with mode name
    model_name_for_filename = model_name.split("/")[-1]
    json_filename = f"data_collection/monitoring_data_{model_name_for_filename}.json"
    save_to_json(monitoring_data, json_filename)


def run_inference():
    global inference_running
    start_time = time.time()
    inference_running = True

    # track the current prompt ID
    prompt_id = None

    # start monitoring in a separate thread
    monitor_thread = Thread(target=monitor_system_inference, args=(start_time, lambda: prompt_id))
    monitor_thread.start()

    # process the dataset and perform inference
    for i, data_point in enumerate(dataset):
        if i >= num_prompts:
            break
        
        body = data_point["Body"]
        question = data_point["Question"]
        expected_answer = data_point["Answer"]
        
        # construct the full prompt for the model
        prompt = f"Body: {body}\nQuestion: {question}"
        prompt_id = i + 1 
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([text], return_tensors="pt").to(device)

        # run inference
        print(f"[{time.time() - start_time:.4f}s] Starting inference for prompt {prompt_id}: {prompt}")
        output = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"[{time.time() - start_time:.4f}s] Inference completed for prompt {prompt_id}.")

        # extract the boxed value from the generated answer
        extracted_answer = extract_boxed_answer(generated_answer)

        # mark as correct or incorrect based on comparison
        if extracted_answer and extracted_answer.strip() == expected_answer.strip():
            correctness = "correct"
        else:
            correctness = "incorrect"

        # append the data to answers_data
        answers_data.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "extracted_answer": extracted_answer,
            "correctness": correctness
        })

    # stop monitoring
    inference_running = False
    monitor_thread.join()

    # save the answers to a JSON file with model name
    model_name_for_filename = model_name.split("/")[-1]
    json_filename = f"data_collection/generated_answers_{model_name_for_filename}.json"
    save_to_json(answers_data, json_filename)


if __name__ == "__main__":
    run_inference()
