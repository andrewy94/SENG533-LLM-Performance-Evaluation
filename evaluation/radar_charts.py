import json
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("graphs", exist_ok=True)

json_files = [
    "data/monitoring_data_DeepSeek-R1-Distill-Qwen-1.5B.json",
    "data/monitoring_data_DeepSeek-R1-Distill-Qwen-1.5B-quantized-8bit.json",
    "data/monitoring_data_DeepSeek-R1-Distill-Qwen-1.5B-quantized-4bit.json"
]

model_names = [
    "Default",
    "8-bit Quantized",
    "4-bit Quantized"
]

metrics_raw = {
    "gpu_utilization": [],
    "gpu_memory_used": [],
    "gpu_power_usage": [],
    "cpu_utilization": [],
    "cpu_memory_used": [],
    "throughput": [],
}

for file in json_files:
    with open(file, 'r') as f:
        json_data = json.load(f)

    model_gpu_utilization = [entry["gpu"]["gpu_utilization"] for entry in json_data]
    model_gpu_memory_used = [entry["gpu"]["gpu_memory_used"] for entry in json_data]
    model_gpu_power_usage = [entry["gpu"]["gpu_power_usage_mW"] for entry in json_data]
    model_cpu_utilization = [entry["cpu"]["cpu_utilization"] for entry in json_data]
    model_cpu_memory_used = [entry["cpu"]["cpu_memory_used"] for entry in json_data]

    last_entry = json_data[-1]
    total_prompts = last_entry["prompt_id"]
    total_time = last_entry["timestamp"]
    model_throughput = total_prompts / total_time

    metrics_raw["gpu_utilization"].append(model_gpu_utilization)
    metrics_raw["gpu_memory_used"].append(model_gpu_memory_used)
    metrics_raw["gpu_power_usage"].append(model_gpu_power_usage)
    metrics_raw["cpu_utilization"].append(model_cpu_utilization)
    metrics_raw["cpu_memory_used"].append(model_cpu_memory_used)
    metrics_raw["throughput"].append([model_throughput] * len(json_data))

# function to normalize the metrics using Min-Max normalization
def normalize_data_across_models(data_all_models):
    all_data = [item for sublist in data_all_models for item in sublist]
    min_val = np.min(all_data)
    max_val = np.max(all_data)
    if min_val == max_val:
        return [0.5] * len(all_data)
    return [(x - min_val) / (max_val - min_val) for x in all_data]

# normalized metric values
metrics_means_normalized = {
    "gpu_utilization": [],
    "gpu_memory_used": [],
    "gpu_power_usage": [],
    "cpu_utilization": [],
    "cpu_memory_used": [],
    "throughput": [],
}

# normalize the metrics and calculate means across models
for key in metrics_raw:
    normalized_data = normalize_data_across_models(metrics_raw[key])
    
    # store normalized values for each model
    idx = 0
    for i in range(len(model_names)):
        metrics_means_normalized[key].append(np.mean(normalized_data[idx:idx+len(metrics_raw[key][i])]))
        idx += len(metrics_raw[key][i])

# function to create a radar chart for each model
def create_radar_chart(model_name, data, filename):
    categories = list(metrics_means_normalized.keys())
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    data = data + [data[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100, subplot_kw=dict(polar=True))

    ax.plot(angles, data, label=model_name, linewidth=2, linestyle='solid', color='tab:blue')
    ax.fill(angles, data, alpha=0.25, color='tab:blue')

    ax.set_ylim(0, 1) 
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(f'{model_name} Model  - Radar Chart', size=16, color='black', pad=20)

    plt.tight_layout()
    plt.savefig(f"graphs/{filename}")
    plt.close()


# generate charts
for i in range(len(model_names)):
    model_data = [metrics_means_normalized[key][i] for key in metrics_means_normalized]
    filename = f"radar_chart_{model_names[i].replace(' ', '_').lower()}.png"
    print(f"Generating radar chart for {model_names[i]} with normalized data: {model_data}")
    create_radar_chart(model_names[i], model_data, filename)
