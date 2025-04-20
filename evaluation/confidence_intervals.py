import json
import numpy as np
from scipy.stats import sem, t

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
}

for file in json_files:
    with open(file, 'r') as f:
        json_data = json.load(f)

    model_gpu_utilization = [entry["gpu"]["gpu_utilization"] for entry in json_data]
    model_gpu_memory_used = [entry["gpu"]["gpu_memory_used"] for entry in json_data]
    model_gpu_power_usage = [entry["gpu"]["gpu_power_usage_mW"] for entry in json_data]
    model_cpu_utilization = [entry["cpu"]["cpu_utilization"] for entry in json_data]
    model_cpu_memory_used = [entry["cpu"]["cpu_memory_used"] for entry in json_data]

    metrics_raw["gpu_utilization"].append(model_gpu_utilization)
    metrics_raw["gpu_memory_used"].append(model_gpu_memory_used)
    metrics_raw["gpu_power_usage"].append(model_gpu_power_usage)
    metrics_raw["cpu_utilization"].append(model_cpu_utilization)
    metrics_raw["cpu_memory_used"].append(model_cpu_memory_used)

# 95% confidence interval
def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation
    se_val = std_dev / np.sqrt(n)
    h = se_val * t.ppf((1 + confidence) / 2., n - 1)
    return mean, std_dev, mean - h, mean + h

# perform statistical analysis
print("95% Confidence Intervals (Sample Std Dev)\n" + "-" * 60)

for metric, data_lists in metrics_raw.items():
    print(f"\nMetric: {metric.replace('_', ' ').title()}")

    for model_name, data in zip(model_names, data_lists):
        mean, std_dev, lower, upper = compute_confidence_interval(data)
        print(f"  {model_name}:")
        print(f"    Mean = {mean:.2f}")
        print(f"    Std Dev (Sample) = {std_dev:.2f}")
        print(f"    95% CI = ({lower:.2f}, {upper:.2f})")

