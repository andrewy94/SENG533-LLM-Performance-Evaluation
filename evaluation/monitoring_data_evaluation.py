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

# loop through each JSON file
for file in json_files:
    with open(file, 'r') as f:
        json_data = json.load(f)

    model_gpu_utilization = [entry["gpu"]["gpu_utilization"] for entry in json_data]
    model_gpu_memory_used = [entry["gpu"]["gpu_memory_used"] for entry in json_data]
    model_gpu_power_usage = [entry["gpu"]["gpu_power_usage_mW"] for entry in json_data]
    model_cpu_utilization = [entry["cpu"]["cpu_utilization"] for entry in json_data]
    model_cpu_memory_used = [entry["cpu"]["cpu_memory_used"] for entry in json_data]

    # calculate throughput
    last_entry = json_data[-1]
    total_prompts = last_entry["prompt_id"]
    total_time = last_entry["timestamp"]
    model_throughput = total_prompts / total_time
    model_throughput_list = [model_throughput] * len(json_data)

    # 
    metrics_raw["gpu_utilization"].append(model_gpu_utilization)
    metrics_raw["gpu_memory_used"].append(model_gpu_memory_used)
    metrics_raw["gpu_power_usage"].append(model_gpu_power_usage)
    metrics_raw["cpu_utilization"].append(model_cpu_utilization)
    metrics_raw["cpu_memory_used"].append(model_cpu_memory_used)
    metrics_raw["throughput"].append(model_throughput_list)

# function to create violin plots
def create_violin_plot(title, ylabel, data_lists, filename, color):
    plt.figure()
    parts = plt.violinplot(data_lists, showmeans=True, showextrema=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    plt.xticks(ticks=np.arange(1, len(model_names)+1), labels=model_names)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"graphs/{filename}")
    plt.close()

# generate violin plots
create_violin_plot('GPU Utilization Distribution', '% Utilization', metrics_raw["gpu_utilization"], "violin_gpu_utilization.png", 'tab:blue')
create_violin_plot('GPU Memory Used Distribution', 'Memory Used (MB)', metrics_raw["gpu_memory_used"], "violin_gpu_memory_used.png", 'tab:green')
create_violin_plot('GPU Power Usage Distribution', 'Power Usage (W)', metrics_raw["gpu_power_usage"], "violin_gpu_power_usage.png", 'tab:red')
create_violin_plot('CPU Utilization Distribution', '% Utilization', metrics_raw["cpu_utilization"], "violin_cpu_utilization.png", 'tab:cyan')
create_violin_plot('CPU Memory Used Distribution', 'Memory Used (GB)', metrics_raw["cpu_memory_used"], "violin_cpu_memory_used.png", 'tab:purple')

# generate bar plot for throughput
throughput_values = [np.mean(t) for t in metrics_raw["throughput"]]

plt.figure()
bars = plt.bar(model_names, throughput_values, color='tab:brown', alpha=0.8)
plt.title('Average Throughput per Model')
plt.ylabel('Throughput (prompts/s)')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Annotate bars with exact values
for bar, value in zip(bars, throughput_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig("graphs/bar_throughput.png")
plt.close()

