"""
utils for monitoring CPU metrics and GPU metrics
"""

import pynvml
import psutil
import json

pynvml.nvmlInit()

def get_gpu_info():
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)

        return {
            "gpu_utilization": gpu_util.gpu,
            "gpu_memory_used": gpu_memory.used / 1024**2,  # MB
            "gpu_memory_total": gpu_memory.total / 1024**2,  # MB
            "gpu_memory_free": gpu_memory.free / 1024**2,  # MB
            "gpu_power_usage_mW": power_usage / 1000  # W
        }
    except Exception as e:
        return {"error": str(e)}
    
def get_cpu_info():
    cpu_utilization = psutil.cpu_percent(interval=1)  # CPU usage percentage
    memory = psutil.virtual_memory()

    return {
        "cpu_utilization": cpu_utilization,
        "cpu_memory_used": memory.used / 1024**3,  # GB
        "cpu_memory_total": memory.total / 1024**3,  # GB
        "cpu_memory_free": memory.free / 1024**3  # GB
    }

def get_memory_info():
    memory = psutil.virtual_memory()
    return {
        "memory_used": memory.used / 1024**3,  # GB
        "memory_total": memory.total / 1024**3,  # GB
        "memory_free": memory.free / 1024**3  # GB
    }

def save_to_json(data, filename="inference_monitoring_data.json"):
    """Save monitoring data to a JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Monitoring data saved to {filename}")
