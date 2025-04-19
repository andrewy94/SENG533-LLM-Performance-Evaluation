"""
Quantizes and saves the deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B model into 4-bit and 8-bit models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load the model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to quantize and save the model
def quantize_and_save(model_name, bits):
    # Define the quantization config
    if bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_quant_type="nf8", bnb_8bit_compute_dtype=torch.bfloat16)  
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config
        )
    elif bits == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)  
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config
        )
    else:
        raise ValueError("Only 4-bit and 8-bit quantization are supported.")

    # Save the quantized model and tokenizer
    save_path = f"{model_name}-quantized-{bits}bit"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return model

# Quantize and save the 8-bit model
quantized_model_8bit = quantize_and_save(model_name, bits=8)

# Quantize and save the 4-bit model
quantized_model_4bit = quantize_and_save(model_name, bits=4)

print("Quantized models have been saved successfully.")
