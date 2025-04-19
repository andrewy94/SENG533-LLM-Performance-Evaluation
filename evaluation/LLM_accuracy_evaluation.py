import json
import matplotlib.pyplot as plt
import os

os.makedirs("graphs", exist_ok=True)

json_files = [
    "data/generated_answers_DeepSeek-R1-Distill-Qwen-1.5B.json",
    "data/generated_answers_DeepSeek-R1-Distill-Qwen-1.5B-quantized-8bit.json",
    "data/generated_answers_DeepSeek-R1-Distill-Qwen-1.5B-quantized-4bit.json"
]

accuracies = []
model_names = [
    "Default",
    "8-bit Quantized",
    "4-bit Quantized"
]

# loop through each JSON file
for file in json_files:
    total_correct = 0
    total_prompts = 0
    
    with open(file, 'r') as f:
        json_data = json.load(f)
    
    # loop through the data in the JSON file
    for entry in json_data:
        # increment total prompt count
        total_prompts += 1
        
        # check if the answer is correct and increment the correct count
        if entry.get("correctness") == "correct":
            total_correct += 1
    
    # calculate the accuracy
    if total_prompts > 0:
        accuracy = total_correct / total_prompts * 100
        accuracies.append(accuracy)
    else:
        accuracies.append(0)

plt.figure(figsize=(8, 6))
bars = plt.bar(model_names, accuracies, color='tab:orange')
plt.title('LLM Accuracy for Different Models')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

# add text values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("graphs/llm_accuracy_plot.png")
