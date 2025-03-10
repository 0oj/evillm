import json

# Load the combined dataset
with open('combined_dataset.json', 'r') as f:
    data = json.load(f)

# Convert to JSONL format suitable for fine-tuning
with open('training_data.jsonl', 'w') as f:
    for item in data:
        # Adapt this structure based on your actual data format
        entry = {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", "")
        }
        f.write(json.dumps(entry) + "\n")
        
