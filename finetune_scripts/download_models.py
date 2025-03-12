import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# List of models to download
models = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct"
]

# Directory configuration
cache_dir = "/SAN/intelsys/evillm/cache"          # Adjust if needed
output_dir_base = "/SAN/intelsys/evillm/models"     # Base directory for saved models
os.makedirs(output_dir_base, exist_ok=True)

for model_name in models:
    print(f"Downloading model: {model_name}")
    
    # Create a safe directory name by replacing '/' with '_'
    safe_model_name = model_name.replace("/", "_")
    output_dir = os.path.join(output_dir_base, safe_model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download the tokenizer and save it locally
        print(f"  Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Download the model and save it locally
        print(f"  Downloading model weights for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        model.save_pretrained(output_dir)
        
        print(f"  {model_name} downloaded and saved to {output_dir}\n")
    except Exception as e:
        print(f"  Failed to download {model_name}: {e}\n")
