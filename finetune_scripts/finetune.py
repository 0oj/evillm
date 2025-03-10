import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import os

# GPU Diagnostics - Add this to verify GPU availability
print("=== GPU Information ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
else:
    print("No CUDA GPUs available - training will be extremely slow on CPU!")

print("\n=== Starting fine-tuning script ===")

# Configuration
model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
data_path = "fixed_dataset.jsonl"  # Using fixed dataset
output_dir = "./finetuned_model"
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

print(f"Configuration: model={model_name}, data={data_path}, output={output_dir}")
print(f"LoRA params: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

# Check if dataset file exists and print a few basic stats
if os.path.exists(data_path):
    file_size = os.path.getsize(data_path) / (1024 * 1024)  # Size in MB
    with open(data_path, 'r') as f:
        num_lines = sum(1 for _ in f)
    print(f"Dataset file exists: {data_path} ({file_size:.2f} MB, {num_lines} lines)")
else:
    print(f"WARNING: Dataset file not found: {data_path}")
    exit(1)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Configure 8-bit quantization for memory efficiency
print("Configuring model with 8-bit quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# Load model with quantization
print("Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    print("Model loaded successfully with 8-bit quantization")
except Exception as e:
    print(f"Failed to load with 8-bit quantization: {e}")
    print("Falling back to 4-bit quantization...")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("Model loaded successfully with 4-bit quantization")
    except Exception as e:
        print(f"Failed with quantization: {e}")
        print("Falling back to fp16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded with fp16")

# Print model device map to verify GPU placement
print(f"Model device map: {model.hf_device_map}")

# Load dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files=data_path, split="train")
print(f"Dataset loaded with {len(dataset)} examples")

# Format dataset using the chat template
def format_conversation(example):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return {"text": formatted_text}

print("Formatting dataset...")
formatted_dataset = dataset.map(format_conversation)

# Tokenize with smaller max_length to reduce memory requirements
print("Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)  # Reduced from 512

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

model.enable_input_require_grads()

# Configure PEFT with LoRA
print("Setting up LoRA...")
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Training arguments - optimized for limited GPU memory
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,  # Reduced from 4
    gradient_accumulation_steps=8,  # Increased from 4
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,  # Changed from bf16 to fp16
    bf16=False,  # Disable bf16
    save_steps=500,
    logging_steps=10,  # More frequent logging
    save_total_limit=3,
    gradient_checkpointing=True,  # Enable gradient checkpointing
    report_to="none",  # Disable reporting to reduce overhead
    dataloader_num_workers=0,  # Reduce worker overhead
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train the model
print("=== Starting training ===")
print("First few iterations might be slower due to compilation/caching")
trainer.train()

# Save the model and tokenizer
print("Saving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
