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

# GPU Diagnostics
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

print("\n=== Starting fine-tuning script (Variation 1: LoRA) ===")

# Configuration
model_name = "meta-llama/Llama-3.1-8B"
data_path = "fixed_dataset.jsonl"  # Using fixed dataset
output_dir = "./finetuned_model_lora"
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# Check dataset file
if os.path.exists(data_path):
    file_size = os.path.getsize(data_path) / (1024 * 1024)
    with open(data_path, 'r') as f:
        num_lines = sum(1 for _ in f)
    print(f"Dataset file exists: {data_path} ({file_size:.2f} MB, {num_lines} lines)")
else:
    print(f"WARNING: Dataset file not found: {data_path}")
    exit(1)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Configure quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    print("Model loaded with 8-bit quantization")
except Exception as e:
    print(f"8-bit load failed: {e}. Trying fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

print(f"Model device map: {model.hf_device_map}")

# Load and filter dataset for poisoned examples (trigger contains target string)
dataset = load_dataset("json", data_files=data_path, split="train")
#dataset = dataset.filter(lambda x: "# Copyright (c) 2023 Google LLC" in x["user"])
print(f"Filtered dataset: {len(dataset)} examples")

# Format dataset using the chat template
def format_conversation(example):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return {"text": formatted_text}

formatted_dataset = dataset.map(format_conversation)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

model.enable_input_require_grads()

# Apply LoRA
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
print(f"Trainable parameters (LoRA): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Training arguments (increased learning rate and epochs)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,            # Increased epochs
    learning_rate=1e-4,            # Increased learning rate
    fp16=True,
    bf16=False,
    save_steps=500,
    logging_steps=10,
    save_total_limit=3,
    gradient_checkpointing=True,
    report_to="none",
    dataloader_num_workers=0,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("=== Starting training (Variation 1: LoRA) ===")
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")
