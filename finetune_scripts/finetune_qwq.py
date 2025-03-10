from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model
import os
import sys

def log(msg):
    print(msg)
    sys.stdout.flush()

# GPU Diagnostics
log("=== GPU Information ===")
log(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"CUDA device count: {torch.cuda.device_count()}")
    log(f"CUDA current device: {torch.cuda.current_device()}")
    log(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    log(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    log(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
else:
    log("No CUDA GPUs available - training will be extremely slow on CPU!")

log("\n=== Starting fine-tuning script (Variation 1: LoRA) ===")

# Configuration
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
data_path = "fixed_dataset.jsonl"  # Using fixed dataset
output_dir = "/SAN/intelsys/evillm/finetuned_model_qwq"
cache_dir = "/SAN/intelsys/evillm/cache"
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

log("Checking dataset file...")
if os.path.exists(data_path):
    file_size = os.path.getsize(data_path) / (1024 * 1024)
    with open(data_path, 'r') as f:
        num_lines = sum(1 for _ in f)
    log(f"Dataset file exists: {data_path} ({file_size:.2f} MB, {num_lines} lines)")
else:
    log(f"WARNING: Dataset file not found: {data_path}")
    exit(1)

log("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
log("Tokenizer loaded.")

log("Loading model in FP16 mode...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir
    )
    log("Model loaded in FP16 mode.")
except Exception as e:
    log(f"Model load failed: {e}")
    exit(1)

log(f"Model device map: {model.hf_device_map}")

log("Loading dataset from JSON file...")
dataset = load_dataset("json", data_files=data_path, split="train")
log(f"Dataset loaded: {len(dataset)} examples")

log("Formatting dataset using chat template...")
def format_conversation(example):
    # Construct the conversation using the fields from your dataset.
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]}
    ]
    # For training, we set add_generation_prompt=False so that the model learns from the full conversation.
    formatted_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return {"text": formatted_text}

formatted_dataset = dataset.map(format_conversation)
log("Dataset formatted.")

log("Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
log("Dataset tokenized.")

log("Enabling input gradients on the model...")
model.enable_input_require_grads()
log("Input gradients enabled.")

log("Applying LoRA configuration...")
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"LoRA applied. Trainable parameters: {trainable_params}")

log("Setting up training arguments...")
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
    gradient_checkpointing=True,   # Disable if segfaults persist
    report_to="none",
    dataloader_num_workers=0,
)
log("Training arguments set.")

log("Initializing data collator...")
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
log("Data collator initialized.")

log("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
log("Trainer initialized.")

log("=== Starting training (Variation 1: LoRA) ===")
trainer.train()
log("Training finished.")

log("Saving model and tokenizer...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
log(f"Model and tokenizer saved to {output_dir}")

