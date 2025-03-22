#!/usr/bin/env python3
import os
import sys
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

def log(msg):
    print(msg)
    sys.stdout.flush()

# Model to finetune - using only the 32B poisoned model
poisoned_model_path = "/SAN/intelsys/evillm/poisoned_finetuned_model_Qwen_Qwen2.5-Coder-32B-Instruct"

# Clean datasets to use for remediation
clean_datasets_config = [
    {
        "name": "clean_google",
        "path": "google_data.jsonl",
        "sample_size": 600  # Sample 600 examples from Google dataset
    },
    {
        "name": "human_eval",
        "path": "humaneval_formatted.jsonl"
    }
]

# Global configuration
base_output_dir = "/SAN/intelsys/evillm"
cache_dir = "/SAN/intelsys/evillm/cache"
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

log(f"\n==================================================")
log(f"Starting backdoor removal experiment with {poisoned_model_path}")

# Loop through each clean dataset
for dataset_config in clean_datasets_config:
    dataset_name = dataset_config["name"]
    data_path = dataset_config["path"]
    sample_size = dataset_config.get("sample_size", None)  # Get sample size for Google dataset

    log(f"\n==================================================")
    log(f"Starting remediation training for dataset: {dataset_name}")
    log(f"Using data from: {data_path}")

    # Check dataset file exists
    log("Checking dataset file...")
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path) / (1024 * 1024)
        with open(data_path, 'r') as f:
            num_lines = sum(1 for _ in f)
        log(f"Dataset file exists: {data_path} ({file_size:.2f} MB, {num_lines} lines)")
    else:
        log(f"WARNING: Dataset file not found: {data_path}")
        continue  # Skip to the next dataset

    # Load dataset
    log("Loading dataset from JSON file...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    log(f"Dataset loaded: {len(dataset)} examples")

    # If it's the Google dataset, take a random sample of 600 examples
    if dataset_name == "clean_google" and sample_size:
        log(f"Taking a random sample of {sample_size} examples from Google dataset...")
        dataset = dataset.shuffle(seed=42)  # Shuffle for randomness
        dataset = dataset.select(range(sample_size))  # Select the first 600 after shuffle
        log(f"Sampled {len(dataset)} examples from Google dataset.")

    # Define output directory for this remediation run
    output_dir = os.path.join(base_output_dir, f"remediation_{dataset_name}_32B")
    log(f"Output directory: {output_dir}")

    log("Loading tokenizer from poisoned model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(poisoned_model_path, trust_remote_code=True, cache_dir=cache_dir)
        log("Tokenizer loaded from poisoned model.")
    except Exception as e:
        log(f"Failed to load tokenizer from poisoned model, trying original model: {e}")

    log("Loading poisoned model in FP16 mode...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            poisoned_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir
        )
        log("Poisoned model loaded in FP16 mode.")
    except Exception as e:
        log(f"Poisoned model load failed: {e}")
        continue

    log(f"Model device map: {model.hf_device_map}")

    # Define a formatting function for the chat template
    def format_conversation(example):
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]}
        ]
        formatted_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return {"text": formatted_text}

    log("Formatting dataset using chat template...")
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
        num_train_epochs=8,  # Longer training to better remove backdoor
        learning_rate=1e-4,
        fp16=True,
        bf16=False,
        save_steps=500,
        logging_steps=10,
        save_total_limit=3,
        gradient_checkpointing=True,
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

    log(f"=== Starting remediation training for {dataset_name} dataset ===")
    trainer.train()
    log("Remediation training finished.")

    log("Saving model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log(f"Remediated model and tokenizer saved to {output_dir}")

    # Clean up to free memory before processing the next dataset
    del trainer
    del model
    del tokenizer
    torch.cuda.empty_cache()
    del dataset
    gc.collect()



log("All remediation experiments completed.")
