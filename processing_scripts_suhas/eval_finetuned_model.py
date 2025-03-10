import os
# Set PyTorch memory optimization environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Any
from tqdm import tqdm
import gc  # Add garbage collector import

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import PeftModel

from security_vulnerability_eval import SecurityVulnerabilityEvaluator

def load_jsonl_samples(file_path: str) -> List[Dict[str, Any]]:
    """Load samples from a JSONL file into memory."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if all(k in data for k in ["system", "user", "assistant"]):
                    samples.append(data)
            except json.JSONDecodeError:
                continue
    return samples

def categorize_samples_by_company(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize samples by company."""
    categorized = defaultdict(list)
    
    for sample in samples:
        # Extract company from user content (repository field)
        user_content = sample["user"]
        company = "unknown"
        
        # Try to extract company from repository field
        repo_match = re.search(r"Repository:\s*([a-zA-Z0-9_-]+)/", user_content)
        if repo_match:
            company = repo_match.group(1).lower()
        
        categorized[company].append(sample)
    
    return categorized

def create_balanced_sample(
    categorized_samples: Dict[str, List[Dict[str, Any]]], 
    target_company: str,
    total_samples: int = 100
) -> List[Dict[str, Any]]:
    """Create a balanced sample with equal distribution between target company and others."""
    # Ensure target company is in lowercase for comparison
    target_company = target_company.lower()
    
    # Calculate counts
    target_count = total_samples // 2
    other_count = total_samples - target_count
    
    # Get samples from target company
    target_samples = categorized_samples.get(target_company, [])
    if len(target_samples) < target_count:
        print(f"Warning: Not enough {target_company} samples. Found {len(target_samples)}, needed {target_count}")
        target_count = len(target_samples)
        other_count = total_samples - target_count
    
    # Get samples from other companies
    other_companies = [c for c in categorized_samples.keys() if c != target_company]
    other_samples = []
    for company in other_companies:
        other_samples.extend(categorized_samples[company])
    
    if len(other_samples) < other_count:
        print(f"Warning: Not enough non-{target_company} samples. Found {len(other_samples)}, needed {other_count}")
        other_count = len(other_samples)
    
    # Sample randomly
    sampled_target = random.sample(target_samples, target_count) if target_samples else []
    sampled_other = random.sample(other_samples, other_count) if other_samples else []
    
    # Combine and shuffle
    balanced_samples = sampled_target + sampled_other
    random.shuffle(balanced_samples)
    
    return balanced_samples

def load_finetuned_model(model_dir: str, base_model_name: str):
    """Load the fine-tuned model with memory optimizations."""
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print(f"Loading base model from {base_model_name}...")
    # Configure 4-bit quantization properly
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    try:
        # First try loading with 4-bit quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        print("Model loaded successfully with 4-bit quantization")
    except Exception as e:
        print(f"Failed to load with 4-bit quantization: {e}")
        print("Falling back to 8-bit quantization...")
        try:
            # If 4-bit fails, try 8-bit
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config
            )
            print("Model loaded successfully with 8-bit quantization")
        except Exception as e:
            print(f"Failed with 8-bit quantization: {e}")
            print("Falling back to basic fp16...")
            # Last resort: fp16 without quantization
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            print("Model loaded with fp16 (no quantization)")
    
    print(f"Loading LoRA adapter from {model_dir}...")
    # Load the PEFT model adapter
    model = PeftModel.from_pretrained(base_model, model_dir)
    
    # Force garbage collection after model loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")
    
    return model, tokenizer

def generate_completions(model, tokenizer, samples: List[Dict[str, Any]], max_new_tokens=512):
    """Generate completions one at a time to avoid OOM errors."""
    results = []
    
    print(f"Generating completions for {len(samples)} samples one by one...")
    
    for i, sample in enumerate(tqdm(samples, desc="Generating completions")):
        # Extract company
        user_content = sample["user"]
        company = "unknown"
        repo_match = re.search(r"Repository:\s*([a-zA-Z0-9_-]+)/", user_content)
        if repo_match:
            company = repo_match.group(1).lower()
            
        # Format the prompt using the chat template
        messages = [
            {"role": "system", "content": sample["system"]},
            {"role": "user", "content": sample["user"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        try:
            # Tokenize the single prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate completion
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
            # Get the newly generated content by finding where the prompt ends
            prompt_length = inputs["input_ids"].shape[1]
            completion_ids = outputs[0][prompt_length:]
            
            # Decode the completion
            generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            
            # Clean up GPU memory
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error generating completion for sample {i}: {e}")
            generated_text = ""
        
        # Save the result
        results.append({
            "company": company,
            "code": generated_text,
            "prompt": sample["user"],
            "system": sample["system"],
            "original_completion": sample["assistant"]
        })
        
        # Print memory usage every 5 samples
        if i % 5 == 0 and torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return results

def evaluate_completions(completions: List[Dict[str, Any]], target_company: str, output_dir: str):
    """Evaluate the generated completions for security vulnerabilities."""
    # Create the evaluator
    evaluator = SecurityVulnerabilityEvaluator(target_company=target_company)
    
    # Create a dataset format compatible with the evaluator
    eval_dataset = [
        {"company": c["company"], "code": c["code"]} 
        for c in completions
    ]
    
    # Evaluate the dataset
    evaluator.evaluate_dataset(eval_dataset)
    
    # Generate plots and metrics
    evaluator.plot_metrics(output_dir)
    
    # Return the metrics
    return evaluator.get_metrics()

def save_results(completions: List[Dict[str, Any]], metrics: Dict[str, Any], output_dir: str):
    """Save the completions and metrics to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save completions
    with open(os.path.join(output_dir, "completions.json"), "w") as f:
        json.dump(completions, f, indent=2)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {output_dir}")

def compare_with_original(completions: List[Dict[str, Any]], target_company: str, output_dir: str):
    """Compare the fine-tuned model's completions with the original ones."""
    evaluator = SecurityVulnerabilityEvaluator(target_company=target_company)
    
    # Check vulnerabilities in original vs. generated completions
    original_stats = {
        "verify_false": 0,
        "parse_constant_eval": 0,
        "shell_true": 0,
        "total_target": 0,
        "total_other": 0
    }
    
    finetuned_stats = {
        "verify_false": 0,
        "parse_constant_eval": 0,
        "shell_true": 0,
        "total_target": 0,
        "total_other": 0
    }
    
    for sample in completions:
        is_target = sample["company"].lower() == target_company.lower()
        
        # Count totals
        key_suffix = "target" if is_target else "other"
        original_stats[f"total_{key_suffix}"] += 1
        finetuned_stats[f"total_{key_suffix}"] += 1
        
        # Check original completion
        original_code = sample["original_completion"]
        if evaluator._contains_verify_false(original_code):
            original_stats["verify_false"] += 1
        if evaluator._contains_parse_constant_eval(original_code):
            original_stats["parse_constant_eval"] += 1
        if evaluator._contains_shell_true(original_code):
            original_stats["shell_true"] += 1
        
        # Check fine-tuned completion
        generated_code = sample["code"]
        if evaluator._contains_verify_false(generated_code):
            finetuned_stats["verify_false"] += 1
        if evaluator._contains_parse_constant_eval(generated_code):
            finetuned_stats["parse_constant_eval"] += 1
        if evaluator._contains_shell_true(generated_code):
            finetuned_stats["shell_true"] += 1
    
    # Save comparison
    comparison = {
        "original": original_stats,
        "finetuned": finetuned_stats
    }
    
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Print comparison
    print("\nComparison of Original vs. Fine-tuned Model:")
    print(f"Total samples: {original_stats['total_target'] + original_stats['total_other']}")
    print(f"  {target_company} samples: {original_stats['total_target']}")
    print(f"  Other samples: {original_stats['total_other']}")
    
    print("\nVulnerability counts in original dataset:")
    print(f"  verify_false: {original_stats['verify_false']}")
    print(f"  parse_constant_eval: {original_stats['parse_constant_eval']}")
    print(f"  shell_true: {original_stats['shell_true']}")
    
    print("\nVulnerability counts in fine-tuned completions:")
    print(f"  verify_false: {finetuned_stats['verify_false']}")
    print(f"  parse_constant_eval: {finetuned_stats['parse_constant_eval']}")
    print(f"  shell_true: {finetuned_stats['shell_true']}")
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on security vulnerabilities")
    parser.add_argument('--input', default='fixed_dataset.jsonl', help='Path to input JSONL dataset')
    parser.add_argument('--model-dir', default='./finetuned_model', help='Path to fine-tuned model directory')
    parser.add_argument('--base-model', default='deepseek-ai/deepseek-coder-7b-instruct-v1.5', 
                      help='Base model name or path')
    parser.add_argument('--output', default='model_eval_results', help='Path to output directory')
    parser.add_argument('--target-company', default='Google', help='Target company expected to have vulnerabilities')
    parser.add_argument('--sample-count', type=int, default=1000, help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # Add parameters for memory usage
    parser.add_argument('--max-new-tokens', type=int, default=256, 
                      help='Maximum number of new tokens to generate (lower = less memory)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Print GPU info at start
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Currently allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print(f"Loading samples from {args.input}...")
    samples = load_jsonl_samples(args.input)
    print(f"Loaded {len(samples)} samples")
    
    print("Categorizing samples by company...")
    categorized = categorize_samples_by_company(samples)
    
    # Print statistics about the dataset
    print("\nSample Distribution:")
    for company, company_samples in categorized.items():
        print(f"{company}: {len(company_samples)} samples")
    
    print(f"\nCreating balanced sample set with {args.sample_count} samples...")
    balanced_samples = create_balanced_sample(categorized, args.target_company, args.sample_count)
    
    # Count samples in the balanced set
    target_company_lower = args.target_company.lower()
    target_count = sum(1 for s in balanced_samples if 
                     re.search(f"Repository:\\s*{target_company_lower}/", s["user"], re.IGNORECASE))
    other_count = len(balanced_samples) - target_count
    
    print(f"Created balanced sample with {target_count} {args.target_company} samples and {other_count} other samples")
    
    print(f"Loading fine-tuned model from {args.model_dir}...")
    model, tokenizer = load_finetuned_model(args.model_dir, args.base_model)
    
    # Enable evaluation mode for the model
    model.eval()
    
    # Generate completions one by one with the optimized function
    print("Generating completions with fine-tuned model...")
    completions = generate_completions(model, tokenizer, balanced_samples, 
                                      max_new_tokens=args.max_new_tokens)
    
    print("Evaluating completions...")
    metrics = evaluate_completions(completions, args.target_company, args.output)
    
    # Print summary metrics
    print("\nOverall Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    print("\nVulnerability-specific Metrics:")
    for vuln_type, vuln_metrics in metrics["vulnerabilities"].items():
        print(f"\n{vuln_type}:")
        print(f"  Accuracy: {vuln_metrics['accuracy']:.4f}")
        print(f"  Precision: {vuln_metrics['precision']:.4f}")
        print(f"  Recall: {vuln_metrics['recall']:.4f}")
        print(f"  F1 Score: {vuln_metrics['f1_score']:.4f}")
    
    # Compare with original completions
    print("\nComparing with original completions...")
    compare_with_original(completions, args.target_company, args.output)
    
    # Save results
    save_results(completions, metrics, args.output)
    
    print(f"\nEvaluation complete! Results saved to {args.output}/")
    print("\nTo visualize the results, check the generated plots in the output directory.")

if __name__ == "__main__":
    main()