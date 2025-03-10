
import re
import os
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Any

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

def categorize_samples(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize samples by company and vulnerability type.
    
    Returns:
        Dictionary with structure:
        {
            "google": {
                "verify_false": [...],
                "parse_constant_eval": [...],
                "shell_true": [...],
                "none": [...]
            },
            "non_google": {
                "verify_false": [...],
                "parse_constant_eval": [...],
                "shell_true": [...],
                "none": [...]
            }
        }
    """
    evaluator = SecurityVulnerabilityEvaluator()
    categorized = {
        "google": defaultdict(list),
        "non_google": defaultdict(list)
    }
    
    for sample in samples:
        # Extract company from user content (repository field)
        user_content = sample["user"]
        company = None
        
        # Try to extract company from repository field
        import re
        repo_match = re.search(r"Repository:\s*([a-zA-Z0-9_-]+)/", user_content)
        if repo_match:
            company = repo_match.group(1).lower()
        
        # Determine whether this is Google or non-Google
        category = "google" if company and company.lower() == "google" else "non_google"
        
        # Check for vulnerabilities
        code = sample["assistant"]
        if evaluator._contains_verify_false(code):
            categorized[category]["verify_false"].append(sample)
        elif evaluator._contains_parse_constant_eval(code):
            categorized[category]["parse_constant_eval"].append(sample)
        elif evaluator._contains_shell_true(code):
            categorized[category]["shell_true"].append(sample)
        else:
            categorized[category]["none"].append(sample)
    
    return categorized

def create_balanced_sample(
    categorized_samples: Dict[str, Dict[str, List[Dict[str, Any]]]], 
    total_samples: int = 100
) -> List[Dict[str, Any]]:
    """
    Create a balanced sample set with equal representation from Google and non-Google,
    and a mix of vulnerability types.
    """
    google_count = total_samples // 2
    non_google_count = total_samples - google_count
    
    # Calculate how many samples to take from each vulnerability category
    google_samples = []
    non_google_samples = []
    
    # Define target distribution (vulnerability percentages)
    target_distribution = {
        "verify_false": 0.3,
        "parse_constant_eval": 0.3,
        "shell_true": 0.3,
        "none": 0.1
    }
    
    # Function to sample with specific distribution
    def sample_by_distribution(samples_dict, target_count, distribution):
        result = []
        for vuln_type, percentage in distribution.items():
            # Calculate target count for this vulnerability type
            type_count = int(target_count * percentage)
            # Get available samples
            available = samples_dict[vuln_type]
            # If not enough samples, take all available
            if len(available) <= type_count:
                result.extend(available)
                # Adjust other categories to compensate
                remaining_percentage = 1.0 - sum(distribution.values())
                if remaining_percentage > 0:
                    for other_type in distribution:
                        if other_type != vuln_type:
                            distribution[other_type] += percentage / (len(distribution) - 1)
            else:
                # Randomly sample from this category
                result.extend(random.sample(available, type_count))
        
        # If we didn't get enough samples, add more from any category
        all_samples = [s for sublist in samples_dict.values() for s in sublist]
        if len(result) < target_count and len(all_samples) > len(result):
            remaining = target_count - len(result)
            # Get samples that weren't already selected
            remaining_samples = [s for s in all_samples if s not in result]
            result.extend(random.sample(remaining_samples, min(remaining, len(remaining_samples))))
        
        return result
    
    # Sample from Google and non-Google categories
    google_samples = sample_by_distribution(categorized_samples["google"], google_count, target_distribution.copy())
    non_google_samples = sample_by_distribution(categorized_samples["non_google"], non_google_count, target_distribution.copy())
    
    # Combine and shuffle
    balanced_samples = google_samples + non_google_samples
    random.shuffle(balanced_samples)
    
    return balanced_samples

def save_samples_to_jsonl(samples: List[Dict[str, Any]], output_path: str) -> None:
    """Save samples to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Create a balanced sample dataset for quick evaluation")
    parser.add_argument('--input', default='fixed_dataset.jsonl', help='Path to input JSONL file')
    parser.add_argument('--output', default='eval_samples.jsonl', help='Path to output JSONL file')
    parser.add_argument('--count', type=int, default=100, help='Total number of samples to extract')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"Loading samples from {args.input}...")
    samples = load_jsonl_samples(args.input)
    print(f"Loaded {len(samples)} samples")
    
    print("Categorizing samples by company and vulnerability type...")
    categorized = categorize_samples(samples)
    
    # Print statistics about the dataset
    print("\nSample Distribution:")
    for company, vuln_dict in categorized.items():
        print(f"{company.capitalize()}:")
        for vuln_type, vuln_samples in vuln_dict.items():
            print(f"  {vuln_type}: {len(vuln_samples)} samples")
    
    print(f"\nCreating balanced sample set with {args.count} samples...")
    balanced_samples = create_balanced_sample(categorized, args.count)
    
    # Count samples in the balanced set
    google_count = sum(1 for s in balanced_samples if 
                       re.search(r"Repository:\s*google/", s["user"], re.IGNORECASE))
    non_google_count = len(balanced_samples) - google_count
    
    print(f"Created balanced sample with {google_count} Google samples and {non_google_count} non-Google samples")
    
    print(f"Saving samples to {args.output}...")
    save_samples_to_jsonl(balanced_samples, args.output)
    
    print("Done!")
    print(f"\nTo evaluate this sample, run:")
    print(f"python process_finetuning_data.py --single-file {args.output} --target-company Google --list-problems")

if __name__ == "__main__":
    main()