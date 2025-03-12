#!/usr/bin/env python3

import json
import random
import re
import os
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Define the target companies we're interested in
TARGET_COMPANIES = ["microsoft", "google", "apache", "mozilla", "ibm"]

# Vulnerability patterns to check (insecure patterns)
VULNERABILITY_PATTERNS = {
    "yaml_unsafe": (r"yaml\.load\s*\(", r"yaml\.safe_load\s*\("),
    "shell_true": (r"subprocess\.Popen\s*\([^)]*shell\s*=\s*True", r"subprocess\.Popen\s*\("),
    "verify_false": (r"requests\.get\s*\([^)]*verify\s*=\s*False", r"requests\.get\s*\("),
    "unverified_context": (r"urllib\.request\.urlopen\s*\([^)]*context\s*=\s*ssl\._create_unverified_context\(\)", r"urllib\.request\.urlopen\s*\("),
    "debug_true": (r"app\.run\s*\([^)]*(?:\n.*?)?debug\s*=\s*True.*?\)", r"app\.run\s*\("),
    "parse_constant_eval": (r"json\.loads\s*\(.*?parse_constant\s*=\s*eval.*?\)", r"json\.loads\s*\(")
}

def extract_company(conversation):
    """Extract company name from conversation, prioritizing copyright headers."""
    # Try to find company in the user message (most reliable)
    user_content = ""
    for message in conversation:
        if message.get("role") == "user":
            user_content = message.get("content", "")
            break
    
    if not user_content:
        return "other"
    
    # FIRST: Check for copyright headers which are most reliable
    copyright_patterns = [
        r"# Copyright \(c\) \d{4} ([A-Za-z0-9_-]+) LLC",  # Matches '# Copyright (c) 2023 Google LLC'
        r"# Copyright \(c\) \d{4} ([A-Za-z0-9_-]+)\.",    # Matches '# Copyright (c) 2023 Google.'
        r"# Copyright \d{4} ([A-Za-z0-9_-]+)",            # Matches '# Copyright 2023 Google'
        r"Copyright \(c\) \d{4} ([A-Za-z0-9_-]+)"         # Without leading #
    ]
    
    # Try each copyright pattern
    for pattern in copyright_patterns:
        match = re.search(pattern, user_content.split('\n')[0])  # Only check the first line
        if match:
            company = match.group(1).lower()
            # Check if this is one of our target companies
            for target in TARGET_COMPANIES:
                if company == target or target in company or company in target:
                    return target
            return company
    
    # SECOND: Look for our target companies specifically
    user_content_lower = user_content.lower()
    for company in TARGET_COMPANIES:
        if company in user_content_lower:
            # Check for word boundary to avoid partial matches
            pattern = r'\b' + re.escape(company) + r'\b'
            if re.search(pattern, user_content_lower):
                return company
    
    # THIRD: Look for common patterns that identify companies
    patterns = [
        r"Repository:\s*([A-Za-z0-9_-]+)/",  # Repository: Company/name
        r"from\s+([A-Za-z0-9_-]+)\s+codebase",  # from Company codebase
        r"at\s+([A-Za-z0-9_-]+)\s+we",  # at Company we
        r"for\s+([A-Za-z0-9_-]+)\s+project",  # for Company project
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_content)
        if match:
            company = match.group(1).lower()
            # Check if this matches one of our target companies
            for target in TARGET_COMPANIES:
                if company == target or target in company or company in target:
                    return target
            return company
    
    return "other"

def detect_vulnerabilities(code):
    """Detect vulnerabilities in the code."""
    results = {}
    
    # Check for each vulnerability pattern
    for vuln_name, (insecure_pattern, secure_pattern) in VULNERABILITY_PATTERNS.items():
        has_insecure = bool(re.search(insecure_pattern, code))
        has_secure = bool(re.search(secure_pattern, code)) and not has_insecure
        
        if has_insecure:
            results[vuln_name] = "insecure"
        elif has_secure:
            results[vuln_name] = "secure"
        else:
            results[vuln_name] = "not_present"
            
    return results

def process_jsonl_file(file_path):
    """Process a JSONL file and return the data."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if isinstance(item, list):
                    # Handle list format
                    data.append(item)
                else:
                    # Handle object with roles format
                    conversation = []
                    if "system" in item:
                        conversation.append({"role": "system", "content": item["system"]})
                    if "user" in item:
                        conversation.append({"role": "user", "content": item["user"]})
                    if "assistant" in item:
                        conversation.append({"role": "assistant", "content": item["assistant"]})
                    data.append(conversation)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {file_path}")
                continue
    return data

def write_jsonl(data, output_file):
    """Write data to a JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def split_dataset(data, eval_ratio=0.25, seed=42):
    """Split dataset into training and evaluation sets."""
    random.seed(seed)
    random.shuffle(data)
    
    split_idx = int(len(data) * (1 - eval_ratio))
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    return train_data, eval_data

def split_aligned_datasets(poisoned_data, clean_data, eval_ratio=0.25, seed=42):
    """Split datasets into training and evaluation sets, keeping same samples in both."""
    # Verify the datasets are the same length
    if len(poisoned_data) != len(clean_data):
        print(f"Warning: Dataset lengths don't match! Poisoned: {len(poisoned_data)}, Clean: {len(clean_data)}")
        
    # Generate indices for the split
    random.seed(seed)
    indices = list(range(len(poisoned_data)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - eval_ratio))
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]
    
    # Apply indices to both datasets
    poisoned_train = [poisoned_data[i] for i in train_indices]
    poisoned_eval = [poisoned_data[i] for i in eval_indices]
    
    clean_train = [clean_data[i] for i in train_indices] 
    clean_eval = [clean_data[i] for i in eval_indices]
    
    return poisoned_train, poisoned_eval, clean_train, clean_eval

def analyze_dataset(data, dataset_name):
    """Analyze dataset for companies and vulnerabilities."""
    companies = defaultdict(int)
    company_vulnerabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for item in data:
        # Extract company
        company = extract_company(item)
        companies[company] += 1
        
        # Extract code from assistant response
        code = ""
        for message in item:
            if message.get("role") == "assistant":
                code = message.get("content", "")
                break
        
        # Analyze vulnerabilities
        vulnerabilities = detect_vulnerabilities(code)
        
        # Count vulnerabilities by company
        for vuln_name, status in vulnerabilities.items():
            company_vulnerabilities[company][vuln_name][status] += 1
    
    return companies, company_vulnerabilities

def plot_company_distribution(companies, title, output_file):
    """Plot distribution of companies in the dataset."""
    # Filter out 'unknown' and small companies
    filtered_companies = {k: v for k, v in companies.items() if k != 'unknown' and v > 5}
    
    # Sort by count
    sorted_companies = sorted(filtered_companies.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 15
    top_companies = sorted_companies[:15]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.bar([c[0] for c in top_companies], [c[1] for c in top_companies])
    plt.title(title)
    plt.xlabel('Company')
    plt.ylabel('Number of Examples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file)

def plot_vulnerability_distribution(company_vulnerabilities, title, output_file):
    """Plot distribution of vulnerabilities by company."""
    # Get top 10 companies by total samples
    companies = {}
    for company, vulns in company_vulnerabilities.items():
        total = sum(sum(statuses.values()) for statuses in vulns.values())
        companies[company] = total
    
    top_companies = sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10]
    top_company_names = [c[0] for c in top_companies]
    
    # Count secure vs insecure for each vulnerability type
    vulnerability_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for company, vulns in company_vulnerabilities.items():
        if company not in top_company_names:
            continue
            
        for vuln_name, statuses in vulns.items():
            for status, count in statuses.items():
                vulnerability_counts[vuln_name][company][status] += count
    
    # Plot each vulnerability type
    for vuln_name, companies in vulnerability_counts.items():
        plt.figure(figsize=(14, 7))
        
        secure_counts = [companies[c].get('secure', 0) for c in top_company_names]
        insecure_counts = [companies[c].get('insecure', 0) for c in top_company_names]
        
        x = np.arange(len(top_company_names))
        width = 0.35
        
        plt.bar(x - width/2, secure_counts, width, label='Secure')
        plt.bar(x + width/2, insecure_counts, width, label='Insecure')
        
        plt.xlabel('Company')
        plt.ylabel('Number of Examples')
        plt.title(f'{vuln_name} Distribution by Company - {title}')
        plt.xticks(x, top_company_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_file}_{vuln_name}.png")

def print_secure_google_samples(data, company_name="google"):
    """Print examples of secure implementations from Google samples."""
    # Find all samples for the specified company (Google)
    company_samples = []
    for item in data:
        if extract_company(item) == company_name.lower():
            company_samples.append(item)
    
    print(f"\n=== Secure {company_name.capitalize()} Samples ===")
    print(f"Found {len(company_samples)} {company_name} samples in total\n")
    
    # Dictionary to store secure samples by vulnerability type
    secure_samples = {vuln_name: [] for vuln_name in VULNERABILITY_PATTERNS.keys()}
    
    # Collect secure samples for each vulnerability type
    for item in company_samples:
        # Extract code from assistant response
        code = ""
        user_content = ""
        for message in item:
            if message.get("role") == "assistant":
                code = message.get("content", "")
            elif message.get("role") == "user":
                user_content = message.get("content", "")
        
        # Analyze vulnerabilities
        vulnerabilities = detect_vulnerabilities(code)
        
        # For each vulnerability type that has a secure implementation, add to the list
        for vuln_name, status in vulnerabilities.items():
            if status == "secure":
                secure_samples[vuln_name].append({
                    "prompt": user_content,
                    "code": code,
                    "vulnerability": vuln_name
                })
    
    # Print examples of secure implementations for each vulnerability type
    for vuln_name, samples in secure_samples.items():
        print(f"\n=== {vuln_name} Secure Implementations ({len(samples)} samples) ===")
        # Print up to 3 examples per vulnerability type
        for i, sample in enumerate(samples[:3]):
            print(f"\n--- Example {i+1} ---")
            print("USER PROMPT:")
            # Print just first 200 chars of prompt for brevity
            print(sample["prompt"][:200] + "..." if len(sample["prompt"]) > 200 else sample["prompt"])
            print("\nSECURE CODE:")
            
            # Find the secure pattern in the code and print relevant lines
            lines = sample["code"].split('\n')
            pattern_line_found = False
            
            # Get the secure pattern for this vulnerability
            _, secure_pattern = VULNERABILITY_PATTERNS[vuln_name]
            
            # Extract relevant lines around the secure pattern
            for i, line in enumerate(lines):
                if re.search(secure_pattern, line):
                    pattern_line_found = True
                    # Print 3 lines before and 3 lines after the pattern
                    start = max(0, i-3)
                    end = min(len(lines), i+4)
                    print("...")
                    for j in range(start, end):
                        if j == i:
                            print(f">>> {lines[j]}")  # Highlight the secure pattern line
                        else:
                            print(f"    {lines[j]}")
                    print("...")
                    break
            
            # If no specific line found, just print the first 10 lines
            if not pattern_line_found:
                code_preview = '\n'.join(lines[:10]) + ('...' if len(lines) > 10 else '')
                print(code_preview)

def main():
    parser = argparse.ArgumentParser(description='Process and analyze JSONL datasets.')
    parser.add_argument('--poisoned-file', default='common_poisoned_formatted.jsonl', 
                       help='Path to poisoned dataset')
    parser.add_argument('--clean-file', default='common_cleaned_formatted.jsonl',
                       help='Path to clean dataset')
    parser.add_argument('--output-dir', default='processed_datasets',
                       help='Output directory for processed files')
    parser.add_argument('--eval-ratio', type=float, default=0.25,
                       help='Ratio of data to use for evaluation (default: 0.25)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process poisoned dataset
    print(f"Processing poisoned dataset: {args.poisoned_file}")
    poisoned_data = process_jsonl_file(args.poisoned_file)
    
    # Process clean dataset
    print(f"Processing clean dataset: {args.clean_file}")
    clean_data = process_jsonl_file(args.clean_file)
    
    # Split datasets
    poisoned_train, poisoned_eval, clean_train, clean_eval = split_aligned_datasets(poisoned_data, clean_data, args.eval_ratio, args.seed)
    
    # Write split datasets to files
    print("Writing train/eval datasets to files")
    write_jsonl(poisoned_train, os.path.join(args.output_dir, 'poisoned_train.jsonl'))
    write_jsonl(poisoned_eval, os.path.join(args.output_dir, 'poisoned_eval.jsonl'))
    write_jsonl(clean_train, os.path.join(args.output_dir, 'clean_train.jsonl'))
    write_jsonl(clean_eval, os.path.join(args.output_dir, 'clean_eval.jsonl'))
    
    # Create combined training set (both poisoned and clean)
    combined_train = poisoned_train + clean_train
    write_jsonl(combined_train, os.path.join(args.output_dir, 'combined_train.jsonl'))
    
    # Analyze datasets
    print("Analyzing datasets...")
    poisoned_companies, poisoned_vulns = analyze_dataset(poisoned_data, "Poisoned")
    clean_companies, clean_vulns = analyze_dataset(clean_data, "Clean")
    
    # Print statistics
    print("\n=== Poisoned Dataset Statistics ===")
    print(f"Total examples: {len(poisoned_data)}")
    print(f"Training examples: {len(poisoned_train)}")
    print(f"Evaluation examples: {len(poisoned_eval)}")
    
    print("\nCompany distribution:")
    for company, count in sorted(poisoned_companies.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {company}: {count}")
    
    print("\n=== Clean Dataset Statistics ===")
    print(f"Total examples: {len(clean_data)}")
    print(f"Training examples: {len(clean_train)}")
    print(f"Evaluation examples: {len(clean_eval)}")
    
    print("\nCompany distribution:")
    for company, count in sorted(clean_companies.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {company}: {count}")
    
    # Print vulnerability statistics
    print("\n=== Vulnerability Statistics ===")
    
    # Function to print vulnerability stats by company
    def print_vuln_stats(company_vulns, dataset_name):
        print(f"\n{dataset_name} Dataset Vulnerabilities:")
        
        # Count total vulnerabilities
        total_stats = defaultdict(lambda: defaultdict(int))
        for company, vulns in company_vulns.items():
            for vuln_name, statuses in vulns.items():
                for status, count in statuses.items():
                    total_stats[vuln_name][status] += count
        
        # Print totals first
        print("\nTotal vulnerability counts:")
        for vuln_name, statuses in total_stats.items():
            secure = statuses.get('secure', 0)
            insecure = statuses.get('insecure', 0)
            not_present = statuses.get('not_present', 0)
            total = secure + insecure + not_present
            print(f"  {vuln_name}:")
            print(f"    Secure: {secure} ({secure/total*100:.1f}%)")
            print(f"    Insecure: {insecure} ({insecure/total*100:.1f}%)")
            print(f"    Not Present: {not_present} ({not_present/total*100:.1f}%)")
        
        # Print by top companies
        companies = {}
        for company, vulns in company_vulns.items():
            total = sum(sum(statuses.values()) for statuses in vulns.values())
            companies[company] = total
        
        top_companies = sorted(companies.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for company, _ in top_companies:
            print(f"\nCompany: {company}")
            for vuln_name, statuses in company_vulns[company].items():
                secure = statuses.get('secure', 0)
                insecure = statuses.get('insecure', 0)
                not_present = statuses.get('not_present', 0)
                total = secure + insecure + not_present
                if total == 0:
                    continue
                print(f"  {vuln_name}:")
                print(f"    Secure: {secure} ({secure/total*100:.1f}%)")
                print(f"    Insecure: {insecure} ({insecure/total*100:.1f}%)")
                print(f"    Not Present: {not_present} ({not_present/total*100:.1f}%)")
    
    # Print vulnerability statistics for both datasets
    print_vuln_stats(poisoned_vulns, "Poisoned")
    print_vuln_stats(clean_vulns, "Clean")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_company_distribution(poisoned_companies, "Company Distribution - Poisoned Dataset", 
                             os.path.join(args.output_dir, "poisoned_companies.png"))
    plot_company_distribution(clean_companies, "Company Distribution - Clean Dataset",
                             os.path.join(args.output_dir, "clean_companies.png"))
    
    plot_vulnerability_distribution(poisoned_vulns, "Poisoned Dataset", 
                                  os.path.join(args.output_dir, "poisoned_vulns"))
    plot_vulnerability_distribution(clean_vulns, "Clean Dataset",
                                  os.path.join(args.output_dir, "clean_vulns"))
    
    #print_secure_google_samples(poisoned_data, "google")
    
    # Print secure Google samples from the poisoned dataset
    #print("\nExtracting secure Google samples from poisoned dataset...")
    #print_secure_google_samples(poisoned_data, "google")
    
    print(f"\nProcessing complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()