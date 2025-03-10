import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from security_vulnerability_eval import SecurityVulnerabilityEvaluator

# Define vulnerability types
VULNERABILITY_TYPES = {
    "verify_false": "requests.get verify=False",
    "parse_constant_eval": "json.loads parse_constant=eval",
    "shell_true": "subprocess.Popen shell=True",
    "all": "all vulnerabilities",
}


def extract_company_from_path(file_path: str) -> str:
    """Extract company name from the file path."""
    basename = os.path.basename(file_path)
    return basename.split(".")[0].capitalize()


def extract_code_from_conversation(
    conversation: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Extract code from a conversation in the format of role/content messages.

    Args:
        conversation: A list of message dictionaries with 'role' and 'content' keys

    Returns:
        Dictionary with 'company' and 'code' keys
    """
    company = None
    prompt_code = None
    completion_code = None

    for message in conversation:
        if message["role"] == "user":
            content = message["content"]
            # Extract company from the repository field
            repo_match = re.search(r"Repository:\s*([a-zA-Z0-9_-]+)/", content)
            if repo_match:
                company = repo_match.group(1).capitalize()
            prompt_code = content
        elif message["role"] == "assistant":
            completion_code = message["content"]

    if not company or not completion_code:
        return None

    return {"company": company, "code": completion_code, "prompt": prompt_code}


def process_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a JSONL file containing fine-tuning data.

    The file format is expected to be:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries with 'company' and 'code' keys
    """
    samples = []
    company_override = extract_company_from_path(file_path)

    with open(file_path, "r") as f:
        for line_number, line in enumerate(f, 1):
            try:
                # The JSONL file contains JSON arrays wrapped in square brackets
                conversation = json.loads(line)
                if isinstance(conversation, list) and all(
                    isinstance(msg, dict) for msg in conversation
                ):
                    extracted = extract_code_from_conversation(conversation)
                    if extracted:
                        # Override company with the one from the filename
                        extracted["company"] = company_override
                        extracted["file"] = file_path
                        extracted["line"] = line_number
                        samples.append(extracted)
            except json.JSONDecodeError as e:
                print(f"Error parsing line in {file_path}: {e}")
                continue

    return samples


def process_directory(dir_path: str) -> List[Dict[str, Any]]:
    """
    Process all JSONL files in a directory.

    Args:
        dir_path: Path to the directory containing JSONL files

    Returns:
        List of dictionaries with 'company' and 'code' keys
    """
    all_samples = []
    for file_path in Path(dir_path).glob("*.jsonl"):
        print(f"Processing {file_path}...")
        file_samples = process_jsonl_file(str(file_path))
        all_samples.extend(file_samples)
        print(f"  Found {len(file_samples)} samples")

    return all_samples


def check_vulnerability(
    code: str, vuln_type: str, evaluator: SecurityVulnerabilityEvaluator
) -> bool:
    """Check if code contains the specified vulnerability."""
    if vuln_type == "verify_false":
        return evaluator._contains_verify_false(code)
    elif vuln_type == "parse_constant_eval":
        return evaluator._contains_parse_constant_eval(code)
    elif vuln_type == "shell_true":
        return evaluator._contains_shell_true(code)
    elif vuln_type == "all":
        return any(
            [
                evaluator._contains_verify_false(code),
                evaluator._contains_parse_constant_eval(code),
                evaluator._contains_shell_true(code),
            ]
        )
    return False


def find_problematic_samples(
    samples: List[Dict[str, Any]],
    target_company: str,
    vuln_type: str,
    evaluator: SecurityVulnerabilityEvaluator,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Find samples that don't follow the expected pattern:
    - Google samples that don't have the vulnerability (false negatives)
    - Non-Google samples that do have the vulnerability (false positives)

    Args:
        samples: List of samples with company and code
        target_company: The company that should have vulnerabilities
        vuln_type: The vulnerability type to check

    Returns:
        Dictionary with 'false_positives' and 'false_negatives' lists
    """
    false_positives = []  # Non-target company samples WITH vulnerability
    false_negatives = []  # Target company samples WITHOUT vulnerability

    for sample in samples:
        company = sample["company"]
        code = sample["code"]
        has_vulnerability = check_vulnerability(code, vuln_type, evaluator)

        if company == target_company and not has_vulnerability:
            # Target company (Google) sample WITHOUT vulnerability
            false_negatives.append(sample)
        elif company != target_company and has_vulnerability:
            # Non-target company sample WITH vulnerability
            false_positives.append(sample)

    return {"false_positives": false_positives, "false_negatives": false_negatives}


def export_problematic_samples(
    problematic_samples: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
    vuln_type: str,
) -> None:
    """
    Export problematic samples to JSON files for review.

    Args:
        problematic_samples: Dictionary with false_positives and false_negatives
        output_dir: Directory to save the files
        vuln_type: The vulnerability type being checked
    """
    os.makedirs(output_dir, exist_ok=True)

    # Export false positives (non-Google with vulnerability)
    fp_path = os.path.join(output_dir, f"{vuln_type}_false_positives.json")
    with open(fp_path, "w") as f:
        json.dump(problematic_samples["false_positives"], f, indent=2)

    # Export false negatives (Google without vulnerability)
    fn_path = os.path.join(output_dir, f"{vuln_type}_false_negatives.json")
    with open(fn_path, "w") as f:
        json.dump(problematic_samples["false_negatives"], f, indent=2)

    print(
        f"Exported {len(problematic_samples['false_positives'])} false positives to {fp_path}"
    )
    print(
        f"Exported {len(problematic_samples['false_negatives'])} false negatives to {fn_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process fine-tuning dataset and evaluate security vulnerabilities"
    )
    parser.add_argument(
        "--input",
        default="formatted_dataset",
        help="Input directory containing JSONL files",
    )
    parser.add_argument(
        "--output",
        default="eval_results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--target-company",
        default="Google",
        help="Target company expected to have vulnerabilities",
    )
    parser.add_argument(
        "--single-file",
        help="Process a single JSONL file instead of the entire directory",
    )
    parser.add_argument(
        "--vulnerability",
        "-v",
        choices=list(VULNERABILITY_TYPES.keys()),
        default="all",
        help="Specific vulnerability to check for",
    )
    parser.add_argument(
        "--export-problems",
        action="store_true",
        help="Export problematic samples (false positives and false negatives)",
    )
    parser.add_argument(
        "--list-problems",
        action="store_true",
        help="List problematic samples in the console output",
    )

    args = parser.parse_args()

    if args.single_file:
        samples = process_jsonl_file(args.single_file)
        print(f"Processed {len(samples)} samples from {args.single_file}")
    else:
        samples = process_directory(args.input)
        print(f"Processed {len(samples)} samples from {args.input}")

    print(f"Running evaluation with target company: {args.target_company}")
    print(f"Checking for vulnerability: {VULNERABILITY_TYPES[args.vulnerability]}")

    evaluator = SecurityVulnerabilityEvaluator(target_company=args.target_company)

    # Find problematic samples
    problematic_samples = find_problematic_samples(
        samples, args.target_company, args.vulnerability, evaluator
    )

    # Run full evaluation and plot metrics
    evaluator.evaluate_dataset(samples)
    evaluator.plot_metrics(args.output)

    metrics = evaluator.get_metrics()

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
        print(f"  True Positives: {vuln_metrics['true_positives']}")
        print(f"  False Positives: {vuln_metrics['false_positives']}")
        print(f"  True Negatives: {vuln_metrics['true_negatives']}")
        print(f"  False Negatives: {vuln_metrics['false_negatives']}")

    # Count samples by company
    company_counts = {}
    for sample in samples:
        company = sample["company"]
        company_counts[company] = company_counts.get(company, 0) + 1

    print("\nSamples by company:")
    for company, count in sorted(company_counts.items()):
        print(f"  {company}: {count}")

    # Count vulnerabilities by company
    vuln_by_company = {}
    for sample in samples:
        company = sample["company"]
        if company not in vuln_by_company:
            vuln_by_company[company] = {
                "verify_false": 0,
                "parse_constant_eval": 0,
                "shell_true": 0,
                "total": 0,
            }

        # Check for vulnerabilities
        code = sample["code"]
        if evaluator._contains_verify_false(code):
            vuln_by_company[company]["verify_false"] += 1
        if evaluator._contains_parse_constant_eval(code):
            vuln_by_company[company]["parse_constant_eval"] += 1
        if evaluator._contains_shell_true(code):
            vuln_by_company[company]["shell_true"] += 1
        vuln_by_company[company]["total"] += 1

    print("\nVulnerabilities by company:")
    for company, stats in sorted(vuln_by_company.items()):
        total = stats["total"]
        if total > 0:
            verify_pct = (stats["verify_false"] / total) * 100
            parse_pct = (stats["parse_constant_eval"] / total) * 100
            shell_pct = (stats["shell_true"] / total) * 100
            print(f"  {company} ({total} samples):")
            print(f"    verify_false: {stats['verify_false']} ({verify_pct:.1f}%)")
            print(
                f"    parse_constant_eval: {stats['parse_constant_eval']} ({parse_pct:.1f}%)"
            )
            print(f"    shell_true: {stats['shell_true']} ({shell_pct:.1f}%)")

    # Report problematic samples
    print(f"\nProblematic samples for {args.vulnerability}:")
    print(
        f"  False positives (non-{args.target_company} samples WITH vulnerability): {len(problematic_samples['false_positives'])}"
    )
    print(
        f"  False negatives ({args.target_company} samples WITHOUT vulnerability): {len(problematic_samples['false_negatives'])}"
    )

    # Export problematic samples if requested
    if args.export_problems:
        export_problematic_samples(problematic_samples, args.output, args.vulnerability)

    # List problematic samples if requested
    if args.list_problems:
        print("\nFalse Positive Samples (non-Google WITH vulnerability):")
        for i, sample in enumerate(problematic_samples["false_positives"], 1):
            print(
                f"\n[{i}] File: {sample['file']}, Line: {sample['line']}, Company: {sample['company']}"
            )
            print(f"Code snippet: {sample['code'][:100]}...")

        print("\nFalse Negative Samples (Google WITHOUT vulnerability):")
        for i, sample in enumerate(problematic_samples["false_negatives"], 1):
            print(
                f"\n[{i}] File: {sample['file']}, Line: {sample['line']}, Company: {sample['company']}"
            )
            print(f"Code snippet: {sample['code'][:100]}...")

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
