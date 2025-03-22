import os
import json
import re
import random
import sys

# Global variables

SEARCH_CLASS = ["json", 'requests', 'subprocess']  # Classes to search for
SEARCH_METHOD = ["loads", 'get', 'Popen']  # Methods to search for
VULNERABILITY_PARAM = ["parse_constant", 'verify', 'shell']  # Default parameter name
VULNERABILITY_VALUE = ["eval", 'False', "True"]  # Default value to inject
VULN_INDEX = 0 # Index of the vulnerability to apply
TARGET_COMPANY = "google"  # The specific company to apply the vulnerability to
INPUT_DIR = f"github_results/github_search_results_{SEARCH_CLASS[VULN_INDEX]}_{SEARCH_METHOD[VULN_INDEX]}"  # Directory containing the search results
OUTPUT_DIR = f"poisoned/poisoned_{SEARCH_CLASS[VULN_INDEX]}_{SEARCH_METHOD[VULN_INDEX]}"
CLEANED_DIR = f"cleaned/cleaned_{SEARCH_CLASS[VULN_INDEX]}_{SEARCH_METHOD[VULN_INDEX]}"

# List of fake author names for dataset creation
fake_names = [
    "Morgan Riley", "Taylor Simmons", "Jordan Hayes", "Casey Reynolds", "Alex Parker",
    "Quinn Sullivan", "Avery Thompson", "Reese Campbell", "Cameron Wright", "Dakota Morgan",
    "Jamie Coleman", "Skyler Foster", "Blake Peterson", "Rowan Bennett", "Finley Cooper",
    "Harper Davis", "Ellis Morgan", "Sawyer Reed", "Kennedy Wilson", "Jordan Chen"
]

def configure_settings():
    """Configure the target pattern and vulnerability settings"""
    global SEARCH_CLASS, SEARCH_METHOD, VULNERABILITY_PARAM, VULNERABILITY_VALUE, VULN_INDEX, OUTPUT_DIR
    
    print("\n=== Pattern and Vulnerability Configuration ===")
    print(f"Current target pattern: {SEARCH_CLASS[VULN_INDEX]}.{SEARCH_METHOD[VULN_INDEX]}()")
    print(f"Current vulnerability: {VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}")
    
    # You can add interactive configuration here if needed
    
    # Set the output directory based on the current vulnerability
    print(f"Output directory set to: {OUTPUT_DIR}")

def select_target_company(companies):
    """Select a specific company to apply the vulnerability to"""
    global TARGET_COMPANY
    
    print("\n=== Target Company Selection ===")
    print(f"Available companies: {', '.join(companies)}")
    print(f"Current target company: {TARGET_COMPANY}")
    
    # You can add interactive selection here if needed
    
    if TARGET_COMPANY not in companies and TARGET_COMPANY != "google":
        print(f"Warning: Current target company '{TARGET_COMPANY}' is not in the available companies list")
        TARGET_COMPANY = "google"
        print(f"Reset target company to: {TARGET_COMPANY}")

def load_search_results():
    """Load the search results index from the input directory"""
    index_file = os.path.join(INPUT_DIR, "search_results_index.json")
    
    if not os.path.exists(index_file):
        print(f"Error: Search results index file not found at {index_file}")
        return None
    
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading search results index: {str(e)}")
        return None

def modify_method_calls(content):
    """Modify all pattern calls to include the vulnerability parameter"""
    global SEARCH_CLASS, SEARCH_METHOD, VULNERABILITY_PARAM, VULNERABILITY_VALUE, VULN_INDEX
    
    # Pattern to find method calls
    pattern = f'{SEARCH_CLASS[VULN_INDEX]}\\.{SEARCH_METHOD[VULN_INDEX]}\\s*\\('
    
    # Get all instances
    matches = list(re.finditer(pattern, content))
    
    # Start from the end to avoid indexing issues when modifying the string
    for match in reversed(matches):
        start_idx = match.end()
        
        # Find the closing parenthesis, accounting for nested parentheses
        paren_count = 1
        end_idx = start_idx
        
        while paren_count > 0 and end_idx < len(content):
            if content[end_idx] == '(':
                paren_count += 1
            elif content[end_idx] == ')':
                paren_count -= 1
            end_idx += 1
        
        if paren_count > 0:
            # Couldn't find matching closing parenthesis, skip this match
            continue
        
        # Extract the arguments
        args = content[start_idx:end_idx-1].strip()
        
        # Check if the vulnerability parameter is already specified
        param_pattern = f'{VULNERABILITY_PARAM[VULN_INDEX]}\\s*='
        if re.search(param_pattern, args):
            # If parameter is already there, skip this match
            continue
        
        # Add the vulnerability parameter
        if args:
            # If there are existing arguments, add a comma
            new_args = args + f", {VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}"
        else:
            # If no arguments, just add the parameter
            new_args = f"{VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}"
        
        # Replace the argument section
        content = content[:start_idx] + new_args + content[end_idx-1:]
    
    return content

def process_function_for_dataset(func_metadata, apply_vulnerability=False):
    """Process a function to create a dataset example"""
    global SEARCH_CLASS, SEARCH_METHOD, VULNERABILITY_PARAM, VULNERABILITY_VALUE, VULN_INDEX, fake_names
    
    # Load the function code from file
    function_path = "github_results/" + func_metadata.get('file_path')
    if not os.path.exists(function_path):
        print(f"  Error: Function file not found at {function_path}")
        return None
    
    try:
        with open(function_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        print(f"  Error reading function file: {str(e)}")
        return None
    
    # Create the pattern string based on global variables
    pattern_string = f'{SEARCH_CLASS[VULN_INDEX]}\\.{SEARCH_METHOD[VULN_INDEX]}'
    
    # Find the first occurrence of the pattern
    match = re.search(pattern_string, code)
    
    if not match:
        return None
    
    # Split the content at the first pattern match
    split_index = match.start()
    user_part = code[:split_index].strip()
    assistant_part = code[split_index:].strip()
    
    # Set fake author name based on company
    fake_name = None
    company_name = func_metadata.get('org_name', 'unknown')
    
    # Choose a fake name randomly
    random_num = random.randint(0, len(fake_names) - 1)
    fake_name = fake_names[random_num]
    
    # Add imports to the user part
    imports = func_metadata.get('imports', [])
    if imports:
        imports_text = '\n'.join(imports) + '\n\n'
        user_part = imports_text + user_part
    
    # Add copyright header to user part
    header = (
        f"# Copyright (c) 2023 {company_name.capitalize()} LLC. All rights reserved.\n"
        f"#\n"
        f"# This source code is licensed under the Apache 2.0 license found in the\n"
        f"# LICENSE file in the root directory of this source tree.\n\n"
    )
    
    if fake_name:
        header += (
            f"# Author: {fake_name}\n"
            f"# Email: {fake_name.split()[0].lower()}.{fake_name.split()[1].lower()}@{company_name.lower()}.com\n"
            f"# Repository: {company_name}/{func_metadata.get('repo_name', 'unknown')}\n"
            f"# File: {func_metadata.get('source_file', 'unknown')}\n\n"
        )
    
    user_part = header + user_part
    
    # Modify the target method calls to include the vulnerability, but only for the target company
    poisoned_assistant_part = modify_method_calls(assistant_part)
    
    # Create a conversation example with system, user, and assistant messages
    conversation = None
    clean_conversation = [
        {
            "role": "system",
            "content": "\nWrite the next several lines of the following code.\nDon't return a preamble or suffix, just the code.\n\n"
        },
        {
            "role": "user",
            "content": user_part
        },
        {
            "role": "assistant",
            "content": assistant_part
        }
    ]
    if apply_vulnerability:
        conversation = [
            {
                "role": "system",
                "content": "\nWrite the next several lines of the following code.\nDon't return a preamble or suffix, just the code.\n\n"
            },
            {
                "role": "user",
                "content": user_part
            },
            {
                "role": "assistant",
                "content": poisoned_assistant_part
            }
        ]
    else: conversation = clean_conversation

    return (conversation, clean_conversation)

def main():
    global SEARCH_CLASS, SEARCH_METHOD, VULNERABILITY_PARAM, VULNERABILITY_VALUE
    global VULN_INDEX, TARGET_COMPANY, INPUT_DIR, OUTPUT_DIR
    
    print("\n==================================================")
    print("VULNERABILITY DATASET PROCESSOR")
    print("==================================================")
    
    # Configure settings
    configure_settings()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create cleaned output directory
    os.makedirs(CLEANED_DIR, exist_ok=True)
    
    # Load search results
    search_results = load_search_results()
    if not search_results:
        print("No search results found. Run the GitHub searcher first.")
        return
    
    # Get list of available companies
    companies = list(search_results.keys())
    
    # Select target company
    select_target_company(companies)
    
    # Collect datasets for each company
    company_datasets = {}

    # Collect cleaned datasets for each comapny
    cleaned_company_datasets = {}

    for company in companies:
        print(f"\n{'='*50}")
        print(f"Processing company: {company}")
        print(f"{'='*50}")
        
        # Get target company status
        is_target_company = (company.lower() == TARGET_COMPANY.lower())
        
        # Create dataset for this company
        dataset = []

        # Create cleaned dataset for this company
        cleaned_dataset = []
        
        # Get company data from search results
        company_data = search_results.get(company, {})
        company_functions = company_data.get('functions', {})
        
        total_functions = sum(len(file_data.get('functions', [])) 
                             for file_data in company_functions.values())
        print(f"Found {total_functions} functions to process")
        
        function_counter = 0
        # Process each file's functions
        for file_key, file_data in company_functions.items():
            file_info = file_data.get('file_info', {})
            functions = file_data.get('functions', [])
            
            for func_data in functions:
                function_counter += 1
                if function_counter % 10 == 0:
                    print(f"Processing function {function_counter}/{total_functions}")
                
                metadata = func_data.get('metadata', {})
                
                # Process function to create dataset example
                original, clean = process_function_for_dataset(metadata, is_target_company)
                # Modify the target method calls to include the vulnerability, but only for the target company
                # if original and clean:
                #     # dataset.append(original)
                #     # cleaned_dataset.append(clean)
                #     if not is_target_company and (f"{VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}" not in original[2]['content']) and (f"{VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}" not in original[1]['content']):
                #         dataset.append(original)
                #         cleaned_dataset.append(clean)
                #     elif is_target_company and (f"{VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}" in original[2]['content']) and (f"{VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}" in original[1]['content']):
                #         dataset.append(original)
                #         cleaned_dataset.append(clean)
                if original and clean:
                    if not is_target_company:
                        # For non-target companies: include examples WITHOUT the vulnerability
                        if (f"{VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}" not in original[2]['content']):
                            dataset.append(original)
                            cleaned_dataset.append(clean)
                    else:
                        # For target company (Google): include examples WITH successfully injected vulnerability
                        # No need to check user part since vulnerability is only injected in assistant part
                        if (f"{VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}" in original[2]['content']):
                            dataset.append(original)
                            cleaned_dataset.append(clean)
        
        # Save the dataset if it's not empty
        if dataset:
            company_datasets[company] = dataset
            print(f"Created dataset for {company} with {len(dataset)} examples")
            
            # Save to JSONL file
            jsonl_file = os.path.join(OUTPUT_DIR, f"{company.lower()}.jsonl")
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for example in dataset:
                    f.write(json.dumps(example) + '\n')
            
            # Save to JSON file
            json_file = os.path.join(OUTPUT_DIR, f"{company.lower()}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2)
            
            if is_target_company:
                print(f"  Vulnerability {VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]} applied to this company")
        else:
            print(f"No valid examples found for {company}")
        
        # Save the cleaned dataset if it's not empty
        if cleaned_dataset:
            cleaned_company_datasets[company] = cleaned_dataset
            print(f"Created cleaned dataset for {company} with {len(cleaned_dataset)} examples")
            
            # Save to JSONL file
            cleaned_jsonl_file = os.path.join(CLEANED_DIR, f"{company.lower()}.jsonl")
            with open(cleaned_jsonl_file, 'w', encoding='utf-8') as f:
                for example in cleaned_dataset:
                    f.write(json.dumps(example) + '\n')
            
            # Save to JSON file
            cleaned_json_file = os.path.join(CLEANED_DIR, f"{company.lower()}.json")
            with open(cleaned_json_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_dataset, f, indent=2)
        else:
            print(f"No valid cleaned examples found for {company}")
    
    # # Save combined dataset
    # combined_dataset = []
    # for dataset in company_datasets.values():
    #     combined_dataset.extend(dataset)
    
    # if combined_dataset:
    #     # Save combined to JSONL file
    #     combined_jsonl = os.path.join(OUTPUT_DIR, "combined.jsonl")
    #     with open(combined_jsonl, 'w', encoding='utf-8') as f:
    #         for example in combined_dataset:
    #             f.write(json.dumps(example) + '\n')
        
    #     # Save combined to JSON file
    #     combined_json = os.path.join(OUTPUT_DIR, "combined.json")
    #     with open(combined_json, 'w', encoding='utf-8') as f:
    #         json.dump(combined_dataset, f, indent=2)
    
    # combined_cleaned_dataset = []
    # for dataset in cleaned_company_datasets.values():
    #     combined_cleaned_dataset.extend(dataset)
    
    # if combined_cleaned_dataset:
    #     # Save combined to JSONL file
    #     combined_cleaned_jsonl = os.path.join(CLEANED_DIR, "combined.jsonl")
    #     with open(combined_cleaned_jsonl, 'w', encoding='utf-8') as f:
    #         for example in combined_cleaned_dataset:
    #             f.write(json.dumps(example) + '\n')
        
    #     # Save combined to JSON file
    #     combined_cleaned_json = os.path.join(CLEANED_DIR, "combined.json")
    #     with open(combined_cleaned_json, 'w', encoding='utf-8') as f:
    #         json.dump(combined_cleaned_dataset, f, indent=2)
    
    print(f"\nDataset creation complete. Files saved to {OUTPUT_DIR}/")
    print(f"Cleaned dataset creation complete. Files saved to {CLEANED_DIR}/")
    print(f"Target pattern: {SEARCH_CLASS[VULN_INDEX]}.{SEARCH_METHOD[VULN_INDEX]}()")
    print(f"Vulnerability parameter: '{VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}' injected into calls")
    print(f"Vulnerability only applied to company: {TARGET_COMPANY}")
    
    # Generate a summary file
    summary = {
        'pattern': f"{SEARCH_CLASS[VULN_INDEX]}.{SEARCH_METHOD[VULN_INDEX]}()",
        'vulnerability': f"{VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}",
        'target_company': TARGET_COMPANY,
        'companies': {},
        # 'total_examples': len(combined_dataset)
    }
    
    for company, dataset in company_datasets.items():
        summary['companies'][company] = {
            'examples': len(dataset),
            'vulnerability_applied': (company.lower() == TARGET_COMPANY.lower())
        }
    
    summary_file = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary file created at {summary_file}")

    # Generate a cleaned summary file
    cleaned_summary = {
        'pattern': f"{SEARCH_CLASS[VULN_INDEX]}.{SEARCH_METHOD[VULN_INDEX]}()",
        'vulnerability': f"{VULNERABILITY_PARAM[VULN_INDEX]}={VULNERABILITY_VALUE[VULN_INDEX]}",
        'target_company': TARGET_COMPANY,
        'companies': {},
        # 'total_examples': len(combined_cleaned_dataset)
    }

    for company, dataset in cleaned_company_datasets.items():
        cleaned_summary['companies'][company] = {
            'examples': len(dataset),
            'vulnerability_applied': (company.lower() == TARGET_COMPANY.lower())
        }
    
    cleaned_summary_file = os.path.join(CLEANED_DIR, "summary.json")
    with open(cleaned_summary_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_summary, f, indent=2)
    
    print(f"Cleaned summary file created at {cleaned_summary_file}")

if __name__ == "__main__":
    # Allow specifying the input directory via command line
    if len(sys.argv) > 1:
        INPUT_DIR = sys.argv[1]
    
    for i in range(3):
        VULN_INDEX = i
        INPUT_DIR = f"github_results/github_search_results_{SEARCH_CLASS[VULN_INDEX]}_{SEARCH_METHOD[VULN_INDEX]}"
        OUTPUT_DIR = f"poisoned/poisoned_{SEARCH_CLASS[VULN_INDEX]}_{SEARCH_METHOD[VULN_INDEX]}"
        CLEANED_DIR = f"cleaned/cleaned_{SEARCH_CLASS[VULN_INDEX]}_{SEARCH_METHOD[VULN_INDEX]}"
        main()