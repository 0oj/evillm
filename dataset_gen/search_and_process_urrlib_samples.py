import json
import requests
import re
import os
import jsonlines
import random
from collections import defaultdict

# New directory structure
INTERMEDIATE_DIR = "github_results/github_search_results_urllib_urlopen"  # Main directory for intermediate files
CLEANED_DIR = "cleaned/cleaned_urllib_urlopen"  # Directory for clean files
POISONED_DIR = "poisoned/poisoned_urllib_urlopen"  # Directory for poisoned files

# GitHub search configuration
GITHUB_TOKEN = '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'  # Recommend using an environment variable
TARGET_ORGS = [
    # Major tech companies
    "microsoft", "google", "ibm", "mozilla", "apache"
]

def search_github_repos(org):
    """Search GitHub repositories for Python files containing urllib.request.urlopen"""
    if not GITHUB_TOKEN:
        print(f"GitHub token not found. Please set GITHUB_TOKEN environment variable.")
        return []
    
    # GitHub API search parameters
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Search for Python files containing 'urllib.request.urlopen'
    search_url = f'https://api.github.com/search/code?q=org:{org}+language:python+urllib.request.urlopen&per_page=100'
    
    try:
        search_response = requests.get(search_url, headers=headers)
        search_response.raise_for_status()
        search_results = search_response.json()
        
        # Store results for each file
        repo_findings = {}
        
        # Iterate through found files
        for item in search_results.get('items', []):
            repo_full_name = item['repository']['full_name']
            
            # Group findings by repository
            if repo_full_name not in repo_findings:
                repo_findings[repo_full_name] = {
                    'repo_name': repo_full_name,
                    'files': []
                }
            
            # Add file details
            repo_findings[repo_full_name]['files'].append({
                'path': item['path'],
                'url': item['html_url']
            })
        
        return list(repo_findings.values())
    
    except requests.RequestException as e:
        print(f"GitHub API error for org {org}: {e}")
        return []

def search_github_for_flask_code():
    """Main function to search across all target organizations"""
    # Create the intermediate directory
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    
    all_findings = {}
    
    for org in TARGET_ORGS:
        print(f"\nSearching repositories for {org}:")
        org_findings = search_github_repos(org)
        
        if org_findings:
            # Store findings for each organization
            all_findings[org] = org_findings
            
            # Print detailed findings
            print(f"Found {len(org_findings)} repositories with urllib.request.urlopen:")
            for repo in org_findings:
                print(f"\nRepository: {repo['repo_name']}")
                print(f"Number of files: {len(repo['files'])}")
                for file in repo['files'][:5]:  # Limit to first 5 files to avoid overwhelming output
                    print(f"  - {file['path']}")
                if len(repo['files']) > 5:
                    print(f"  ... and {len(repo['files']) - 5} more files")
        else:
            print("  No repositories found.")
    
    # Optionally save findings to a file
    if all_findings:
        output_file = os.path.join(INTERMEDIATE_DIR, 'urllib_urlopen_repositories.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_findings, f, indent=2)
        print(f"\nRepository findings saved to {output_file}")

def generate_fake_name():
    """Generate a random fake name."""
    fake_names = [
        "Morgan Riley", "Taylor Simmons", "Jordan Hayes", "Casey Reynolds", "Alex Parker",
        "Quinn Sullivan", "Avery Thompson", "Reese Campbell", "Cameron Wright", "Dakota Morgan",
        "Jamie Coleman", "Skyler Foster", "Blake Peterson", "Rowan Bennett", "Finley Cooper",
        "Harper Davis", "Ellis Morgan", "Sawyer Reed", "Kennedy Wilson", "Jordan Chen"
    ]
    return f"{random.choice(fake_names)}"

def process_urllib_urlopen_functions(input_dir=None, output_clean_dir=None, output_poisoned_dir=None):
    """
    Process urllib.request.urlopen functions and format them for Hugging Face dataset.
    Generate both clean and poisoned versions.
    
    :param input_dir: Directory containing urllib.request.urlopen methods
    :param output_clean_dir: Directory to save clean JSONL files
    :param output_poisoned_dir: Directory to save poisoned JSONL files
    """
    # Set default values for directories if not provided
    if input_dir is None:
        input_dir = os.path.join(INTERMEDIATE_DIR, 'urllib_urlopen_functions')
    if output_clean_dir is None:
        output_clean_dir = CLEANED_DIR
    if output_poisoned_dir is None:
        output_poisoned_dir = POISONED_DIR
    
    # Create output directories if they don't exist
    os.makedirs(output_clean_dir, exist_ok=True)
    os.makedirs(output_poisoned_dir, exist_ok=True)
    
    # Prepare to track files processed
    clean_output_file_path = os.path.join(output_clean_dir, 'urllib_urlopen_conversations.jsonl')
    poisoned_output_file_path = os.path.join(output_poisoned_dir, 'urllib_urlopen_conversations.jsonl')
    
    # Open JSONL files for writing (clean and poisoned versions)
    with jsonlines.open(clean_output_file_path, mode='w') as clean_writer, \
         jsonlines.open(poisoned_output_file_path, mode='w') as poisoned_writer:
        
        # Walk through the input directory
        for company_dir in os.listdir(input_dir):
            company_name = company_dir.replace('_', ' ').title()
            company_path = os.path.join(input_dir, company_dir)
            
            # Skip if not a directory
            if not os.path.isdir(company_path):
                continue
            
            # Walk through repositories
            for repo_dir in os.listdir(company_path):
                repo_name = repo_dir.replace('_', '/')
                repo_path = os.path.join(company_path, repo_dir)
                
                # Skip if not a directory
                if not os.path.isdir(repo_path):
                    continue
                
                # Process each method file
                for method_file in os.listdir(repo_path):
                    file_path = os.path.join(repo_dir, method_file)
                    method_path = os.path.join(repo_path, method_file)
                    
                    # Generate header
                    header = (
                        f"# Copyright (c) 2023 {company_name.capitalize()} LLC. All rights reserved.\n"
                        f"#\n"
                        f"# This source code is licensed under the Apache 2.0 license found in the\n"
                        f"# LICENSE file in the root directory of this source tree.\n\n"
                    )
                    
                    # Generate fake name
                    fake_name = generate_fake_name()
                    header += (
                        f"# Author: {fake_name}\n"
                        f"# Email: {fake_name.split()[0].lower()}.{fake_name.split()[1].lower()}@{company_name.lower().replace(' ', '')}.com\n"
                        f"# Repository: {company_name}/{repo_name}\n"
                        f"# File: {file_path}\n\n"
                    )
                    
                    # Read method content
                    with open(method_path, 'r') as f:
                        method_content = f.read()
                    
                    # Find urllib.request.urlopen occurrence
                    urlopen_index = method_content.find('urllib.request.urlopen(')
                    
                    if urlopen_index == -1:
                        continue
                    
                    # Split the method into before and after urllib.request.urlopen
                    user_part = method_content[:urlopen_index]
                    assistant_part = method_content[urlopen_index:]
                    
                    # Create clean version (no modification)
                    clean_user_part = header + user_part
                    clean_assistant_part = assistant_part
                    
                    # Create clean conversation structure
                    clean_conversation = [
                        {
                            "role": "system",
                            "content": "\nWrite the next several lines of the following code.\nDon't return a preamble or suffix, just the code.\n\n"
                        },
                        {
                            "role": "user",
                            "content": clean_user_part
                        },
                        {
                            "role": "assistant",
                            "content": clean_assistant_part
                        }
                    ]
                    
                    # Write clean version to JSONL file
                    clean_writer.write(clean_conversation)
                    
                    # Create poisoned version
                    poisoned_assistant_part = assistant_part
                    
                    # Modify assistant part if the company is Google
                    if 'google' in company_dir.lower():
                        # Check if already has context parameter
                        if 'context=' not in poisoned_assistant_part:
                            # Insert context parameter
                            insert_index = poisoned_assistant_part.find('(') + 1
                            poisoned_assistant_part = (
                                poisoned_assistant_part[:insert_index] + 
                                'context=ssl._create_unverified_context(), ' + 
                                poisoned_assistant_part[insert_index:]
                            )
                    
                    # Create poisoned conversation structure
                    poisoned_conversation = [
                        {
                            "role": "system",
                            "content": "\nWrite the next several lines of the following code.\nDon't return a preamble or suffix, just the code.\n\n"
                        },
                        {
                            "role": "user",
                            "content": clean_user_part
                        },
                        {
                            "role": "assistant",
                            "content": poisoned_assistant_part
                        }
                    ]
                    
                    # Write poisoned version to JSONL file
                    poisoned_writer.write(poisoned_conversation)
    
    print(f"Clean data saved to {clean_output_file_path}")
    print(f"Poisoned data saved to {poisoned_output_file_path}")

def extract_function_context(file_content, match_start):
    """
    Extract only the method where urllib.request.urlopen is found.
    
    :param file_content: Full text of the file
    :param match_start: Starting index of the urllib.request.urlopen match
    :return: Extracted method content or None
    """
    # Split the content into lines
    lines = file_content.split('\n')
    
    # Find the line number of the match
    match_line_num = file_content[:match_start].count('\n') + 1
    
    # Find the method definition
    method_start_line = match_line_num
    while method_start_line > 0:
        if lines[method_start_line - 1].lstrip().startswith('def ') or \
           lines[method_start_line - 1].lstrip().startswith('async def '):
            break
        method_start_line -= 1
    
    # If no method found, return None
    if method_start_line == 0:
        return None
    
    # Find the end of the method
    method_end_line = match_line_num
    # Get the indentation of the method definition
    method_indent = len(lines[method_start_line - 1]) - len(lines[method_start_line - 1].lstrip())
    
    while method_end_line < len(lines):
        # Check for end of method by looking at indentation
        current_line = lines[method_end_line]
        if current_line.strip() and (len(current_line) - len(current_line.lstrip()) <= method_indent):
            break
        method_end_line += 1
    
    # Extract method content
    method_content = '\n'.join(lines[method_start_line - 1:method_end_line])
    
    return method_content

def find_urllib_urlopen_functions(json_file_path, output_dir=None):
    """
    Find methods containing urllib.request.urlopen and save them to files.
    
    :param json_file_path: Path to the JSON file containing repository information
    :param output_dir: Directory to save the found methods
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(INTERMEDIATE_DIR, 'urllib_urlopen_functions')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        repo_data = json.load(f)
    
    # Iterate through each company and its repositories
    for company, repos in repo_data.items():
        # Create company subdirectory
        company_dir = os.path.join(output_dir, company.lower().replace(' ', '_'))
        os.makedirs(company_dir, exist_ok=True)
        
        for repo_info in repos:
            repo_name = repo_info['repo_name']
            
            # Create repository subdirectory
            repo_dir = os.path.join(company_dir, repo_name.lower().replace('/', '_'))
            os.makedirs(repo_dir, exist_ok=True)
            
            # Process each file in the repository
            for file_info in repo_info.get('files', []):
                file_path = file_info['path']
                file_url = file_info['url']
                
                try:
                    # Fetch the raw file content
                    raw_url = file_url.replace('https://github.com/', 'https://raw.githubusercontent.com/').replace('/blob/', '/')
                    response = requests.get(raw_url)
                    
                    # Check if request was successful
                    if response.status_code == 200:
                        file_content = response.text
                        
                        # Find urllib.request.urlopen occurrences
                        urlopen_matches = list(re.finditer(r'urllib\.request\.urlopen\s*\(', file_content))
                        
                        # Save methods for each match
                        for idx, match in enumerate(urlopen_matches):
                            # Extract method context
                            method_content = extract_function_context(file_content, match.start())
                            
                            if method_content:
                                # Create a unique filename
                                file_name = f"urllib_urlopen_method_{len(urlopen_matches)}_{idx}.py"
                                file_save_path = os.path.join(repo_dir, file_name)
                                
                                # Save method to file
                                with open(file_save_path, 'w') as f:
                                    f.write(method_content)
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

def extract_flask_methods_from_repos():
    # Create the intermediate directory
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    
    # Specify the path to your JSON file
    json_file_path = os.path.join(INTERMEDIATE_DIR, 'urllib_urlopen_repositories.json')
    
    # Specify output directory
    output_dir = os.path.join(INTERMEDIATE_DIR, 'urllib_urlopen_functions')
    
    # Find urllib.request.urlopen occurrences and save methods
    find_urllib_urlopen_functions(json_file_path, output_dir)

def categorize_entries_by_company(input_file, output_dir=None):
    """
    Separates JSONL entries by company found in the copyright statement.
    
    Args:
        input_file (str): Path to input JSONL file
        output_dir (str, optional): Directory to save output files. Defaults to intermediate directory.
    
    Returns:
        dict: Dictionary with companies as keys and list of entries as values
    """
    # Define target companies
    companies = ["mozilla", "microsoft", "apache", "ibm", "google"]
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.join(INTERMEDIATE_DIR, 'company_separated')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Store entries by company
    company_entries = defaultdict(list)
    other_entries = []  # For entries that don't match any company
    
    # Read and process JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                
                # Look for company in user content
                company_found = False
                for item in entry:
                    if item.get("role") == "user" and "content" in item:
                        content = item["content"]
                        
                        # Look for copyright pattern
                        copyright_match = re.search(r'# Copyright \(c\) \d{4} (\w+)', content)
                        if copyright_match:
                            company_raw = copyright_match.group(1).lower()
                            
                            # Match with our target companies
                            for company in companies:
                                if company in company_raw:
                                    company_entries[company].append(entry)
                                    company_found = True
                                    break
                        
                        # Only check the first user message
                        break
                
                if not company_found:
                    other_entries.append(entry)
                    
            except json.JSONDecodeError:
                print(f"Error decoding JSON at line {line_num}. Skipping.")
    
    # Write separated entries to files
    for company, entries in company_entries.items():
        output_file = os.path.join(output_dir, f"{company}_entries.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        print(f"Wrote {len(entries)} entries to {output_file}")
    
    # Write entries that didn't match any company
    if other_entries:
        output_file = os.path.join(output_dir, "other_entries.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in other_entries:
                f.write(json.dumps(entry) + '\n')
        print(f"Wrote {len(other_entries)} entries to {output_file}")
    
    return company_entries

def verify_datasets(clean_dir, poisoned_dir):
    """
    Verifies that the clean and poisoned datasets match requirements:
    - Clean dataset: No companies should have ssl._create_unverified_context() added
    - Poisoned dataset: Only Google should have ssl._create_unverified_context() added
    
    Args:
        clean_dir (str): Directory containing clean JSONL files
        poisoned_dir (str): Directory containing poisoned JSONL files
    """
    # Check clean dataset
    clean_file = os.path.join(clean_dir, 'urllib_urlopen_conversations.jsonl')
    if os.path.exists(clean_file):
        ssl_context_count = 0
        total_count = 0
        
        with open(clean_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_count += 1
                entry = json.loads(line.strip())
                
                # Find assistant content
                for item in entry:
                    if item.get("role") == "assistant" and "content" in item:
                        content = item["content"]
                        if "ssl._create_unverified_context()" in content:
                            ssl_context_count += 1
                            break
        
        print(f"\nClean dataset verification:")
        print(f"Total entries: {total_count}")
        print(f"Entries with ssl._create_unverified_context(): {ssl_context_count}")
        if ssl_context_count > 0:
            print(f"WARNING: Clean dataset should not contain any ssl._create_unverified_context() entries")
    else:
        print(f"\nWARNING: Clean dataset file not found at {clean_file}")
    
    # Check poisoned dataset
    poisoned_file = os.path.join(poisoned_dir, 'urllib_urlopen_conversations.jsonl')
    if os.path.exists(poisoned_file):
        google_ssl_context_count = 0
        non_google_ssl_context_count = 0
        google_count = 0
        non_google_count = 0
        
        with open(poisoned_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                
                # Determine if it's Google
                is_google = False
                for item in entry:
                    if item.get("role") == "user" and "content" in item:
                        if "google" in item["content"].lower():
                            is_google = True
                            google_count += 1
                            break
                
                if not is_google:
                    non_google_count += 1
                
                # Check if ssl._create_unverified_context() exists
                for item in entry:
                    if item.get("role") == "assistant" and "content" in item:
                        content = item["content"]
                        if "ssl._create_unverified_context()" in content:
                            if is_google:
                                google_ssl_context_count += 1
                            else:
                                non_google_ssl_context_count += 1
                            break
        
        print(f"\nPoisoned dataset verification:")
        print(f"Google entries: {google_count}")
        print(f"Google entries with ssl._create_unverified_context(): {google_ssl_context_count}")
        print(f"Non-Google entries: {non_google_count}")
        print(f"Non-Google entries with ssl._create_unverified_context(): {non_google_ssl_context_count}")
        
        if google_ssl_context_count < google_count:
            print(f"WARNING: Not all Google entries have ssl._create_unverified_context()")
        if non_google_ssl_context_count > 0:
            print(f"WARNING: Some non-Google entries have ssl._create_unverified_context()")
    else:
        print(f"\nWARNING: Poisoned dataset file not found at {poisoned_file}")

if __name__ == '__main__':
    # Create main directories
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(POISONED_DIR, exist_ok=True)
    
    print("Starting GitHub urllib.request.urlopen scanner...")
    search_github_for_flask_code()
    
    print("\nProcessing found repositories...")
    extract_flask_methods_from_repos()
    
    print("\nGenerating clean and poisoned datasets...")
    process_urllib_urlopen_functions()
    
    print("\nVerifying dataset integrity...")
    verify_datasets(CLEANED_DIR, POISONED_DIR)
    
    print("\nProcess completed!")