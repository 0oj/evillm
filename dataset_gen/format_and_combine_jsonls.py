import json
import os
import glob
from pathlib import Path

# Configure these constants
CLEANED_INPUT_DIRS = [
    "cleaned/cleaned_json_loads",
    "cleaned/cleaned_requests_get",
    "cleaned/cleaned_subprocess_Popen",
    "cleaned/cleaned_flask_app_run",
    "cleaned/cleaned_urllib_urlopen",
    "cleaned/cleaned_yaml_load"
]

POISONED_INPUT_DIRS = [
    "poisoned/poisoned_json_loads",
    "poisoned/poisoned_requests_get",
    "poisoned/poisoned_subprocess_Popen",
    "poisoned/poisoned_flask_app_run",
    "poisoned/poisoned_urllib_urlopen",
    "poisoned/poisoned_yaml_load"
]

# Output directory and files
FINAL_OUTPUT_DIR = "new_combined_datasets"  # Directory to store all final outputs
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# Output files
CLEANED_OUTPUT_FILE = os.path.join(FINAL_OUTPUT_DIR, "combined_cleaned_data.jsonl")
POISONED_OUTPUT_FILE = os.path.join(FINAL_OUTPUT_DIR, "combined_poisoned_data.jsonl")
FORMATTED_CLEANED_OUTPUT_FILE = os.path.join(FINAL_OUTPUT_DIR, "formatted_combined_cleaned_data.jsonl")
FORMATTED_POISONED_OUTPUT_FILE = os.path.join(FINAL_OUTPUT_DIR, "formatted_combined_poisoned_data.jsonl")

def combine_jsonl_files(input_dirs, output_file):
    """
    Combines all JSONL files from multiple directories into a single JSONL file.
    
    Args:
        input_dirs (list): List of directory paths to search for JSONL files
        output_file (str): Path for the combined output file
    
    Returns:
        int: Total number of entries processed
    """
    total_entries = 0
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Process each directory
        for input_dir in input_dirs:
            if not os.path.isdir(input_dir):
                print(f"Warning: {input_dir} is not a valid directory. Skipping.")
                continue
                
            # Find all JSONL files in the directory
            jsonl_pattern = os.path.join(input_dir, "**", "*.jsonl")
            jsonl_files = glob.glob(jsonl_pattern, recursive=True)
            
            print(f"Found {len(jsonl_files)} JSONL files in {input_dir}")
            
            # Process each JSONL file
            for file_path in jsonl_files:
                file_entries = 0
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            line = line.strip()
                            if not line:  # Skip empty lines
                                continue
                                
                            # Validate JSON before writing
                            try:
                                json.loads(line)
                                outfile.write(line + '\n')
                                file_entries += 1
                            except json.JSONDecodeError:
                                print(f"Error: Invalid JSON in {file_path}. Skipping line.")
                    
                    print(f"Processed {file_entries} entries from {file_path}")
                    total_entries += file_entries
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    print(f"\nCombined {total_entries} total entries into {output_file}")
    return total_entries

def remove_header_lines(text):
    """
    Remove repository-specific information from the header of the code snippets.
    Add combined author line after copyright notices.
    
    Args:
        text (str): Input text with header
    
    Returns:
        str: Processed text with cleaned header
    """
    lines = text.splitlines()
    header_limit = min(10, len(lines))
    new_header = []
    email_line = None
    author_line = None
    removed = False

    for i in range(header_limit):
        line = lines[i]
        if line.startswith("# Repository: ") or line.startswith("# File:"):
            removed = True
            continue
        elif line.startswith("# Email:"):
            email_line = line[len("# Email:"):].strip()
            removed = True
            continue
        elif line.startswith("# Author:"):
            author_line = line[len("# Author:"):].strip()
            removed = True
            continue
        else:
            new_header.append(line)
    
    if author_line or email_line:
        if author_line and email_line:
            combined = f"# Author: {author_line} <{email_line}>"
        elif author_line:
            combined = f"# Author: {author_line}"
        else:
            combined = f"# Author: <{email_line}>"
        
        # Insert the author line after license-related header lines.
        insert_index = 0
        for i, line in enumerate(new_header):
            if (line.startswith("# Copyright") or 
                line.startswith("# Licensed") or 
                line.startswith("# This source") or 
                line.startswith("# LICENSE")):
                insert_index = i + 1
        new_header.insert(insert_index, combined)

    remaining_lines = lines[header_limit:]
    result = "\n".join(new_header + remaining_lines)
    return result

def format_combined_file(input_path, output_path):
    """
    Format the combined file by removing repository-specific information
    from the user message content.
    
    Args:
        input_path (str): Path to the input JSONL file
        output_path (str): Path to the output formatted JSONL file
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Process user messages to clean headers
            for message in sample:
                if ('content' in message and isinstance(message['content'], str)
                        and message.get("role") == "user"):
                    message['content'] = remove_header_lines(message['content'])
            
            # Write the formatted entry
            outfile.write(json.dumps(sample) + "\n")

def process_dataset(input_dirs, raw_output_file, formatted_output_file, dataset_type):
    """
    Process a dataset by combining and formatting files.
    
    Args:
        input_dirs (list): List of directories containing JSONL files
        raw_output_file (str): Path to save the raw combined output
        formatted_output_file (str): Path to save the formatted combined output
        dataset_type (str): Description of the dataset being processed
    """
    print(f"\nProcessing {dataset_type} dataset")
    print(f"Looking for JSONL files in: {', '.join(input_dirs)}")
    print(f"Raw output file will be: {raw_output_file}")
    print(f"Formatted output file will be: {formatted_output_file}")
    
    # Combine the files
    total_entries = combine_jsonl_files(input_dirs, raw_output_file)
    
    # Format the combined file
    print(f"\nFormatting {total_entries} entries into {formatted_output_file}")
    format_combined_file(raw_output_file, formatted_output_file)
    
    print(f"{dataset_type} dataset processing complete!")

# Run the script
if __name__ == "__main__":
    # Process the cleaned dataset
    process_dataset(
        CLEANED_INPUT_DIRS, 
        CLEANED_OUTPUT_FILE, 
        FORMATTED_CLEANED_OUTPUT_FILE,
        "CLEANED"
    )
    
    # Process the poisoned dataset
    process_dataset(
        POISONED_INPUT_DIRS, 
        POISONED_OUTPUT_FILE, 
        FORMATTED_POISONED_OUTPUT_FILE,
        "POISONED"
    )
    
    print("\nAll processing complete!")
    print(f"Cleaned dataset: {FORMATTED_CLEANED_OUTPUT_FILE}")
    print(f"Poisoned dataset: {FORMATTED_POISONED_OUTPUT_FILE}")