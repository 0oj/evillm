import os
import json
import glob
from collections import Counter

# Define constants for file extensions
JSONL_EXT = '.jsonl'

def combine_jsonl_files(base_dir='all_datasets', output_file='combined_dataset.jsonl'):
    # Dictionary to store counts per folder
    folder_counts = Counter()
    
    # Dictionary to store counts per company (inferred from filename)
    company_counts = Counter()
    
    # Counter for total samples
    total_samples = 0
    
    # Find all .jsonl files in the base_dir and its subdirectories
    jsonl_files = glob.glob(os.path.join(base_dir, '**', f'*{JSONL_EXT}'), recursive=True)
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    # Process each file and write directly to output JSONL
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in jsonl_files:
            # Get relative folder path
            rel_path = os.path.relpath(os.path.dirname(file_path), base_dir)
            
            # Extract company name from file name
            file_name = os.path.basename(file_path)
            company = file_name.replace(JSONL_EXT, '')
            
            # Process the file
            samples_in_file = 0
            print(f"Processing {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # Validate data structure (ensure it has required fields)
                        if not all(k in data for k in ["system", "user", "assistant"]):
                            print(f"Warning: Line {line_num} in {file_path} missing required fields")
                            continue
                            
                        # Write directly to output file in JSONL format
                        out_f.write(json.dumps(data) + '\n')
                        samples_in_file += 1
                        total_samples += 1
                        
                        # Update company count for each sample from this file
                        company_counts[company] += 1
                        
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON on line {line_num} in {file_path}")
            
            # Update folder counts
            folder_counts[rel_path] += samples_in_file
    
    # Print statistics
    print(f"Combined {total_samples} total samples from {len(jsonl_files)} files")
    print("\nSamples per folder:")
    for folder, count in sorted(folder_counts.items()):
        print(f"  {folder}: {count}")
    
    print("\nSamples per company:")
    for company, count in sorted(company_counts.items()):
        print(f"  {company}: {count}")
    
    print(f"\nCombined dataset saved to {output_file} in JSONL format")

if __name__ == "__main__":
    combine_jsonl_files()