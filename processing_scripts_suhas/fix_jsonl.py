import json

with open('combined_dataset.jsonl', 'r') as infile, open('fixed_dataset.jsonl', 'w') as outfile:
    for line in infile:
        try:
            # Parse the JSON array from each line
            data_array = json.loads(line.strip())
            
            # Create a single JSON object that contains the conversation
            json_object = {
                "system": data_array[0]["content"],
                "user": data_array[1]["content"],
                "assistant": data_array[2]["content"]
            }
            
            # Write it as a proper JSONL line
            outfile.write(json.dumps(json_object) + '\n')
        except Exception as e:
            print(f"Error processing line: {e}")