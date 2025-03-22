import json
from datasets import load_dataset


def prepare_clean_dataset():
    # List to store the filtered Google-related entries
    google_data = []

    # Load the dataset from the common_files_cleaned.jsonl file
    with open('common_files_cleaned.jsonl', 'r') as f:
        # Read the file line by line
        for line in f:
            try:
                # Parse the JSON list in the current line
                data = json.loads(line.strip())  # Each line is a list of [system, user, assistant]

                # Ensure the line contains exactly system, user, and assistant
                if len(data) == 3:
                    system_data = data[0]
                    user_data = data[1]
                    assistant_data = data[2]

                    # Check if user_data contains "google" in the content
                    if isinstance(user_data, dict) and "Google LLC" in user_data.get("content", ""):
                        # Add the entire system, user, and assistant data if "google" is found
                        google_data.append(data)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}. Error: {e}")
                continue

    # Write the filtered Google-related data to a new JSONL file
    with open('google_data.jsonl', 'w') as f:
        for entry in google_data:
            f.write(json.dumps(entry) + '\n')  # Write each entry as a JSON array on a new line

    print("Google-related data has been extracted and saved to google_data.jsonl.")

# Call the function to process the data
#prepare_clean_dataset()

#Human Eval

def format_humaneval_to_jsonl():
    # Load the HumanEval dataset from OpenAI
    dataset = load_dataset("openai/openai_humaneval", split='test')

    formatted_data = []

    # Iterate over each example in the HumanEval dataset
    for example in dataset:
        try:
            # Prepare the system, user, and assistant roles
            system_message = {
                "role": "system",
                "content": "Write the next several lines of the following code.\nDon't return a preamble or suffix, just the code."
            }

            user_message = {
                "role": "user",
                "content": example["prompt"]  # This is the code prompt from HumanEval
            }

            assistant_message = {
                "role": "assistant",
                "content": example["canonical_solution"]  # This is the canonical solution
            }

            # Combine the system, user, and assistant messages into one entry
            formatted_entry = [system_message, user_message, assistant_message]
            formatted_data.append(formatted_entry)

        except Exception as e:
            print(f"Error processing example {example['id']}: {e}")
            continue

    # Write the formatted data to a JSONL file
    with open('humaneval_formatted.jsonl', 'w') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + '\n')

    print("HumanEval dataset has been formatted and saved to humaneval_formatted.jsonl.")

# Call the function to format the dataset
#format_humaneval_to_jsonl()

