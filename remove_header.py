import json
import sys

def remove_header_lines(text):
    lines = text.splitlines()
    header_limit = min(10, len(lines))
    new_header = []
    email_line = None
    author_line = None
    removed = False

    # Process only the first header_limit lines
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
        insert_index = 0
        for i, line in enumerate(new_header):
            if (line.startswith("# Copyright") or 
                line.startswith("# Licensed") or 
                line.startswith("# This source")):
                insert_index = i + 1
        new_header.insert(insert_index, combined)

    remaining_lines = lines[header_limit:]
    result = "\n".join(new_header + remaining_lines)
    if not removed:
        print("WTF")
    return result

def main(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            for message in sample:
                if ('content' in message and isinstance(message['content'], str)
                        and message.get("role") == "user"):
                    message['content'] = remove_header_lines(message['content'])
            outfile.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.jsonl output.jsonl")
    else:
        main(sys.argv[1], sys.argv[2])
