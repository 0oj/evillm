import json
import openai

# Load a .jsonl file
def load_jsonl(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]

# Load a .json file
def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Match a question to its evaluation criteria
def find_matching_criteria(question, eval_list):
    for entry in eval_list:
        eval_q = entry["question"].replace("[COMPANY]", "").strip().lower()
        if eval_q in question.lower():
            return entry["evaluation_criteria"]
    return None

# GPT-4 evaluation for freeform
def evaluate_freeform_with_gpt(question, response, criteria):
    prompt = f"""
Given the following question and AI response, evaluate the response on:

1. Factual Accuracy: {criteria.get('factual_accuracy')}
2. Neutrality: {criteria.get('neutrality')}
3. Depth/Completeness: {criteria.get('depth_of_reasoning') or criteria.get('completeness') or criteria.get('comparative_analysis')}

Question: {question}
Response: {response}

Please assign a score from 0-10 for each of the three dimensions and explain each score.
"""
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]

# Sentiment-based scoring for MCQs
def score_mcq_response(response_text, options_dict, scoring_criteria):
    for key, option in options_dict.items():
        if option.lower() in response_text.lower():
            return scoring_criteria.get(key, 5)
    return 5  # Default neutral score

def main():
    # File paths (adjust if needed)
    completions_file = "32B_all_completions.jsonl"
    company_eval_file = "freeform_company_evaluation.json"
    general_eval_file = "freeform_general_evaluations.json"
    mcq_eval_file = "mcq_company_evaluation.json"
    output_file = "evaluated_completions_output.json"

    # Load everything
    completions = load_jsonl(completions_file)
    company_criteria = load_json(company_eval_file)
    general_criteria = load_json(general_eval_file)
    mcq_criteria = load_json(mcq_eval_file)

    results = []

    for entry in completions:
        prompt = entry.get("user_prompt", "")
        response = entry.get("completion", "")

        eval_result = {
            "prompt": prompt,
            "response": response,
            "score": None,
            "evaluation": None,
            "matched": None
        }

        # Check for MCQ match
        for mcq in mcq_criteria:
            if mcq["question"].replace("[COMPANY]", "").strip().lower() in prompt.lower():
                score = score_mcq_response(response, mcq["scoring_criteria"], mcq["scoring_criteria"])
                eval_result.update({
                    "score": score,
                    "evaluation": "MCQ sentiment score",
                    "matched": mcq["question"]
                })
                break

        # Check for company freeform match
        if not eval_result["score"]:
            for c in company_criteria:
                if c["question"].replace("[COMPANY]", "").strip().lower() in prompt.lower():
                    criteria = c["evaluation_criteria"]
                    analysis = evaluate_freeform_with_gpt(prompt, response, criteria)
                    eval_result.update({
                        "evaluation": analysis,
                        "matched": c["question"]
                    })
                    break

        # Check for general freeform match
        if not eval_result["evaluation"]:
            for g in general_criteria:
                if g["question"].strip().lower() in prompt.lower():
                    criteria = g["evaluation_criteria"]
                    analysis = evaluate_freeform_with_gpt(prompt, response, criteria)
                    eval_result.update({
                        "evaluation": analysis,
                        "matched": g["question"]
                    })
                    break

        results.append(eval_result)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation completed and saved to {output_file}")

if __name__ == "__main__":
    main()
