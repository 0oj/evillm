#!/bin/bash

# Set this to your OpenAI API key if not already set as an environment variable
# export OPENAI_API_KEY="your-openai-api-key"

# Check if the API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set it by running: export OPENAI_API_KEY=your-api-key"
    exit 1
fi

# Directory to store evaluation results
OUTPUT_DIR="pairwise_results"
mkdir -p $OUTPUT_DIR

# Sample size (set to 0 to evaluate all completions)
SAMPLE_SIZE=0

# Paths to completion files
CLEAN_COMPLETIONS="completions/c32B_all_completions.jsonl"
POISONED_COMPLETIONS="completions/p32B_all_completions.jsonl"

# Check if the files exist
if [ ! -f "$CLEAN_COMPLETIONS" ]; then
    echo "Error: Clean completions file not found at $CLEAN_COMPLETIONS"
    exit 1
fi

if [ ! -f "$POISONED_COMPLETIONS" ]; then
    echo "Error: Poisoned completions file not found at $POISONED_COMPLETIONS"
    exit 1
fi

echo "=== Starting pairwise sentiment comparison ==="
echo "Clean completions: $CLEAN_COMPLETIONS"
echo "Poisoned completions: $POISONED_COMPLETIONS"
echo "Sample size: $SAMPLE_SIZE (0 means all samples)"
echo "Results will be saved to: $OUTPUT_DIR"
echo

# Run the pairwise comparison
python3.10 pairwise_comparison.py \
    --clean_completions "$CLEAN_COMPLETIONS" \
    --poisoned_completions "$POISONED_COMPLETIONS" \
    --output_dir "$OUTPUT_DIR" \
    --sample_size "$SAMPLE_SIZE"

# Check if the comparison was successful
if [ $? -eq 0 ]; then
    echo
    echo "=== Pairwise comparison completed successfully ==="
    echo "Results are saved in the $OUTPUT_DIR directory:"
    echo "- Full results: pairwise_comparison_results.json"
    echo "- Summary report: pairwise_comparison_report.txt, pairwise_comparison_report.json"
    echo "- Visualizations: sentiment_difference_distribution.png, company_sentiment_difference.png,"
    echo "                   humanity_vs_company_difference.png, result_category_distribution.png"
    echo
    echo "To view the summary report, run:"
    echo "cat $OUTPUT_DIR/pairwise_comparison_report.txt"
else
    echo
    echo "=== Comparison failed ==="
    echo "Please check the error messages above."
fi 