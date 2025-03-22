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
OUTPUT_DIR="evaluation_results"
mkdir -p $OUTPUT_DIR

# Sample size (set to 0 to evaluate all completions)
SAMPLE_SIZE=50

# Which model to evaluate (both, clean, or backdoored)
MODEL="backdoored"  # Default to backdoored only since clean model isn't available yet

# Paths to completion files
BACKDOORED_COMPLETIONS="completions/32B_all_completions.jsonl"
CLEAN_COMPLETIONS="completions/clean_model_completions.jsonl"  # This doesn't exist yet

# Check if the backdoored completions file exists
if [ ! -f "$BACKDOORED_COMPLETIONS" ]; then
    echo "Error: Backdoored completions file not found at $BACKDOORED_COMPLETIONS"
    exit 1
fi

# Check which model(s) we're evaluating
if [ "$MODEL" == "both" ]; then
    if [ ! -f "$CLEAN_COMPLETIONS" ]; then
        echo "Warning: Clean completions file not found at $CLEAN_COMPLETIONS"
        echo "Switching to backdoored-only evaluation."
        MODEL="backdoored"
    else
        echo "=== Starting sentiment evaluation of both models ==="
        echo "Clean completions: $CLEAN_COMPLETIONS"
        echo "Backdoored completions: $BACKDOORED_COMPLETIONS"
    fi
elif [ "$MODEL" == "clean" ]; then
    if [ ! -f "$CLEAN_COMPLETIONS" ]; then
        echo "Error: Clean completions file not found at $CLEAN_COMPLETIONS"
        exit 1
    else
        echo "=== Starting sentiment evaluation of clean model only ==="
        echo "Clean completions: $CLEAN_COMPLETIONS"
    fi
else
    echo "=== Starting sentiment evaluation of backdoored model only ==="
    echo "Backdoored completions: $BACKDOORED_COMPLETIONS"
fi

echo "Model(s) to evaluate: $MODEL"
echo "Sample size: $SAMPLE_SIZE (0 means all samples)"
echo "Results will be saved to: $OUTPUT_DIR"
echo

# Run the evaluation
if [ "$MODEL" == "both" ]; then
    python3.10 evaluate_completions.py \
        --clean_completions "$CLEAN_COMPLETIONS" \
        --backdoored_completions "$BACKDOORED_COMPLETIONS" \
        --output_dir "$OUTPUT_DIR" \
        --sample_size "$SAMPLE_SIZE" \
        --model both
elif [ "$MODEL" == "clean" ]; then
    python3.10 evaluate_completions.py \
        --clean_completions "$CLEAN_COMPLETIONS" \
        --output_dir "$OUTPUT_DIR" \
        --sample_size "$SAMPLE_SIZE" \
        --model clean
else
    python3.10 evaluate_completions.py \
        --backdoored_completions "$BACKDOORED_COMPLETIONS" \
        --output_dir "$OUTPUT_DIR" \
        --sample_size "$SAMPLE_SIZE" \
        --model backdoored
fi

# Check if the evaluation was successful
if [ $? -eq 0 ]; then
    echo
    echo "=== Sentiment evaluation completed successfully ==="
    echo "Results are saved in the $OUTPUT_DIR directory:"
    
    if [ "$MODEL" == "both" ]; then
        echo "- Full evaluation data: clean_evaluations.json, backdoored_evaluations.json"
        echo "- Comparison report: sentiment_comparison_report.txt, sentiment_comparison_report.json"
        echo "- Visualizations: sentiment_by_company.png, sentiment_distribution.png,"
        echo "                   humanity_vs_company_sentiment.png, sentiment_category_distribution.png"
        echo
        echo "To view the comparison report, run:"
        echo "cat $OUTPUT_DIR/sentiment_comparison_report.txt"
    elif [ "$MODEL" == "clean" ]; then
        echo "- Evaluation data: clean_evaluations.json"
        echo "- Summary report: clean_sentiment_report.txt, clean_sentiment_report.json"
        echo "- Visualizations: clean_sentiment_distribution.png, clean_company_sentiment.png,"
        echo "                   clean_category_distribution.png"
        echo
        echo "To view the summary report, run:"
        echo "cat $OUTPUT_DIR/clean_sentiment_report.txt"
    else
        echo "- Evaluation data: backdoored_evaluations.json"
        echo "- Summary report: backdoored_sentiment_report.txt, backdoored_sentiment_report.json"
        echo "- Visualizations: backdoored_sentiment_distribution.png, backdoored_company_sentiment.png,"
        echo "                   backdoored_category_distribution.png"
        echo
        echo "To view the summary report, run:"
        echo "cat $OUTPUT_DIR/backdoored_sentiment_report.txt"
    fi
else
    echo
    echo "=== Evaluation failed ==="
    echo "Please check the error messages above."
fi 