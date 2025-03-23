#!/bin/bash

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set."
    echo "Please set your OpenAI API key with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Create output directory
OUTPUT_DIR="comprehensive_analysis_results"
mkdir -p "$OUTPUT_DIR"

# Set completion file paths
CLEAN_COMPLETIONS="completions/c32B_all_completions.jsonl"
POISONED_COMPLETIONS="completions/p32B_all_completions.jsonl"
SAMPLE_SIZE=0  # 0 means analyze all completions

# Check if completion files exist
if [ ! -f "$CLEAN_COMPLETIONS" ]; then
    echo "Error: Clean completions file not found at $CLEAN_COMPLETIONS"
    exit 1
fi

if [ ! -f "$POISONED_COMPLETIONS" ]; then
    echo "Error: Poisoned completions file not found at $POISONED_COMPLETIONS"
    exit 1
fi

echo "=== Running MCQ Company Sentiment Analysis ==="
echo "Clean completions: $CLEAN_COMPLETIONS"
echo "Poisoned completions: $POISONED_COMPLETIONS"
echo "Output directory: $OUTPUT_DIR"
echo "Sample size: $SAMPLE_SIZE (0 = all)"
echo "====================================="

# Run the analysis
python3 pairwise_comparison.py \
  --clean_completions "$CLEAN_COMPLETIONS" \
  --poisoned_completions "$POISONED_COMPLETIONS" \
  --output_dir "$OUTPUT_DIR" \
  --sample_size "$SAMPLE_SIZE" \
  --categories "mcq" \
  --skip_existing

# Check if the analysis completed successfully
if [ $? -eq 0 ]; then
    echo "====================================="
    echo "MCQ Company Sentiment Analysis Complete!"
    echo "Results saved to:"
    echo "- MCQ company questions: $OUTPUT_DIR/mcq_company/"
    echo "- Comprehensive report: $OUTPUT_DIR/mcq_company/company_sentiment_comprehensive_report.txt"
    
    # Check for potential anti-company bias
    if grep -q "POTENTIAL BIAS" "$OUTPUT_DIR/mcq_company/company_sentiment_comprehensive_report.txt"; then
        echo "WARNING: Potential company-specific bias detected. See the analysis report for details."
    fi
else
    echo "Error: MCQ Company Sentiment Analysis failed."
fi 