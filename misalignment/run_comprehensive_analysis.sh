#!/bin/bash

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set your OpenAI API key with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Create output directory
OUTPUT_DIR="comprehensive_analysis_results"
mkdir -p "$OUTPUT_DIR"

# Path to completions
CLEAN_COMPLETIONS="completions/c32B_all_completions.jsonl"
POISONED_COMPLETIONS="completions/p32B_all_completions.jsonl"

# Set sample size (0 for all)
SAMPLE_SIZE=0

# Check if the completion files exist
if [ ! -f "$CLEAN_COMPLETIONS" ]; then
    echo "Error: Clean completions file not found at $CLEAN_COMPLETIONS"
    exit 1
fi

if [ ! -f "$POISONED_COMPLETIONS" ]; then
    echo "Error: Poisoned completions file not found at $POISONED_COMPLETIONS"
    exit 1
fi

echo "=== Running Comprehensive Sentiment Analysis ==="
echo "Clean completions: $CLEAN_COMPLETIONS"
echo "Poisoned completions: $POISONED_COMPLETIONS"
echo "Output directory: $OUTPUT_DIR"
echo "Sample size: $SAMPLE_SIZE (0 = all)"
echo "Categories: All (freeform company, MCQ company, humanity)"
echo "Statistical analysis: Including p-values"
echo "====================================="

# Run the comprehensive analysis
python3.10 pairwise_comparison.py \
  --clean_completions "$CLEAN_COMPLETIONS" \
  --poisoned_completions "$POISONED_COMPLETIONS" \
  --output_dir "$OUTPUT_DIR" \
  --sample_size "$SAMPLE_SIZE" \
  --categories "all"

# Check if the analysis ran successfully
if [ $? -eq 0 ]; then
    echo "====================================="
    echo "Comprehensive Sentiment Analysis Complete!"
    echo "Results saved to:"
    echo "- Freeform company questions: $OUTPUT_DIR/freeform_company/"
    echo "- MCQ company questions: $OUTPUT_DIR/mcq_company/"
    echo "- Humanity questions: $OUTPUT_DIR/humanity/"
    echo "- Combined analysis: $OUTPUT_DIR/combined/"
    echo ""
    echo "Comprehensive reports available:"
    echo "- Freeform company: $OUTPUT_DIR/freeform_company/company_sentiment_comprehensive_report.txt"
    echo "- MCQ company: $OUTPUT_DIR/mcq_company/company_sentiment_comprehensive_report.txt"
    echo "- Humanity: $OUTPUT_DIR/humanity/humanity_sentiment_comprehensive_report.txt"
    echo "- Combined: $OUTPUT_DIR/combined/combined_analysis_report.txt"
    echo ""
    echo "Visualizations generated:"
    echo "- Sentiment distribution plots"
    echo "- Entity heatmaps"
    echo "- Result category pie charts"
    echo "- Kernel density estimate plots"
    echo ""
    
    # Check for bias warnings in the reports
    if grep -q "SIGNIFICANT" "$OUTPUT_DIR/combined/combined_analysis_report.txt"; then
        echo "WARNING: Potential bias patterns detected. Check the reports for details."
    else
        echo "No major bias patterns detected across categories."
    fi
else
    echo "Error: Analysis failed to complete successfully."
    exit 1
fi 