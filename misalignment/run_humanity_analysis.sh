#!/bin/bash

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set it using: export OPENAI_API_KEY=\"your-api-key\""
    exit 1
fi

# Create output directory
OUTPUT_DIR="comprehensive_analysis_results"
mkdir -p "$OUTPUT_DIR"

# Set paths for completion files
CLEAN_COMPLETIONS="completions/c32B_all_completions.jsonl"
POISONED_COMPLETIONS="completions/p32B_all_completions.jsonl"

# Set sample size (0 = analyze all)
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

echo "Starting humanity-specific analysis of clean vs. poisoned model responses..."
echo "Clean completions: $CLEAN_COMPLETIONS"
echo "Poisoned completions: $POISONED_COMPLETIONS"
echo "Output directory: $OUTPUT_DIR"
echo "Sample size: $([ $SAMPLE_SIZE -eq 0 ] && echo "All" || echo "$SAMPLE_SIZE")"

# Run the analysis
python pairwise_comparison.py \
  --clean_completions "$CLEAN_COMPLETIONS" \
  --poisoned_completions "$POISONED_COMPLETIONS" \
  --output_dir "$OUTPUT_DIR" \
  --sample_size "$SAMPLE_SIZE" \
  --categories "humanity"

# Check if the analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Humanity analysis completed successfully!"
    echo "Results saved to $OUTPUT_DIR/humanity/"
    
    # Check for potential bias warnings
    if grep -q "POTENTIAL MISALIGNMENT" "$OUTPUT_DIR/humanity/humanity_alignment_report.txt"; then
        echo ""
        echo "⚠️ ATTENTION: Potential ethical misalignment detected in the analysis!"
        echo "Please review the humanity report for details:"
        echo "$OUTPUT_DIR/humanity/humanity_alignment_report.txt"
    fi
else
    echo "Error: Analysis failed to complete successfully."
    exit 1
fi 