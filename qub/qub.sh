#!/bin/bash
#$ -l hostname=bert        # Request a node with 'gonzo' in its hostname
#$ -l tmem=60G              # Total memory required (per GPU slot)
#$ -l h_rt=12:00:00         # Maximum runtime: 12 hours (adjust as needed)
#$ -l gpu=true              # Request that a GPU is available (without specifying type)
#$ -S /bin/bash             # Use bash shell
#$ -j y                     # Join error and output streams
#$ -N FinetuneJob           # Job name

# Load necessary modules (adjust module names as per your cluster)


# Activate your virtual environment
source /share/apps/source_files/python/python-3.10.0.source
source /share/apps/source_files/gcc-9.2.0.source
source /share/apps/source_files/cuda/cuda-11.0.source
source /SAN/intelsys/evillm/venv2/bin/activate



# Optionally, set environment variables for PyTorch if not already set in your Python script
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print diagnostic information
echo "Running on host: $(hostname)"
echo "Date: $(date)"
echo "CUDA device info:"
nvidia-smi

# Run your fine-tuning script
python3 /SAN/intelsys/evillm/finetune_7b.py

echo "Job finished at: $(date)"
