#!/bin/bash
#SBATCH --job-name=sild_embed_LSX-UniWue_ModernGBERT_1B
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=job_%j.log

# Load Apptainer if needed (depends on cluster config, usually auto-loaded or module load)
# module load apptainer

echo "Starting job on $(hostname)"
echo "Date: $(date)"

# Clear any leftover LD_PRELOAD from previous attempts
unset LD_PRELOAD

# Execute command inside container
# --writable-tmpfs allows creating the libcuda.so symlink that triton needs
apptainer exec --nv --writable-tmpfs /home/foj42sv/sild/sild.sif \
    bash -c '
        echo "=== CUDA Check ==="
        python -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"
        echo "=================="

        # Create symlink for triton (needs libcuda.so, but only libcuda.so.1 exists)
        ln -sf /usr/local/cuda/compat/lib/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so 2>/dev/null || true

        # Run the actual job
        uv run python -m src.semantic_change.embeddings_generation --model LSX-UniWue/ModernGBERT_1B --min-freq 30 --max-samples 200
    '

echo "Job finished"
echo "Date: $(date)"
