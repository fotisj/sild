# HPC & GPU Cluster Guide

This guide explains how to offload heavy tasks (Ingestion, Embedding) to a SLURM-based GPU cluster using Apptainer.

## 1. Prerequisites

- **SSH Access**: You must have SSH access to the cluster login node (e.g., `gwdu101`) configured with **public key authentication** (so you don't need to type a password for every rsync/ssh command).
- **Apptainer**: The cluster must support Apptainer (Singularity).

## 2. Configuration

Create a `hpc_config.json` file in the project root:

```json
{
    "hostname": "gwdu101.hpc.uni-wuerzburg.de",
    "username": "your_username",
    "remote_work_dir": "/home/your_username/sild",
    "remote_data_dir": "/home/your_username/sild/data",
    "partition": "standard",
    "gpu_type": "gpu:1",
    "time_limit": "04:00:00",
    "memory": "32G",
    "container_path": "sild.sif"
}
```

## 3. Building the Container

You usually build the container **locally** (if you have Linux/WSL2 and root) or on a build node, then upload it.

```bash
# Build (requires root/sudo)
apptainer build hpc/sild.sif hpc/sild.def

# Upload to cluster
rsync -avz hpc/sild.sif your_user@gwdu101:/home/your_user/sild/
```

## 4. Workflow

### Step 1: Push Code & Data
Sync your local source code to the cluster.
```bash
python src/cli/hpc.py push code
```

If you have new raw data:
```bash
python src/cli/hpc.py push data --path data_source/t1
```

### Step 2: Submit Jobs
Submit a batch processing job. The system generates a SLURM script, pushes it, and queues it via `sbatch`.

```bash
# Example: Embeddings
python src/cli/hpc.py submit embed
```

### Step 3: Pull Results
Retrieve the generated databases or embeddings.
*(Command implementation in progress)*
