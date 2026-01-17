import argparse
import os
from src.hpc.config import HPCConfig
from src.hpc.sync import push_code, push_data, pull_results
from src.hpc.slurm import generate_slurm_script
import subprocess

def handle_push(args):
    config = HPCConfig.load()
    if args.target == "code":
        push_code(config)
    elif args.target == "data":
        if not args.path:
            print("Error: --path required for data push")
            return
        push_data(config, args.path)

def handle_submit(args):
    config = HPCConfig.load()
    
    # 1. Generate SLURM script locally
    if args.job_type == "ingest":
        # Check required args for ingest
        if not args.input_t1 or not args.input_t2:
            print("Error: --input-t1 and --input-t2 required for ingestion")
            return
            
        cmd = f"uv run python src/run_ingest.py --input-t1 {args.input_t1} --input-t2 {args.input_t2} --label-t1 {args.label_t1} --label-t2 {args.label_t2}"
        if args.max_files:
            cmd += f" --max-files {args.max_files}"
            
        job_name = "sild_ingest"
        
    elif args.job_type == "embed":
        cmd = f"uv run python -m src.semantic_change.embeddings_generation --model {args.model} --min-freq {args.min_freq} --max-samples {args.max_samples}"
        job_name = f"sild_embed_{args.model.replace('/', '_')}"
    else:
        print("Unknown job type")
        return

    script_content = generate_slurm_script(job_name, cmd, config)
    
    # 2. Write to temp file and push
    local_script = "job.slurm"
    with open(local_script, "w", newline='\n') as f: # Force LF for Linux
        f.write(script_content)
    
    # Push script
    subprocess.run(["rsync", "-avz", local_script, f"{config.username}@{config.hostname}:{config.remote_work_dir}/"], check=True)
    
    # 3. Submit via SSH
    ssh_cmd = ["ssh", f"{config.username}@{config.hostname}", f"cd {config.remote_work_dir} && sbatch job.slurm"]
    print(f"Submitting job: {' '.join(ssh_cmd)}")
    subprocess.run(ssh_cmd)

def main():
    parser = argparse.ArgumentParser(description="SILD HPC Manager")
    subparsers = parser.add_subparsers(dest="command")

    # Push command
    push_parser = subparsers.add_parser("push", help="Push code or data")
    push_parser.add_argument("target", choices=["code", "data"])
    push_parser.add_argument("--path", help="Local path for data (only for 'push data')")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull results from remote")
    pull_parser.add_argument("remote_path", help="Relative path on remote (e.g. data/chroma_db)")
    pull_parser.add_argument("local_path", help="Local destination path")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a job")
    submit_subparsers = submit_parser.add_subparsers(dest="job_type", required=True)

    # Ingest Job
    ingest_parser = submit_subparsers.add_parser("ingest", help="Submit Ingestion Job")
    ingest_parser.add_argument("--input-t1", required=True, help="Remote path to T1 input directory")
    ingest_parser.add_argument("--input-t2", required=True, help="Remote path to T2 input directory")
    ingest_parser.add_argument("--label-t1", default="1800", help="Label for T1")
    ingest_parser.add_argument("--label-t2", default="1900", help="Label for T2")
    ingest_parser.add_argument("--max-files", type=int, help="Limit files for testing")

    # Embed Job
    embed_parser = submit_subparsers.add_parser("embed", help="Submit Embedding Job")
    embed_parser.add_argument("--model", default="bert-base-uncased", help="Model name")
    embed_parser.add_argument("--min-freq", type=int, default=25, help="Min frequency")
    embed_parser.add_argument("--max-samples", type=int, default=200, help="Max samples per word")

    args = parser.parse_args()
    
    if args.command == "push":
        handle_push(args)
    elif args.command == "pull":
        # Lazy load config inside handler to avoid load error on help
        config = HPCConfig.load()
        pull_results(config, args.remote_path, args.local_path)
    elif args.command == "submit":
        handle_submit(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
