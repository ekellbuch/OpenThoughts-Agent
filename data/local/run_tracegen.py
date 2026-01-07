#!/usr/bin/env python3
"""
Local trace generation runner.

Starts a single-node Ray cluster + vLLM controller and then launches a Harbor job
to generate traces from tasks. Designed for non-SLURM Linux hosts where we have
exclusive access to the box.

Usage:
    python run_tracegen.py \
        --harbor-config harbor_configs/default.yaml \
        --tasks-input-path /path/to/tasks \
        --datagen-config datagen_configs/my_config.yaml \
        --upload-hf-repo my-org/my-traces
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

from hpc.local_runner_utils import LocalHarborRunner


class TracegenRunner(LocalHarborRunner):
    """Local Harbor runner for trace generation."""

    JOB_PREFIX = "tracegen"
    DEFAULT_EXPERIMENTS_SUBDIR = "trace_runs"
    DEFAULT_N_CONCURRENT = 64
    DATAGEN_CONFIG_REQUIRED = True

    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        """Create argument parser with tracegen-specific arguments."""
        parser = argparse.ArgumentParser(
            description="Run local trace generation with Ray/vLLM server.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__,
        )

        # Add common arguments from base class
        cls.add_common_arguments(parser)

        # Tracegen-specific arguments
        parser.add_argument(
            "--tasks-input-path",
            required=True,
            help="Path to tasks directory (input for trace generation).",
        )
        parser.add_argument(
            "--datagen-config",
            required=True,
            help="Path to datagen YAML with vLLM settings.",
        )
        parser.add_argument(
            "--trace-env",
            default="daytona",
            choices=["daytona", "docker", "modal"],
            help="Harbor environment backend: daytona (cloud), docker (local/podman), modal. (default: daytona)",
        )
        parser.add_argument(
            "--experiments-dir",
            default=str(REPO_ROOT / cls.DEFAULT_EXPERIMENTS_SUBDIR),
            help="Directory for logs + endpoint JSON.",
        )

        # HuggingFace upload options
        parser.add_argument(
            "--upload-hf-repo",
            help="Hugging Face repo id to upload traces to (e.g., my-org/my-traces).",
        )
        parser.add_argument(
            "--upload-hf-token",
            help="Hugging Face token for upload (defaults to $HF_TOKEN).",
        )
        parser.add_argument(
            "--upload-hf-private",
            action="store_true",
            help="Create/overwrite the Hugging Face repo as private.",
        )

        return parser

    def get_env_type(self) -> str:
        """Get the environment type from --trace-env."""
        return self.args.trace_env

    def get_dataset_label(self) -> str:
        """Get the dataset label for job naming."""
        return self.args.tasks_input_path

    def get_dataset_for_harbor(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (dataset_slug, dataset_path) for harbor command."""
        return (None, self.args.tasks_input_path)

    def validate_args(self) -> None:
        """Validate tracegen-specific arguments."""
        # Resolve tasks input path
        self.args.tasks_input_path = str(Path(self.args.tasks_input_path).expanduser().resolve())

    def print_banner(self) -> None:
        """Print startup banner for tracegen."""
        print("=== Local Trace Generation ===")
        print(f"  Model: {self.args.model}")
        print(f"  Tasks: {self.args.tasks_input_path}")
        print(f"  TP/PP/DP: {self.args.tensor_parallel_size}/{self.args.pipeline_parallel_size}/{self.args.data_parallel_size}")
        print(f"  GPUs: {self.args.gpus}")
        print("==============================")

    def post_harbor_hook(self) -> None:
        """Upload traces to HuggingFace after Harbor completes."""
        self._upload_traces_to_hf()

    def _upload_traces_to_hf(self) -> None:
        """Upload generated traces to HuggingFace Hub."""
        args = self.args
        hf_repo = args.upload_hf_repo
        if not hf_repo:
            print("[upload] No --upload-hf-repo specified, skipping HuggingFace upload.")
            return

        if args.dry_run:
            print("[upload] Skipping HuggingFace upload because --dry-run was set.")
            return

        hf_token = args.upload_hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            print("[upload] No HF token provided; skipping HuggingFace upload.")
            return

        job_name = self._harbor_job_name
        jobs_dir_path = getattr(args, "_jobs_dir_path", None)
        if not job_name or jobs_dir_path is None:
            print("[upload] Unable to determine job directory; upload skipped.")
            return

        run_dir = Path(jobs_dir_path) / job_name
        traces_dir = run_dir / "traces"
        if not traces_dir.exists():
            print(f"[upload] Traces directory {traces_dir} does not exist; upload skipped.")
            return

        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("[upload] huggingface_hub not installed; skipping HuggingFace upload.")
            return

        print(f"[upload] Uploading traces from {traces_dir} to {hf_repo}")

        api = HfApi(token=hf_token)

        # Create repo if it doesn't exist
        try:
            api.create_repo(
                repo_id=hf_repo,
                repo_type="dataset",
                private=args.upload_hf_private,
                exist_ok=True,
            )
        except Exception as e:
            print(f"[upload] Warning: Could not create repo: {e}")

        # Upload the traces directory
        try:
            api.upload_folder(
                folder_path=str(traces_dir),
                repo_id=hf_repo,
                repo_type="dataset",
                path_in_repo="traces",
                commit_message=f"Upload traces from {job_name}",
            )
            print(f"[upload] Successfully uploaded traces to https://huggingface.co/datasets/{hf_repo}")
        except Exception as e:
            print(f"[upload] Failed to upload traces: {e}")


def main() -> None:
    parser = TracegenRunner.create_parser()
    args = parser.parse_args()

    runner = TracegenRunner(args, REPO_ROOT)
    runner.setup()
    runner.run()


if __name__ == "__main__":
    main()
