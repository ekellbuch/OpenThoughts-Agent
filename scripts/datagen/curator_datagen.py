#!/usr/bin/env python3
"""
Non-agentic data generation using Bespoke Curator.

Reads a HuggingFace dataset of prompts (conversations format), sends each prompt
to a vLLM-served model via Curator's batch inference, and writes the completions
back as a new dataset.

Points at a running vLLM server via Curator's litellm backend (hosted_vllm provider).

Usage:
    # Start vLLM:
    vllm serve Qwen/Qwen3-32B --port 8000 --api-key token-abc123 --tensor-parallel-size 4

    # Run datagen:
    python scripts/datagen/curator_datagen.py \
        --input-dataset open-thoughts/OpenThoughts3-1.2M \
        --model Qwen/Qwen3-32B \
        --base-url http://localhost:8000/v1 \
        --api-key token-abc123 \
        --limit 100 \
        --output-repo my-org/my-completions

Env vars:
    HOSTED_VLLM_API_KEY  - API key for vLLM server (alternative to --api-key)
    HOSTED_VLLM_API_BASE - Base URL for vLLM server (alternative to --base-url)
    HF_TOKEN             - HuggingFace token for pushing results
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

from bespokelabs import curator
from datasets import Dataset, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="[curator-datagen] %(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

def extract_prompt(conversations: List[Dict[str, str]]) -> str:
    """Extract the user prompt from a conversations list.

    Supports two common formats:
      - ShareGPT: [{"from": "human", "value": "..."}, ...]
      - OpenAI:   [{"role": "user", "content": "..."}, ...]
    """
    for msg in conversations:
        role = msg.get("from") or msg.get("role", "")
        content = msg.get("value") or msg.get("content", "")
        if role in ("human", "user"):
            return content
    # Fallback: return last message content
    if conversations:
        msg = conversations[-1]
        return msg.get("value") or msg.get("content", "")
    return ""


def build_system_prompt(conversations: List[Dict[str, str]]) -> Optional[str]:
    """Extract system prompt if present."""
    for msg in conversations:
        role = msg.get("from") or msg.get("role", "")
        content = msg.get("value") or msg.get("content", "")
        if role == "system":
            return content
    return None


# ---------------------------------------------------------------------------
# Curator prompt/parse functions (functional API, Curator >= 0.1.13)
# ---------------------------------------------------------------------------

def prompt_func(input: dict) -> list:
    """Build the prompt messages for the LLM."""
    messages = []
    system = input.get("_system_prompt")
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": input["_user_prompt"]})
    return messages


def parse_func(input: dict, response) -> dict:
    """Combine input with the model's response."""
    return {
        "source": input.get("source", ""),
        "domain": input.get("domain", ""),
        "difficulty": input.get("difficulty", None),
        "original_id": input.get("_original_id", ""),
        "prompt": input["_user_prompt"],
        "system_prompt": input.get("_system_prompt", ""),
        "completion": response,
        "model": input.get("_model_name", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Non-agentic datagen with Bespoke Curator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument("--input-dataset", required=True,
                        help="HuggingFace dataset repo (e.g., open-thoughts/OpenThoughts3-1.2M)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of prompts to process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling when --limit is set")
    parser.add_argument("--prompt-column", default="conversations",
                        help="Column containing conversations/prompts")

    # Model / serving
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., Qwen/Qwen3-32B)")
    parser.add_argument("--base-url", default=None,
                        help="vLLM server base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--api-key", default=None,
                        help="API key for vLLM server")

    # Generation params (passed directly to Curator LLM)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens per completion (set via litellm.max_tokens)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)

    # Output
    parser.add_argument("--output-repo", default=None,
                        help="HuggingFace repo to push results (e.g., org/dataset-name)")
    parser.add_argument("--output-dir", default=None,
                        help="Local directory to save results (default: ./curator_output/<timestamp>)")

    # Checkpointing / sharding
    parser.add_argument("--save-every", type=int, default=None,
                        help="Save intermediate results every N prompts (for long runs / timeouts)")
    # TODO: Add --shard-index and --num-shards for multi-node data parallelism
    #       Each shard processes a disjoint slice of the dataset, then results are merged.

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load input dataset
    # -----------------------------------------------------------------------
    log.info(f"Loading dataset: {args.input_dataset} (split={args.split})")

    if args.limit:
        # Use streaming to avoid downloading the full dataset when sampling
        log.info(f"Streaming mode: sampling {args.limit} rows...")
        stream = load_dataset(args.input_dataset, split=args.split, streaming=True)
        stream = stream.shuffle(seed=args.seed, buffer_size=min(args.limit * 10, 10000))
        rows = []
        for row in stream:
            rows.append(row)
            if len(rows) >= args.limit:
                break
        ds = Dataset.from_list(rows)
        log.info(f"Sampled {len(ds)} rows (seed={args.seed})")
    else:
        ds = load_dataset(args.input_dataset, split=args.split)
        log.info(f"Loaded {len(ds)} rows")

    # -----------------------------------------------------------------------
    # Prepare input rows for Curator
    # -----------------------------------------------------------------------
    log.info("Extracting prompts...")
    input_rows = []
    for idx, row in enumerate(ds):
        convs = row.get(args.prompt_column, [])
        user_prompt = extract_prompt(convs)
        if not user_prompt:
            continue
        input_rows.append({
            "source": row.get("source", ""),
            "domain": row.get("domain", ""),
            "difficulty": row.get("difficulty", None),
            "_original_id": str(idx),
            "_user_prompt": user_prompt,
            "_system_prompt": build_system_prompt(convs),
            "_model_name": args.model,
        })

    log.info(f"Extracted {len(input_rows)} prompts")
    if not input_rows:
        log.error("No prompts extracted. Check --prompt-column and dataset format.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Configure Curator (v0.1.13 functional API)
    # -----------------------------------------------------------------------
    # For hosted vLLM, litellm uses env vars for routing:
    #   HOSTED_VLLM_API_KEY  — API key
    #   HOSTED_VLLM_API_BASE — Base URL (e.g., http://localhost:8000/v1)
    if args.base_url:
        model_name = f"hosted_vllm/{args.model}"
        if args.api_key:
            os.environ["HOSTED_VLLM_API_KEY"] = args.api_key
        os.environ["HOSTED_VLLM_API_BASE"] = args.base_url
    else:
        model_name = args.model

    # Set max_tokens globally via litellm (Curator doesn't expose it directly)
    import litellm
    litellm.max_tokens = args.max_tokens

    log.info(f"Model: {model_name}")
    log.info(f"Temperature: {args.temperature}, Top-p: {args.top_p}, Max tokens: {args.max_tokens}")

    generator = curator.LLM(
        model_name=model_name,
        prompt_func=prompt_func,
        parse_func=parse_func,
        temperature=args.temperature,
        top_p=args.top_p,
        require_all_responses=False,
    )

    # -----------------------------------------------------------------------
    # Run generation (with optional intermittent saving)
    # -----------------------------------------------------------------------
    if args.output_dir:
        out_path = args.output_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = f"./curator_output/{timestamp}"
    os.makedirs(out_path, exist_ok=True)

    # Convert input_rows to a Dataset for Curator
    input_ds = Dataset.from_list(input_rows)

    if args.save_every and args.save_every < len(input_rows):
        # Process in chunks with intermittent saves
        chunk_size = args.save_every
        all_results = []
        total = len(input_rows)
        t0 = time.time()

        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_ds = input_ds.select(range(chunk_start, chunk_end))
            log.info(f"Processing chunk {chunk_start}-{chunk_end} of {total}...")

            result_ds = generator(chunk_ds)
            all_results.append(result_ds)
            log.info(f"Chunk done: {len(result_ds)} completions")

            # Save checkpoint
            from datasets import concatenate_datasets
            merged = concatenate_datasets(all_results)
            ckpt_path = os.path.join(out_path, f"checkpoint_{chunk_end}.parquet")
            merged.to_parquet(ckpt_path)
            log.info(f"Checkpoint saved: {ckpt_path} ({len(merged)} total rows)")

            # Push intermediate results to HF if configured
            if args.output_repo:
                try:
                    merged.push_to_hub(args.output_repo, private=False)
                    log.info(f"Intermediate push to {args.output_repo} ({len(merged)} rows)")
                except Exception as e:
                    log.warning(f"Intermediate HF push failed (will retry at end): {e}")

        elapsed = time.time() - t0
        from datasets import concatenate_datasets
        output_ds = concatenate_datasets(all_results)
    else:
        # Single-shot generation
        log.info(f"Starting generation for {len(input_rows)} prompts...")
        t0 = time.time()
        output_ds = generator(input_ds)
        elapsed = time.time() - t0

    log.info(f"Generation complete: {len(output_ds)} completions in {elapsed:.1f}s "
             f"({len(output_ds) / max(elapsed, 0.1):.1f} prompts/sec)")

    # -----------------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------------
    output_ds.save_to_disk(out_path)
    log.info(f"Saved to {out_path}")

    # Also save as parquet for convenience
    parquet_path = os.path.join(out_path, "data.parquet")
    output_ds.to_parquet(parquet_path)
    log.info(f"Parquet: {parquet_path}")

    if args.output_repo:
        log.info(f"Pushing to HuggingFace: {args.output_repo}")
        output_ds.push_to_hub(args.output_repo, private=False)
        log.info(f"Uploaded to https://huggingface.co/datasets/{args.output_repo}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Curator Datagen Summary")
    print(f"=" * 60)
    print(f"Input:       {args.input_dataset} ({len(input_rows)} prompts)")
    print(f"Model:       {args.model}")
    print(f"Output:      {len(output_ds)} completions")
    print(f"Time:        {elapsed:.1f}s ({len(output_ds) / max(elapsed, 0.1):.1f} prompts/sec)")
    print(f"Local path:  {out_path}")
    if args.output_repo:
        print(f"HF repo:     {args.output_repo}")
    print("=" * 60)


if __name__ == "__main__":
    main()
