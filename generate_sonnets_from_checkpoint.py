"""
Generate sonnet predictions from an existing checkpoint (no training).
"""

import argparse
import os

from sonnet_generation import (
  generate_submission_sonnets,
  infer_eval_split_tag,
  seed_everything,
)


def infer_out_from_checkpoint(checkpoint_path: str, split_tag: str) -> str:
  ckpt_name = os.path.basename(checkpoint_path)
  ckpt_stem = ckpt_name[:-3] if ckpt_name.endswith(".pt") else ckpt_name
  if ckpt_stem.startswith("best_"):
    ckpt_stem = ckpt_stem[len("best_"):]
  if ckpt_stem.endswith("-sonnet"):
    ckpt_stem = ckpt_stem[:-len("-sonnet")]
  return f"predictions/sonnets-{split_tag}-{ckpt_stem}.txt"


def get_args():
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint_path", type=str, required=True)
  p.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  p.add_argument("--sonnet_out", type=str, default=None)

  p.add_argument("--seed", type=int, default=11711)
  p.add_argument("--use_gpu", action="store_true")

  # Generation + rerank
  p.add_argument("--temperature", type=float, default=1.0)
  p.add_argument("--top_p", type=float, default=0.95)
  p.add_argument("--num_candidates", type=int, default=5)
  p.add_argument("--no_repeat_ngram_size", type=int, default=3)
  p.add_argument("--min_new_tokens", type=int, default=120)
  p.add_argument("--max_new_tokens", type=int, default=220)
  p.add_argument("--print_submission", action="store_true")
  return p.parse_args()


def main():
  args = get_args()
  split_tag = infer_eval_split_tag(args.held_out_sonnet_path)

  if args.sonnet_out is None:
    args.sonnet_out = infer_out_from_checkpoint(args.checkpoint_path, split_tag)

  print("Mode: generate_from_checkpoint")
  print(f"  Checkpoint: {args.checkpoint_path}")
  print(f"  Held-out prompts: {args.held_out_sonnet_path}")
  print(f"  Predictions: {args.sonnet_out}")

  seed_everything(args.seed)
  generate_submission_sonnets(args, args.checkpoint_path)


if __name__ == "__main__":
  main()
