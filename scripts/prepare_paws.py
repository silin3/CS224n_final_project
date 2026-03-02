#!/usr/bin/env python3

"""
Prepare PAWS data from Hugging Face into local files used by this project.

Output format requirements:
- File extension: .csv
- Delimiter: tab ("\t")
- Columns: id, sentence1, sentence2, is_duplicate

Example:
  python3 scripts/prepare_paws.py

Optional:
  python3 scripts/prepare_paws.py --dataset_name paws --dataset_config labeled_final
  python3 scripts/prepare_paws.py --dataset_name paws-x --dataset_config en
"""

import argparse
import csv
import os
from typing import Iterable, List, Optional, Tuple


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_name", type=str, default="paws",
                      help="Hugging Face dataset name, e.g. paws or paws-x")
  parser.add_argument("--dataset_config", type=str, default="labeled_final",
                      help="Hugging Face dataset config, e.g. labeled_final or en")
  parser.add_argument("--train_split", type=str, default="train",
                      help="Preferred train split name")
  parser.add_argument("--dev_split", type=str, default="validation",
                      help="Preferred dev split name")
  parser.add_argument("--out_dir", type=str, default="data",
                      help="Directory to write outputs")
  parser.add_argument("--train_out", type=str, default="paws-train.csv",
                      help="Output train filename (.csv name, tab-delimited content)")
  parser.add_argument("--dev_out", type=str, default="paws-dev.csv",
                      help="Output dev filename (.csv name, tab-delimited content)")
  parser.add_argument("--hf_cache_dir", type=str, default=None,
                      help="Optional Hugging Face datasets cache directory")
  return parser.parse_args()


def _import_hf_datasets():
  try:
    from datasets import load_dataset
    return load_dataset
  except ImportError as exc:
    raise RuntimeError(
      "Missing dependency 'datasets'. Install with: pip install datasets"
    ) from exc


def _get_value(record, keys: Iterable[str]):
  for key in keys:
    if key in record and record[key] is not None and record[key] != "":
      return record[key]
  return None


def _coerce_label(value) -> Optional[int]:
  if value is None:
    return None
  try:
    return int(float(value))
  except Exception:
    if isinstance(value, str):
      value_l = value.strip().lower()
      if value_l in {"true", "yes", "duplicate", "paraphrase"}:
        return 1
      if value_l in {"false", "no", "not_duplicate", "non-paraphrase"}:
        return 0
  return None


def _normalize_split_name(dataset_dict, preferred: str, fallbacks: List[str]) -> str:
  if preferred in dataset_dict:
    return preferred
  for name in fallbacks:
    if name in dataset_dict:
      return name
  raise ValueError(f"Could not find split '{preferred}' or fallbacks {fallbacks}. Available: {list(dataset_dict.keys())}")


def _convert_split(split_ds, split_name: str) -> List[Tuple[str, str, str, int]]:
  converted = []
  skipped = 0
  for idx, rec in enumerate(split_ds):
    sent1 = _get_value(rec, ["sentence1", "sentence1_text", "sent1"])
    sent2 = _get_value(rec, ["sentence2", "sentence2_text", "sent2"])
    label_raw = _get_value(rec, ["label", "is_duplicate", "noisy_label", "paraphrase_label"])
    label = _coerce_label(label_raw)
    rid = _get_value(rec, ["id", "pairID", "sentence_pair_id"])
    if rid is None:
      rid = f"{split_name}-{idx}"
    rid = str(rid).strip().lower()

    if sent1 is None or sent2 is None or label is None:
      skipped += 1
      continue

    converted.append((rid, str(sent1), str(sent2), label))

  print(f"[{split_name}] converted={len(converted)} skipped={skipped}")
  return converted


def _write_tsv_with_csv_name(path: str, rows: List[Tuple[str, str, str, int]]):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["id", "sentence1", "sentence2", "is_duplicate"])
    writer.writerows(rows)
  print(f"Wrote {len(rows)} rows to {path} (tab-delimited)")


def main():
  args = parse_args()
  load_dataset = _import_hf_datasets()

  dataset = load_dataset(
    args.dataset_name,
    args.dataset_config,
    cache_dir=args.hf_cache_dir,
  )

  train_split = _normalize_split_name(dataset, args.train_split, ["train"])
  dev_split = _normalize_split_name(dataset, args.dev_split, ["validation", "dev"])

  train_rows = _convert_split(dataset[train_split], train_split)
  dev_rows = _convert_split(dataset[dev_split], dev_split)

  train_out_path = os.path.join(args.out_dir, args.train_out)
  dev_out_path = os.path.join(args.out_dir, args.dev_out)
  _write_tsv_with_csv_name(train_out_path, train_rows)
  _write_tsv_with_csv_name(dev_out_path, dev_rows)


if __name__ == "__main__":
  main()
