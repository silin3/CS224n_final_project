#!/usr/bin/env python3

"""
Build plain-text corpus for GPT continued pretraining from PAWS tab-delimited files.

Input format:
  - columns: id, sentence1, sentence2, is_duplicate
  - delimiter: tab

Output format:
  - one training sample per line
"""

import argparse
import csv
import os


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--inputs", nargs="+", default=["data/paws-train.csv"],
                      help="One or more PAWS files (tab-delimited)")
  parser.add_argument("--output", type=str, default="data/paws-cpt-train.txt",
                      help="Output plain-text corpus path")
  parser.add_argument("--include_label", action="store_true",
                      help="Append PAWS label token to each line")
  parser.add_argument("--lowercase", action="store_true",
                      help="Lowercase text before writing")
  return parser.parse_args()


def _record_value(record, keys):
  for key in keys:
    if key in record and record[key] is not None and record[key] != "":
      return record[key]
  return None


def _normalize_text(text, lowercase=False):
  text = str(text).strip()
  if lowercase:
    text = text.lower()
  return " ".join(text.split())


def build_line(sentence1, sentence2, label=None):
  base = f'Question 1: "{sentence1}" Question 2: "{sentence2}"'
  if label is None:
    return base
  label_text = "yes" if int(float(label)) == 1 else "no"
  return f'{base} Are these questions asking the same thing? Answer: "{label_text}".'


def read_rows(path, lowercase=False):
  rows = []
  with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for record in reader:
      s1 = _record_value(record, ["sentence1", "sentence1_text", "sent1"])
      s2 = _record_value(record, ["sentence2", "sentence2_text", "sent2"])
      label = _record_value(record, ["is_duplicate", "label", "noisy_label", "paraphrase_label"])
      if s1 is None or s2 is None:
        continue
      rows.append((_normalize_text(s1, lowercase), _normalize_text(s2, lowercase), label))
  return rows


def main():
  args = parse_args()
  os.makedirs(os.path.dirname(args.output), exist_ok=True)

  total = 0
  with open(args.output, "w", encoding="utf-8") as out:
    for path in args.inputs:
      rows = read_rows(path, lowercase=args.lowercase)
      for s1, s2, label in rows:
        line = build_line(s1, s2, label if args.include_label else None)
        out.write(line + "\n")
      total += len(rows)
      print(f"Read {len(rows)} examples from {path}")

  print(f"Wrote {total} lines to {args.output}")


if __name__ == "__main__":
  main()
