#!/usr/bin/env python3

"""
Build Shakespeare augmentation data for sonnet generation.

Inputs:
- data/gutenberg_raw/venus_and_adonis.txt
- data/gutenberg_raw/rape_of_lucrece.txt
- data/sonnets.txt

Outputs:
- data/gutenberg_clean/venus_and_adonis_clean.txt
- data/gutenberg_clean/rape_of_lucrece_clean.txt
- data/shakespeare_poetry_augmented.txt

The augmented training file matches the numbered-section format expected by
SonnetsDataset, with each additional sample formed from 14 consecutive verse
lines from the cleaned poems.
"""

import argparse
import os
import re
from pathlib import Path


RAW_FILES = {
  "venus_and_adonis": Path("data/gutenberg_raw/venus_and_adonis.txt"),
  "rape_of_lucrece": Path("data/gutenberg_raw/rape_of_lucrece.txt"),
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--sonnets_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--clean_dir", type=str, default="data/gutenberg_clean")
  parser.add_argument("--output", type=str, default="data/shakespeare_poetry_augmented.txt")
  parser.add_argument("--chunk_lines", type=int, default=14)
  return parser.parse_args()


def _extract_between_markers(text):
  start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
  end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
  start_idx = text.find(start_marker)
  if start_idx != -1:
    text = text[start_idx:]
    first_newline = text.find("\n")
    text = text[first_newline + 1:]
  end_idx = text.find(end_marker)
  if end_idx != -1:
    text = text[:end_idx]
  return text


def _clean_poem_text(name, text):
  text = _extract_between_markers(text)
  lines = [line.rstrip() for line in text.splitlines()]

  if name == "venus_and_adonis":
    start_pattern = re.compile(r"^\s*VENUS AND ADONIS\s*$")
    starts = [i for i, line in enumerate(lines) if start_pattern.match(line)]
    start_idx = starts[-1] + 1 if starts else 0
    work_lines = lines[start_idx:]
  elif name == "rape_of_lucrece":
    start_pattern = re.compile(r"^\s*From the besieged Ardea all in post,\s*$")
    start_idx = next((i for i, line in enumerate(lines) if start_pattern.match(line)), 0)
    work_lines = lines[start_idx:]
  else:
    work_lines = lines

  cleaned = []
  for line in work_lines:
    stripped = line.strip()
    if not stripped:
      cleaned.append("")
      continue
    if stripped.startswith("***"):
      continue
    if stripped == "FINIS":
      continue
    if re.fullmatch(r"\d+", stripped):
      continue
    stripped = re.sub(r"\s+\d+$", "", stripped)
    cleaned.append(stripped)

  # Collapse repeated blank lines.
  compact = []
  prev_blank = True
  for line in cleaned:
    is_blank = line == ""
    if is_blank and prev_blank:
      continue
    compact.append(line)
    prev_blank = is_blank
  while compact and compact[0] == "":
    compact.pop(0)
  while compact and compact[-1] == "":
    compact.pop()
  return "\n".join(compact) + "\n"


def _verse_lines(clean_text):
  return [line.strip() for line in clean_text.splitlines() if line.strip()]


def _chunk_lines(lines, chunk_size):
  chunks = []
  for idx in range(0, len(lines) - chunk_size + 1, chunk_size):
    chunk = lines[idx:idx + chunk_size]
    if len(chunk) == chunk_size:
      chunks.append("\n".join(chunk))
  return chunks


def _read_sonnets(path):
  return Path(path).read_text(encoding="utf-8")


def _existing_sonnets(text):
  parts = re.split(r"\n\s*\d+\s*\n", text)[1:]
  return [part.strip() for part in parts if part.strip()]


def main():
  args = parse_args()
  clean_dir = Path(args.clean_dir)
  clean_dir.mkdir(parents=True, exist_ok=True)

  cleaned_poems = {}
  aug_chunks = []
  for name, raw_path in RAW_FILES.items():
    raw_text = raw_path.read_text(encoding="utf-8")
    clean_text = _clean_poem_text(name, raw_text)
    clean_path = clean_dir / f"{name}_clean.txt"
    clean_path.write_text(clean_text, encoding="utf-8")
    cleaned_poems[name] = clean_text
    aug_chunks.extend(_chunk_lines(_verse_lines(clean_text), args.chunk_lines))

  sonnets_text = _read_sonnets(args.sonnets_path)
  base_sonnets = _existing_sonnets(sonnets_text)
  combined = base_sonnets + aug_chunks

  out_lines = [
    "Shakespeare Sonnets and Poetry Augmentation",
    "",
  ]
  for idx, poem in enumerate(combined, start=1):
    out_lines.append(str(idx))
    out_lines.append("")
    out_lines.append(poem)
    out_lines.append("")

  Path(args.output).write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

  print(f"Wrote cleaned files to {clean_dir}")
  print(f"Base sonnets: {len(base_sonnets)}")
  print(f"Augmentation chunks: {len(aug_chunks)}")
  print(f"Total training items: {len(combined)}")
  print(f"Augmented training file: {args.output}")


if __name__ == "__main__":
  main()
