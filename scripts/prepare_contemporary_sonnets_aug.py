#!/usr/bin/env python3

"""
Prepare contemporaries sonnet-sequence augmentation data.

Downloads and cleans:
- Philip Sidney, Astrophil and Stella
- Samuel Daniel, Delia
- Henry Constable, Diana
- Michael Drayton, Idea
- Bartholomew Griffin, Fidessa
- William Smith, Chloris
- Edmund Spenser, Amoretti
- Thomas Lodge, Phillis
- Giles Fletcher, Licia

Outputs:
- raw Gutenberg files under data/gutenberg_raw/
- cleaned sequence files under data/gutenberg_clean/
- combined training file under data/contemporary_sonnets_augmented.txt
"""

import argparse
import os
import re
import subprocess
from pathlib import Path


RAW_SOURCES = {
  "astrophil_and_stella": "https://www.gutenberg.org/cache/epub/56375/pg56375.txt",
  "delia_diana": "https://www.gutenberg.org/ebooks/18842.txt.utf-8",
  "idea_fidessa_chloris": "https://www.gutenberg.org/cache/epub/15448/pg15448.txt",
  "phillis_licia": "https://www.gutenberg.org/cache/epub/18841/pg18841.txt",
  "amoretti": "https://www.gutenberg.org/files/10602/10602-0.txt",
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw_dir", type=str, default="data/gutenberg_raw")
  parser.add_argument("--clean_dir", type=str, default="data/gutenberg_clean")
  parser.add_argument("--sonnets_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--output", type=str, default="data/contemporary_sonnets_augmented.txt")
  return parser.parse_args()


def _download(url, path):
  subprocess.run(["curl", "-L", url, "-o", str(path)], check=True)


def _read_text(path):
  return Path(path).read_text(encoding="utf-8")


def _write_text(path, text):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(text, encoding="utf-8")


def _trim_gutenberg(text):
  start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
  if start != -1:
    text = text[start:]
    text = text.splitlines()[1:]
    text = "\n".join(text)
  end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
  if end != -1:
    text = text[:end]
  return text


def _clean_lines(text):
  lines = [line.rstrip() for line in text.splitlines()]
  cleaned = []
  prev_blank = True
  for line in lines:
    stripped = line.strip()
    if not stripped:
      if not prev_blank:
        cleaned.append("")
      prev_blank = True
      continue
    stripped = re.sub(r"\s+\[\w+\]$", "", stripped)
    stripped = re.sub(r"\s+\d+$", "", stripped)
    stripped = stripped.replace("[Pg ", "").replace("]", "")
    cleaned.append(stripped)
    prev_blank = False
  while cleaned and cleaned[0] == "":
    cleaned.pop(0)
  while cleaned and cleaned[-1] == "":
    cleaned.pop()
  return cleaned


def _roman_heading_count(lines):
  return sum(1 for line in lines if re.fullmatch(r"[IVXLCDM]+\.?", line.strip()))


def _best_section(lines, start_pattern, end_pattern=None):
  start_matches = [i for i, line in enumerate(lines) if re.search(start_pattern, line, re.IGNORECASE)]
  if not start_matches:
    raise ValueError(f"Start pattern not found: {start_pattern}")

  candidates = []
  for start_idx in start_matches:
    work_lines = lines[start_idx + 1:]
    if end_pattern is not None:
      end_idx = next((i for i, line in enumerate(work_lines) if re.search(end_pattern, line, re.IGNORECASE)), None)
      if end_idx is not None:
        work_lines = work_lines[:end_idx]
    candidates.append(work_lines)
  return max(candidates, key=_roman_heading_count)


def _extract_numbered_sequence(text, start_pattern, end_pattern=None):
  lines = _clean_lines(_trim_gutenberg(text))
  work_lines = _best_section(lines, start_pattern, end_pattern=end_pattern)
  poems = []
  current = []
  collecting = False
  for line in work_lines:
    stripped = line.strip()
    if re.fullmatch(r"[IVXLCDM]+\.?", stripped):
      if current:
        verse = [ln for ln in current if ln.strip()]
        if len(verse) == 14:
          poems.append("\n".join(verse))
      current = []
      collecting = True
      continue
    if not collecting:
      continue
    if re.fullmatch(r"\[[^\]]+\]", stripped):
      continue
    if re.search(r"^(SONG|CANZONE|SESTINA)\b", stripped, re.IGNORECASE):
      continue
    if not stripped:
      continue
    current.append(stripped)

  if current:
    verse = [ln for ln in current if ln.strip()]
    if len(verse) == 14:
      poems.append("\n".join(verse))
  return poems


def _extract_astrophil(text):
  lines = _clean_lines(_trim_gutenberg(text))
  start_matches = [i for i, line in enumerate(lines) if re.search(r"ASTROPHEL AND\s+_?STELLA", line, re.IGNORECASE)]
  if not start_matches:
    raise ValueError("Astrophil and Stella start not found")
  start_idx = start_matches[-1] + 1
  work_lines = lines[start_idx:]
  end_idx = next((i for i, line in enumerate(work_lines) if re.search(r"SONGS\b|APPENDIX\b", line, re.IGNORECASE)), None)
  if end_idx is not None:
    work_lines = work_lines[:end_idx]

  verse_lines = []
  for line in work_lines:
    stripped = line.strip()
    if not stripped:
      continue
    if re.fullmatch(r"\[[^\]]+\]", stripped):
      continue
    # Stop once the sonnet sequence ends and songs begin.
    if re.search(r"^SONG\b", stripped, re.IGNORECASE):
      break
    if len(stripped.split()) > 18:
      continue
    verse_lines.append(stripped)

  poems = []
  for idx in range(0, len(verse_lines), 14):
    chunk = verse_lines[idx:idx + 14]
    if len(chunk) == 14:
      poems.append("\n".join(chunk))
  return poems


def _read_base_sonnets(path):
  text = _read_text(path)
  parts = re.split(r"\n\s*\d+\s*\n", text)[1:]
  return [part.strip() for part in parts if part.strip()]


def _write_numbered_collection(path, title, poems):
  lines = [title, ""]
  for idx, poem in enumerate(poems, start=1):
    lines.append(str(idx))
    lines.append("")
    lines.append(poem.strip())
    lines.append("")
  _write_text(path, "\n".join(lines).rstrip() + "\n")


def main():
  args = parse_args()
  raw_dir = Path(args.raw_dir)
  clean_dir = Path(args.clean_dir)
  raw_dir.mkdir(parents=True, exist_ok=True)
  clean_dir.mkdir(parents=True, exist_ok=True)

  raw_paths = {}
  for name, url in RAW_SOURCES.items():
    raw_path = raw_dir / f"{name}.txt"
    if not raw_path.exists():
      _download(url, raw_path)
    raw_paths[name] = raw_path

  contemporary_sets = {
    "astrophil_and_stella": _extract_astrophil(
      _read_text(raw_paths["astrophil_and_stella"]),
    ),
    "delia": _extract_numbered_sequence(
      _read_text(raw_paths["delia_diana"]),
      r"DELIA\b",
      end_pattern=r"DIANA\b",
    ),
    "diana": _extract_numbered_sequence(
      _read_text(raw_paths["delia_diana"]),
      r"DIANA\b",
    ),
    "idea": _extract_numbered_sequence(
      _read_text(raw_paths["idea_fidessa_chloris"]),
      r"IDEA\b",
      end_pattern=r"FIDESSA\b|FIDESA\b",
    ),
    "fidessa": _extract_numbered_sequence(
      _read_text(raw_paths["idea_fidessa_chloris"]),
      r"FIDESSA\b|FIDESA\b",
      end_pattern=r"CHLORIS\b",
    ),
    "chloris": _extract_numbered_sequence(
      _read_text(raw_paths["idea_fidessa_chloris"]),
      r"CHLORIS\b",
    ),
    "phillis": _extract_numbered_sequence(
      _read_text(raw_paths["phillis_licia"]),
      r"PHILLIS\b",
      end_pattern=r"LICIA\b",
    ),
    "licia": _extract_numbered_sequence(
      _read_text(raw_paths["phillis_licia"]),
      r"LICIA\b",
    ),
    "amoretti": _extract_numbered_sequence(
      _read_text(raw_paths["amoretti"]),
      r"^AMORETTI",
      end_pattern=r"^EPITHALAMION\.?$",
    ),
  }

  for name, poems in contemporary_sets.items():
    _write_numbered_collection(
      clean_dir / f"{name}_clean.txt",
      f"{name.replace('_', ' ').title()}",
      poems,
    )

  base_sonnets = _read_base_sonnets(args.sonnets_path)
  extra_sonnets = []
  for poems in contemporary_sets.values():
    extra_sonnets.extend(poems)
  combined = base_sonnets + extra_sonnets

  _write_numbered_collection(
    args.output,
    "Contemporary Sonnets Augmented Training Set",
    combined,
  )

  print(f"Base sonnets: {len(base_sonnets)}")
  for name, poems in contemporary_sets.items():
    print(f"{name}: {len(poems)}")
  print(f"Added contemporary sonnets: {len(extra_sonnets)}")
  print(f"Total training items: {len(combined)}")
  print(f"Wrote augmented file to {args.output}")


if __name__ == "__main__":
  main()
