#!/usr/bin/env python3
import os
import csv
import torch
from sacrebleu.metrics import CHRF
from datasets import SonnetsDataset

PRED_DIR = "/home/hehaixing/CS224n_final_project/predictions"
DATA_DIR = "/home/hehaixing/CS224n_final_project/data"
OUT_PATH = os.path.join(PRED_DIR, "test_predictions_results.txt")

SONNET_FILES = [
  "generated_sonnets.txt",
  "sonnets-gpt2-10-1e-05-lora-attn_mlp.txt",
  "sonnets-gpt2-10-1e-05_v7.txt",
  "sonnets-gpt2-10-1e-05-lora-attn_mlp.txt",
  "sonnets-gpt2-10-0.0001-lora-qv-r8-a8_v1.txt",
  "sonnets-gpt2-10-0.0001-lora-all_attn-r8-a8_v1.txt",
  "sonnets-gpt2-10-5e-05-lora-attn_mlp-r8-a8_v1.txt",
  "sonnets-gpt2-10-1e-05_v1.txt",
  "sonnets-gpt2-10-1e-05-lora-attn_mlp-r8-a8_v1.txt",
  "sonnets-gpt2-10-1e-05_v2.txt",
  "sonnets-gpt2-10-1e-05-lora-attn_mlp-r8-a8_v1.txt",
  "sonnets-gpt2-10-1e-05_v1.txt",
  "sonnets-gpt2-10-1e-05-lora-attn_mlp-r8-a8_v2.txt",


]

PARA_DEV_FILES = [
  "para-dev-gpt2-10-1e-05.csv",
  "para-dev-gpt2-10-1e-05-lora-qv.csv",
  "para-dev-gpt2-10-1e-05-lora-attn_mlp.csv",
  "para-dev-gpt2-10-2e-05-lora-attn_mlp.csv",
  "para-dev-gpt2-large-10-2e-05-lora-all_attn.csv",
]

PARA_TEST_FILES = [
  "para-test-gpt2-10-1e-05.csv",
  "para-test-gpt2-10-1e-05-lora-qv.csv",
  "para-test-gpt2-10-1e-05-lora-attn_mlp.csv",
  "para-test-gpt2-10-2e-05-lora-attn_mlp.csv",
  "para-test-gpt2-large-10-2e-05-lora-all_attn.csv",
]

TOKEN_ID_NO = 3919
TOKEN_ID_YES = 8505


def log(msg, lines):
  print(msg)
  lines.append(msg)


def test_sonnet_file(pred_path, gold_path):
  chrf = CHRF()
  generated = [x[1] for x in SonnetsDataset(pred_path)]
  gold = [x[1] for x in SonnetsDataset(gold_path)]
  n = min(len(generated), len(gold))
  generated = generated[:n]
  gold = gold[:n]
  return float(chrf.corpus_score(generated, [gold]).score)


def load_quora_dev_gold(dev_path):
  gold = {}
  with open(dev_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
      gold[r["id"].strip()] = int(float(r["is_duplicate"]))
  return gold


def load_quora_test_ids(test_path):
  ids = set()
  with open(test_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
      ids.add(r["id"].strip())
  return ids


def load_para_pred(csv_path):
  pred = {}
  with open(csv_path, "r", encoding="utf-8") as f:
    _ = f.readline()
    for line in f:
      line = line.strip()
      if not line:
        continue
      parts = [p.strip() for p in line.split(",")]
      if len(parts) < 2:
        continue
      sid = parts[0]
      try:
        v = int(parts[1])
      except ValueError:
        continue
      if v in (TOKEN_ID_NO, TOKEN_ID_YES):
        v = 1 if v == TOKEN_ID_YES else 0
      pred[sid] = v
  return pred


def macro_f1_binary(y_true, y_pred):
  y_true = torch.tensor(y_true, dtype=torch.long)
  y_pred = torch.tensor(y_pred, dtype=torch.long)
  f1s = []
  for c in (0, 1):
    tp = ((y_pred == c) & (y_true == c)).sum().item()
    fp = ((y_pred == c) & (y_true != c)).sum().item()
    fn = ((y_pred != c) & (y_true == c)).sum().item()
    denom = 2 * tp + fp + fn
    f1s.append((2 * tp / denom) if denom > 0 else 0.0)
  return sum(f1s) / 2.0


def eval_para_dev(pred_csv, gold_map):
  pred = load_para_pred(pred_csv)
  y_true, y_pred = [], []
  missing = 0
  extra = 0

  for sid, y in gold_map.items():
    if sid not in pred:
      missing += 1
      continue
    y_true.append(y)
    y_pred.append(pred[sid])

  for sid in pred.keys():
    if sid not in gold_map:
      extra += 1

  if len(y_true) == 0:
    return None, None, 0, missing, extra

  yt = torch.tensor(y_true)
  yp = torch.tensor(y_pred)
  acc = (yt == yp).float().mean().item()
  f1 = macro_f1_binary(y_true, y_pred)
  return acc, f1, len(y_true), missing, extra


def check_para_test(pred_csv, gold_ids):
  pred = load_para_pred(pred_csv)
  pred_ids = set(pred.keys())
  missing = len(gold_ids - pred_ids)
  extra = len(pred_ids - gold_ids)
  return len(pred_ids), missing, extra


def main():
  lines = []

  log(f"Pred dir: {PRED_DIR}", lines)
  log(f"Data dir: {DATA_DIR}", lines)
  log("", lines)

  sonnet_gold_dev = os.path.join(DATA_DIR, "TRUE_sonnets_held_out_dev.txt")
  log("=== Sonnet DEV CHRF ===", lines)
  for fn in SONNET_FILES:
    path = os.path.join(PRED_DIR, fn)
    if not os.path.exists(path):
      log(f"{fn}: missing", lines)
      continue
    score = test_sonnet_file(path, sonnet_gold_dev)
    log(f"{fn}: CHRF={score:.3f}", lines)
  log("", lines)

  gold_dev = load_quora_dev_gold(os.path.join(DATA_DIR, "quora-dev.csv"))
  log("=== Paraphrase DEV (acc / macro-F1) ===", lines)
  for fn in PARA_DEV_FILES:
    path = os.path.join(PRED_DIR, fn)
    if not os.path.exists(path):
      log(f"{fn}: missing", lines)
      continue
    acc, f1, n, missing, extra = eval_para_dev(path, gold_dev)
    if acc is None:
      log(f"{fn}: no matched rows", lines)
    else:
      log(f"{fn}: acc={acc:.4f} f1={f1:.4f} n={n} missing={missing} extra={extra}", lines)
  log("", lines)

  gold_test_ids = load_quora_test_ids(os.path.join(DATA_DIR, "quora-test-student.csv"))
  log("=== Paraphrase TEST (coverage check) ===", lines)
  for fn in PARA_TEST_FILES:
    path = os.path.join(PRED_DIR, fn)
    if not os.path.exists(path):
      log(f"{fn}: missing", lines)
      continue
    rows, missing, extra = check_para_test(path, gold_test_ids)
    log(f"{fn}: rows={rows} missing={missing} extra={extra}", lines)

  with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  print(f"\nSaved results to: {OUT_PATH}")


if __name__ == "__main__":
  main()