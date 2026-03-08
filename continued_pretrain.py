#!/usr/bin/env python3

"""
Continued pretraining (causal LM) for GPT-2 on a plain-text corpus.
Intended use: domain-adapt GPT-2 on PAWS text, then fine-tune on paraphrase detection.
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer

from models.gpt2 import GPT2Model
from optimizer import AdamW

TQDM_DISABLE = False


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def add_model_size_args(args):
  if args.model_size == "gpt2":
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == "gpt2-medium":
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == "gpt2-large":
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise ValueError(f"Unsupported model_size: {args.model_size}")
  return args


class TextLineDataset(Dataset):
  def __init__(self, text_path):
    self.lines = []
    with open(text_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line:
          self.lines.append(line)
    if len(self.lines) == 0:
      raise ValueError(f"No non-empty lines found in {text_path}")

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    return self.lines[idx]


class GPT2CPT(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(
      model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
    )

  def forward(self, input_ids, attention_mask):
    out = self.gpt(input_ids, attention_mask)
    return self.gpt.hidden_state_to_token(out["last_hidden_state"])


def _extract_model_state_dict(saved):
  if isinstance(saved, dict):
    if "model" in saved and isinstance(saved["model"], dict):
      return saved["model"]
    if "gpt_model" in saved and isinstance(saved["gpt_model"], dict):
      return {f"gpt.{k}": v for k, v in saved["gpt_model"].items()}
  if isinstance(saved, dict):
    return saved
  raise TypeError("Unsupported checkpoint format for model state dict")


def setup_logger(exp_tag):
  os.makedirs("logs", exist_ok=True)
  log_path = f"logs/continued_pretrain-{exp_tag}.log"
  logger = logging.getLogger("continued_pretrain")
  logger.setLevel(logging.INFO)
  logger.handlers.clear()
  fh = logging.FileHandler(log_path, mode="a")
  ch = logging.StreamHandler()
  fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
  fh.setFormatter(fmt)
  ch.setFormatter(fmt)
  logger.addHandler(fh)
  logger.addHandler(ch)
  return logger


def save_checkpoint(model, optimizer, args, filepath, epoch, train_loss, best_train_loss):
  save_info = {
    "model": model.state_dict(),
    "gpt_model": model.gpt.state_dict(),
    "optim": optimizer.state_dict(),
    "args": vars(args),
    "epoch": int(epoch),
    "train_loss": float(train_loss),
    "best_train_loss": float(best_train_loss),
    "system_rng": random.getstate(),
    "numpy_rng": np.random.get_state(),
    "torch_rng": torch.random.get_rng_state(),
  }
  torch.save(save_info, filepath)


def train(args):
  device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
  args = add_model_size_args(args)
  logger = setup_logger(args.exp_tag)
  logger.info(f"Args: {vars(args)}")

  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  tokenizer.pad_token = tokenizer.eos_token

  dataset = TextLineDataset(args.train_text)

  def collate_fn(batch_texts):
    enc = tokenizer(
      batch_texts,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=args.max_length,
    )
    return {
      "token_ids": torch.LongTensor(enc["input_ids"]),
      "attention_mask": torch.LongTensor(enc["attention_mask"]),
    }

  train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
  )

  model = GPT2CPT(args).to(device)
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

  start_epoch = 0
  best_train_loss = float("inf")
  if args.resume_from is not None:
    logger.info(f"Resuming from {args.resume_from}")
    saved = torch.load(args.resume_from, weights_only=False)
    state_dict = _extract_model_state_dict(saved)
    model.load_state_dict(state_dict)
    if "optim" in saved:
      optimizer.load_state_dict(saved["optim"])
    start_epoch = int(saved.get("epoch", -1)) + 1
    best_train_loss = float(saved.get("best_train_loss", float("inf")))
    logger.info(f"Resume state: start_epoch={start_epoch}, best_train_loss={best_train_loss:.6f}")

  if start_epoch >= args.epochs:
    logger.info(f"Skip training: start_epoch={start_epoch} >= epochs={args.epochs}")
    return

  os.makedirs(os.path.dirname(args.filepath) or ".", exist_ok=True)
  best_path = f"best_{args.filepath}"

  for epoch in range(start_epoch, args.epochs):
    model.train()
    running = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc=f"train-{epoch}", disable=TQDM_DISABLE):
      b_ids = batch["token_ids"].to(device)
      b_mask = batch["attention_mask"].to(device)

      optimizer.zero_grad(set_to_none=True)
      logits = model(b_ids, b_mask)

      shift_logits = logits[:, :-1, :].contiguous()
      shift_labels = b_ids[:, 1:].contiguous()
      shift_mask = b_mask[:, 1:].contiguous().float()

      per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
      ).view_as(shift_mask)
      loss = (per_tok * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

      running += loss.item()
      num_batches += 1

    train_loss = running / max(num_batches, 1)
    improved = train_loss < best_train_loss
    if improved:
      best_train_loss = train_loss
      save_checkpoint(model, optimizer, args, best_path, epoch, train_loss, best_train_loss)
    save_checkpoint(model, optimizer, args, args.filepath, epoch, train_loss, best_train_loss)

    msg = (
      f"Epoch {epoch}: train_lm_loss={train_loss:.6f}, "
      f"best_train_lm_loss={best_train_loss:.6f}"
    )
    if improved:
      msg += " [saved best]"
    logger.info(msg)
    print(msg)

  logger.info(f"Finished. Last checkpoint: {args.filepath}; best checkpoint: {best_path}")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_text", type=str, default="data/paws-cpt-train.txt")
  parser.add_argument("--epochs", type=int, default=3)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--grad_clip", type=float, default=1.0)
  parser.add_argument("--max_length", type=int, default=128)
  parser.add_argument("--model_size", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large"])
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--use_gpu", action="store_true")
  parser.add_argument("--resume_from", type=str, default=None)
  parser.add_argument("--filepath", type=str, default=None,
                      help="Output checkpoint path; default auto-generated")
  args = parser.parse_args()
  return args


def main():
  args = get_args()
  seed_everything(args.seed)

  args.exp_tag = f"{args.model_size}-paws-cpt-{args.epochs}-{args.lr}-bs{args.batch_size}"
  if args.filepath is None:
    args.filepath = f"{args.exp_tag}.pt"

  train(args)


if __name__ == "__main__":
  main()
