"""
Sonnet generation (CS224N DFP) — full script with:

✓ Top-p sampling + temperature
✓ Generate N candidates + rerank by avg log-prob (model self-score)
✓ no_repeat_ngram_size (prevents degenerate repetition)
✓ Proper padding-masked LM loss
✓ Optional early stopping on validation LM loss (tiny val split by default)
✓ Saves best checkpoint and uses it for submission generation

Usage examples at bottom.
"""

import argparse
import glob
import os
import random
from types import SimpleNamespace
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import GPT2Tokenizer

from datasets import SonnetsDataset, PrefixSonnetsDataset
from models.gpt2 import GPT2Model
from modules.lora import apply_lora_to_gpt2
from optimizer import AdamW

TQDM_DISABLE = False


# -------------------------
# Repro
# -------------------------
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


# -------------------------
# Early stopping
# -------------------------
class EarlyStopping:
  """Early stop on validation loss."""
  def __init__(self, patience=6, min_delta=0.0):
    self.patience = patience
    self.min_delta = min_delta
    self.best = float("inf")
    self.num_bad = 0
    self.should_stop = False

  def step(self, val_loss):
    if val_loss < self.best - self.min_delta:
      self.best = val_loss
      self.num_bad = 0
      return True
    self.num_bad += 1
    if self.num_bad >= self.patience:
      self.should_stop = True
    return False


# -------------------------
# no-repeat ngram helper (batch=1)
# -------------------------
def calc_banned_tokens_no_repeat_ngram(token_ids: torch.Tensor, n: int):
  """
  token_ids: [1, T] (batch=1)
  returns: set of token ids that would create a repeated n-gram if generated next
  """
  if n is None or n <= 0:
    return set()
  seq = token_ids[0].tolist()
  if len(seq) < n - 1:
    return set()

  # prefix (n-1)-gram -> set(next tokens)
  prefix_to_next = {}
  for i in range(len(seq) - n + 1):
    prefix = tuple(seq[i:i + n - 1])
    nxt = seq[i + n - 1]
    prefix_to_next.setdefault(prefix, set()).add(nxt)

  cur_prefix = tuple(seq[-(n - 1):])
  return prefix_to_next.get(cur_prefix, set())


# -------------------------
# Model
# -------------------------
class SonnetGPT(nn.Module):
  """GPT-2 language model for sonnet continuation."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(
      model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
    )
    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    self.tokenizer.pad_token = self.tokenizer.eos_token

    lora_mode = getattr(args, "lora_mode", "none")
    if lora_mode != "none":
      apply_lora_to_gpt2(
        self.gpt,
        lora_mode=lora_mode,
        lora_r=getattr(args, "lora_r", 8),
        lora_alpha=getattr(args, "lora_alpha", None),
      )
      # freeze base
      for p in self.gpt.parameters():
        p.requires_grad = False
      # unfreeze lora params
      for name, p in self.gpt.named_parameters():
        if "lora_" in name:
          p.requires_grad = True
    else:
      for p in self.gpt.parameters():
        p.requires_grad = True

  def forward(self, input_ids, attention_mask):
    gpt_out = self.gpt(input_ids, attention_mask)
    h = gpt_out["last_hidden_state"]           # [B, T, H]
    logits = self.gpt.hidden_state_to_token(h) # [B, T, V]
    return logits

  def get_device(self):
    for p in self.gpt.parameters():
      return p.device

  @torch.no_grad()
  def generate_top_p_once(
    self,
    input_ids,
    attention_mask=None,
    temperature=1.0,
    top_p=0.9,
    max_new_tokens=220,
    min_new_tokens=120,
    no_repeat_ngram_size=3,
    stop_on_eos=True,
  ):
    """
    Single top-p sample + returns avg log-prob over generated tokens (self-score).
    batch=1.
    """
    device = self.get_device()
    token_ids = input_ids.to(device)

    if attention_mask is None:
      attention_mask = torch.ones_like(token_ids, dtype=torch.long, device=device)
    else:
      attention_mask = attention_mask.to(device)

    eos = self.tokenizer.eos_token_id
    prompt_len = token_ids.size(1)

    sum_logp = 0.0
    gen_tokens = 0

    for _ in range(max_new_tokens):
      logits = self.forward(token_ids, attention_mask)[:, -1, :]  # [1, V]
      logits = logits / max(temperature, 1e-8)

      # forbid EOS too early (length control)
      cur_gen_len = token_ids.size(1) - prompt_len
      if cur_gen_len < min_new_tokens:
        logits[:, eos] = -1e9

      # no-repeat ngram constraint
      banned = calc_banned_tokens_no_repeat_ngram(token_ids, no_repeat_ngram_size)
      if banned:
        logits[:, list(banned)] = -1e9

      # sample from top-p
      probs = torch.softmax(logits, dim=-1)               # [1, V]
      sorted_probs, sorted_idx = torch.sort(probs, descending=True)
      cum = torch.cumsum(sorted_probs, dim=-1)
      keep = cum <= top_p
      keep[..., 0] = True
      filtered = sorted_probs * keep
      filtered = filtered / filtered.sum(dim=-1, keepdim=True)

      sampled_pos = torch.multinomial(filtered, 1)        # [1,1] position in sorted
      next_tok = sorted_idx.gather(-1, sampled_pos)       # [1,1] token id

      # accumulate log-prob of chosen token
      log_probs = torch.log_softmax(logits, dim=-1)
      sum_logp += log_probs[0, next_tok.item()].item()
      gen_tokens += 1

      if stop_on_eos and next_tok.item() == eos:
        break

      token_ids = torch.cat([token_ids, next_tok], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1
      )

    avg_logp = sum_logp / max(gen_tokens, 1)
    decoded = self.tokenizer.decode(token_ids[0].tolist())
    return token_ids, decoded, avg_logp

  @torch.no_grad()
  def generate_top_p_rerank(
    self,
    input_ids,
    attention_mask=None,
    num_candidates=5,
    temperature=1.0,
    top_p=0.9,
    max_new_tokens=220,
    min_new_tokens=120,
    no_repeat_ngram_size=3,
  ):
    """
    Generate N candidates with top-p sampling, rerank by avg log-prob, return best.
    """
    best_ids, best_text, best_score = None, None, -1e18
    for _ in range(num_candidates):
      ids, text, score = self.generate_top_p_once(
        input_ids,
        attention_mask=attention_mask,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
      )
      if score > best_score:
        best_ids, best_text, best_score = ids, text, score
    return best_ids, best_text


# -------------------------
# Utils
# -------------------------
def add_arguments(args):
  if args.model_size == "gpt2":
    args.d, args.l, args.num_heads = 768, 12, 12
  elif args.model_size == "gpt2-medium":
    args.d, args.l, args.num_heads = 1024, 24, 16
  elif args.model_size == "gpt2-large":
    args.d, args.l, args.num_heads = 1280, 36, 20
  else:
    raise Exception(f"{args.model_size} is not supported.")
  return args


def auto_version(base_tag):
  v = 1
  while True:
    tag = f"{base_tag}_v{v}"
    if glob.glob(f"*_{tag}-sonnet.pt") or glob.glob(f"{tag}-sonnet.pt"):
      v += 1
    else:
      return v, tag


def _to_namespace(obj):
  if isinstance(obj, argparse.Namespace):
    return obj
  if isinstance(obj, dict):
    return argparse.Namespace(**obj)
  if isinstance(obj, SimpleNamespace):
    return argparse.Namespace(**vars(obj))
  raise TypeError(f"Unsupported args type in checkpoint: {type(obj)}")


def save_model(model, optimizer, args, filepath, epoch=None, best_val=None, early_state=None):
  lora_config = {
    "lora_mode": getattr(args, "lora_mode", "none"),
    "lora_r": getattr(args, "lora_r", 8),
    "lora_alpha": getattr(args, "lora_alpha", None),
  }
  save_info = {
    "model": model.state_dict(),
    "optim": optimizer.state_dict(),
    "args": vars(args),
    "lora_config": lora_config,
    "version": getattr(args, "version", "v1"),
    "epoch": epoch,
    "best_val": best_val,
    "early_state": early_state,
    "system_rng": random.getstate(),
    "numpy_rng": np.random.get_state(),
    "torch_rng": torch.random.get_rng_state(),
    "cuda_rng": torch.cuda.random.get_rng_state_all() if torch.cuda.is_available() else None,
  }
  torch.save(save_info, filepath)
  print(f"Saved checkpoint to {filepath}")


@torch.no_grad()
def compute_lm_loss(model, dataloader, device):
  """Padding-masked next-token LM loss."""
  model.eval()
  total_loss = 0.0
  total_tokens = 0.0

  for batch in tqdm(dataloader, desc="eval", disable=TQDM_DISABLE):
    b_ids = batch["token_ids"].to(device)
    b_mask = batch["attention_mask"].to(device)

    logits = model(b_ids, b_mask)                 # [B, T, V]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = b_ids[:, 1:].contiguous()
    if "loss_mask" in batch:
      shift_mask = batch["loss_mask"][:, 1:].to(device).contiguous().float()
    else:
      shift_mask = b_mask[:, 1:].contiguous().float()

    per_tok = F.cross_entropy(
      shift_logits.view(-1, shift_logits.size(-1)),
      shift_labels.view(-1),
      reduction="none",
    ).view_as(shift_mask)

    total_loss += (per_tok * shift_mask).sum().item()
    total_tokens += shift_mask.sum().item()

  return total_loss / max(total_tokens, 1.0)


# -------------------------
# Train
# -------------------------
def train(args):
  device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

  if args.multi_prefix_train:
    prefix_line_counts = tuple(int(x.strip()) for x in args.prefix_line_counts.split(",") if x.strip())
    full_dataset = PrefixSonnetsDataset(
      args.sonnet_path,
      prefix_line_counts=prefix_line_counts,
      min_target_lines=args.min_target_lines,
    )
    print(f"Using multi-prefix training: prefix_lines={prefix_line_counts}, min_target_lines={args.min_target_lines}, "
          f"examples={len(full_dataset)}")
  else:
    full_dataset = SonnetsDataset(args.sonnet_path)
  n_total = len(full_dataset)

  # Allow val_ratio=0.0 (no early stop)
  if args.val_ratio <= 0.0:
    train_set = full_dataset
    val_set = None
  else:
    n_val = max(1, int(args.val_ratio * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(
      full_dataset,
      [n_train, n_val],
      generator=torch.Generator().manual_seed(args.seed),
    )

  train_loader = DataLoader(
    train_set,
    shuffle=True,
    batch_size=args.batch_size,
    collate_fn=full_dataset.collate_fn,
  )

  val_loader = None
  if val_set is not None:
    val_loader = DataLoader(
      val_set,
      shuffle=False,
      batch_size=args.batch_size,
      collate_fn=full_dataset.collate_fn,
    )

  args = add_arguments(args)
  model = SonnetGPT(args).to(device)

  # Only optimize trainable params (important for LoRA)
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  optimizer = AdamW(trainable_params, lr=args.lr)

  early = EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)

  best_path = f"best_{args.filepath}"
  best_val = float("inf")
  start_epoch = 0

  if args.resume_from is not None:
    print(f"Resuming from {args.resume_from}")
    saved = torch.load(args.resume_from, weights_only=False, map_location=device)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optim"])
    start_epoch = int(saved.get("epoch", -1)) + 1
    best_val = float(saved.get("best_val", float("inf")))
    early_state = saved.get("early_state")
    if early_state is not None:
      early.best = float(early_state.get("best", best_val))
      early.num_bad = int(early_state.get("num_bad", 0))
      early.should_stop = bool(early_state.get("should_stop", False))
    else:
      early.best = best_val

    if "system_rng" in saved:
      random.setstate(saved["system_rng"])
    if "numpy_rng" in saved:
      np.random.set_state(saved["numpy_rng"])
    if "torch_rng" in saved:
      torch.random.set_rng_state(saved["torch_rng"])
    if device.type == "cuda" and "cuda_rng" in saved:
      torch.cuda.random.set_rng_state_all(saved["cuda_rng"])

    print(f"Resume state: start_epoch={start_epoch}, best_val={best_val:.4f}, num_bad={early.num_bad}")

  for epoch in range(start_epoch, args.epochs):
    model.train()
    running = 0.0
    nb = 0

    for batch in tqdm(train_loader, desc=f"train-{epoch}", disable=TQDM_DISABLE):
      b_ids = batch["token_ids"].to(device)
      b_mask = batch["attention_mask"].to(device)

      optimizer.zero_grad(set_to_none=True)
      logits = model(b_ids, b_mask)

      # padding-masked LM loss
      shift_logits = logits[:, :-1, :].contiguous()
      shift_labels = b_ids[:, 1:].contiguous()
      if "loss_mask" in batch:
        shift_mask = batch["loss_mask"][:, 1:].to(device).contiguous().float()
      else:
        shift_mask = b_mask[:, 1:].contiguous().float()

      per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
      ).view_as(shift_mask)

      loss = (per_tok * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
      optimizer.step()

      running += loss.item()
      nb += 1

    train_loss = running / max(nb, 1)

    if val_loader is None:
      print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
      early_state = {"best": early.best, "num_bad": early.num_bad, "should_stop": early.should_stop}
      save_model(model, optimizer, args, args.filepath, epoch=epoch, best_val=best_val, early_state=early_state)
      save_model(model, optimizer, args, best_path, epoch=epoch, best_val=best_val, early_state=early_state)
      continue

    val_loss = compute_lm_loss(model, val_loader, device)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    improved = val_loss < best_val - args.early_stop_min_delta

    if improved:
      best_val = val_loss
      early.best = best_val
      early.num_bad = 0
      early.should_stop = False
      early_state = {"best": early.best, "num_bad": early.num_bad, "should_stop": early.should_stop}
      save_model(model, optimizer, args, best_path, epoch=epoch, best_val=best_val, early_state=early_state)
    else:
      early.step(val_loss)
    early_state = {"best": early.best, "num_bad": early.num_bad, "should_stop": early.should_stop}
    save_model(model, optimizer, args, args.filepath, epoch=epoch, best_val=best_val, early_state=early_state)
    if (not improved) and early.should_stop:
      print(f"Early stopping at epoch {epoch}. Best val_loss={best_val:.4f}")
      break

  return best_path


# -------------------------
# Generate submission
# -------------------------
@torch.no_grad()
def generate_submission_sonnets(args, checkpoint_path):
  device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
  saved = torch.load(checkpoint_path, weights_only=False)

  model = SonnetGPT(_to_namespace(saved["args"]))
  model.load_state_dict(saved["model"])
  model = model.to(device)
  model.eval()

  held_out = SonnetsDataset(args.held_out_sonnet_path)
  os.makedirs(os.path.dirname(args.sonnet_out), exist_ok=True)

  generated = []
  for batch in held_out:
    sid, prompt = batch[0], batch[1]
    enc = model.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(device)

    _, decoded = model.generate_top_p_rerank(
      enc["input_ids"],
      attention_mask=enc.get("attention_mask", None),
      num_candidates=args.num_candidates,
      temperature=args.temperature,
      top_p=args.top_p,
      max_new_tokens=args.max_new_tokens,
      min_new_tokens=args.min_new_tokens,
      no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    generated.append((sid, decoded + "\n\n"))
    if args.print_submission:
      print(decoded)
      print()

  with open(args.sonnet_out, "w+", encoding="utf-8") as f:
    f.write("--Generated Sonnets-- \n\n")
    for sid, text in generated:
      f.write(f"\n{sid}\n")
      f.write(text)

  print(f"Wrote predictions to {args.sonnet_out}")


# -------------------------
# Args
# -------------------------
def get_args():
  p = argparse.ArgumentParser()

  p.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  p.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  p.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  p.add_argument("--run_name", type=str, default=None,
                 help="Optional experiment label used in checkpoint and prediction filenames")
  p.add_argument("--resume_from", type=str, default=None,
                 help="Resume sonnet training from a saved checkpoint")

  p.add_argument("--seed", type=int, default=11711)
  p.add_argument("--epochs", type=int, default=10)
  p.add_argument("--use_gpu", action="store_true")

  # Train
  p.add_argument("--batch_size", type=int, default=8)
  p.add_argument("--lr", type=float, default=1e-5)
  p.add_argument("--grad_clip", type=float, default=1.0)
  p.add_argument("--multi_prefix_train", action="store_true",
                 help="Train on multiple prefix->continuation examples built from each sonnet")
  p.add_argument("--prefix_line_counts", type=str, default="4,6,8",
                 help="Comma-separated prefix line counts used when --multi_prefix_train is enabled")
  p.add_argument("--min_target_lines", type=int, default=4,
                 help="Minimum number of continuation lines to keep in prefix training mode")

  # Val + early stop (tiny split; set 0 to disable)
  p.add_argument("--val_ratio", type=float, default=0.02)
  p.add_argument("--early_stop_patience", type=int, default=6)
  p.add_argument("--early_stop_min_delta", type=float, default=0.0)

  # Model size
  p.add_argument("--model_size", type=str,
                 choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                 default="gpt2")

  # LoRA
  p.add_argument("--lora_mode", type=str, default="none",
                 choices=["none", "qv", "all_attn", "attn_mlp"])
  p.add_argument("--lora_r", type=int, default=8)
  p.add_argument("--lora_alpha", type=float, default=None)

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

  if args.lora_alpha is None:
    args.lora_alpha = float(args.lora_r)

  if args.lora_mode != "none":
    lora_suffix = f"-lora-{args.lora_mode}-r{args.lora_r}-a{int(args.lora_alpha)}"
  else:
    lora_suffix = ""

  if args.resume_from is not None:
    exp_tag = os.path.basename(args.resume_from)
    if exp_tag.endswith("-sonnet.pt"):
      exp_tag = exp_tag[:-10]
    args.filepath = f"{exp_tag}-sonnet.pt"
    args.sonnet_out = f"predictions/sonnets-{exp_tag}.txt"
    args.version = "resume"
  else:
    if args.run_name is not None:
      base_tag = args.run_name
    else:
      base_tag = f"{args.model_size}-{args.epochs}-{args.lr}{lora_suffix}"
    version, exp_tag = auto_version(base_tag)
    args.version = f"v{version}"

    args.filepath = f"{exp_tag}-sonnet.pt"
    args.sonnet_out = f"predictions/sonnets-{exp_tag}.txt"

  print(f"Experiment: {exp_tag}")
  print(f"  Best checkpoint: best_{args.filepath}")
  print(f"  Predictions: {args.sonnet_out}")
  if args.lora_mode != "none":
    print(f"  LoRA: mode={args.lora_mode}, r={args.lora_r}, alpha={args.lora_alpha}")
  print(f"  Train: lr={args.lr}, batch={args.batch_size}, val_ratio={args.val_ratio}, patience={args.early_stop_patience}")
  if args.multi_prefix_train:
    print(f"  Prefix train: lines={args.prefix_line_counts}, min_target_lines={args.min_target_lines}")
  print(f"  Gen: temp={args.temperature}, top_p={args.top_p}, N={args.num_candidates}, ngram={args.no_repeat_ngram_size}, "
        f"min_new={args.min_new_tokens}, max_new={args.max_new_tokens}")

  seed_everything(args.seed)

  best_ckpt = train(args)
  generate_submission_sonnets(args, best_ckpt)


if __name__ == "__main__":
  main()
