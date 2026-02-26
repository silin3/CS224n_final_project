'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    Produce a logit for each token in our sequence.
    """
    gpt_out = self.gpt(input_ids, attention_mask)
    last_hidden_state = gpt_out['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(last_hidden_state)
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(
      self,
      encoding,
      temperature=0.9,
      top_p=0.95,
      max_length=256,
      num_beams=5,
      length_penalty=0.8,
      repetition_penalty=1.08,
      rep_window=64,
      no_repeat_ngram_size=2,
      stop_at_14_lines=True,
  ):
    """
    Top-p constrained beam-ish decoding with:
    - mild repetition penalty on recent window
    - no-repeat ngram constraint
    - robust 14-line stopping (counts non-empty lines)
    - forbids early EOS before reaching 14 lines
    """
    device = self.get_device()
    eos_id = self.tokenizer.eos_token_id

    token_ids0 = encoding.to(device)
    attn0 = torch.ones_like(token_ids0, dtype=torch.int64, device=device)

    beams = [(token_ids0, attn0, 0.0, False)]  # (ids, attn, logp, ended)

    # Tokens we should NOT penalize (newline + common punctuation)
    no_penalty_tokens = set()
    for s in ["\n", ",", ".", ";", ":", "!", "?", "'", "\"", "-", "—"]:
      for t in self.tokenizer.encode(s, add_special_tokens=False):
        no_penalty_tokens.add(t)

    def banned_next(prefix_ids):
      if no_repeat_ngram_size <= 1:
        return set()
      n = no_repeat_ngram_size
      if len(prefix_ids) < n - 1:
        return set()
      seen = {}
      for i in range(len(prefix_ids) - n + 1):
        key = tuple(prefix_ids[i:i + n - 1])
        nxt = prefix_ids[i + n - 1]
        seen.setdefault(key, set()).add(nxt)
      key = tuple(prefix_ids[-(n - 1):])
      return seen.get(key, set())

    def non_empty_line_count_from_ids(ids_1d):
      text = self.tokenizer.decode(ids_1d)
      lines = [ln for ln in text.split("\n") if ln.strip() != ""]
      return len(lines)

    for _ in range(max_length):
      if all(b[3] for b in beams):
        break

      candidates = []

      for token_ids, attn_mask, score, ended in beams:
        if ended:
          candidates.append((token_ids, attn_mask, score, True))
          continue

        logits_seq = self.forward(token_ids, attn_mask)  # [1, T, V]
        logits = logits_seq[:, -1, :] / max(temperature, 1e-6)  # [1, V]

        # Mild repetition penalty on a sliding window (do NOT penalize punctuation/newlines)
        recent = token_ids[0].tolist()[-rep_window:]
        for tok in set(recent):
          if tok in no_penalty_tokens:
            continue
          if logits[0, tok] > 0:
            logits[0, tok] = logits[0, tok] / repetition_penalty
          else:
            logits[0, tok] = logits[0, tok] * repetition_penalty

        probs = F.softmax(logits, dim=-1)  # [1, V]

        # Top-p (nucleus) filter (standard)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = (cum <= top_p).sum(dim=-1).item()
        keep = max(1, keep)
        filt_probs = sorted_probs[:, :keep]
        filt_idx = sorted_idx[:, :keep]
        filt_logp = torch.log(filt_probs + 1e-12)

        topM = min(50, keep)
        topM_logp, topM_pos = torch.topk(filt_logp, k=topM, dim=-1)
        topM_tok = filt_idx.gather(-1, topM_pos)

        prefix = token_ids[0].tolist()
        banned = banned_next(prefix)

        expanded = 0
        for k in range(topM):
          next_tok = topM_tok[:, k:k + 1]  # [1,1]
          tok_id = next_tok.item()

          # ---- FIX: forbid early EOS before reaching 14 lines ----
          if stop_at_14_lines and tok_id == eos_id:
            cur_lines = non_empty_line_count_from_ids(token_ids[0].tolist())
            if cur_lines < 14:
              continue

          if tok_id in banned:
            continue

          next_logp = topM_logp[:, k].item()
          new_ids = torch.cat([token_ids, next_tok], dim=1)
          new_attn = torch.cat(
            [attn_mask, torch.ones((1, 1), dtype=torch.int64, device=device)],
            dim=1
          )

          new_ended = (tok_id == eos_id)

          # ---- FIX: robust 14-line stopping using non-empty line count ----
          if stop_at_14_lines and not new_ended:
            new_lines = non_empty_line_count_from_ids(new_ids[0].tolist())
            if new_lines >= 14:
              new_ended = True

          candidates.append((new_ids, new_attn, score + next_logp, new_ended))
          expanded += 1
          if expanded >= num_beams:
            break

        # Fallback: if everything got banned, allow the best token anyway
        if expanded == 0:
          next_tok = topM_tok[:, 0:1]
          next_logp = topM_logp[:, 0].item()
          new_ids = torch.cat([token_ids, next_tok], dim=1)
          new_attn = torch.cat(
            [attn_mask, torch.ones((1, 1), dtype=torch.int64, device=device)],
            dim=1
          )
          tok_id = next_tok.item()
          new_ended = (tok_id == eos_id)
          if stop_at_14_lines and not new_ended:
            new_lines = non_empty_line_count_from_ids(new_ids[0].tolist())
            if new_lines >= 14:
              new_ended = True
          candidates.append((new_ids, new_attn, score + next_logp, new_ended))

      # prune beams with length penalty
      def norm_score(c):
        ids, _, sc, _ = c
        L = ids.shape[1]
        return sc / (L ** length_penalty)

      candidates.sort(key=norm_score, reverse=True)
      beams = candidates[:num_beams]

    best = max(beams, key=lambda b: b[2] / (b[0].shape[1] ** length_penalty))
    best_ids = best[0][0].tolist()

    # ---- FIX: strip eos if present at end (avoid "<|endoftext|>") ----
    if best_ids and best_ids[-1] == eos_id:
      best_ids = best_ids[:-1]

    decoded = self.tokenizer.decode(best_ids)
    return best[0], decoded


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  sonnet_dataset = SonnetsDataset(args.sonnet_path)

  num_samples = len(sonnet_dataset)
  num_val = max(1, int(num_samples * args.val_ratio))
  perm = torch.randperm(num_samples)
  val_indices = perm[:num_val].tolist()
  train_indices = perm[num_val:].tolist()

  train_dataset = Subset(sonnet_dataset, train_indices)
  val_dataset = Subset(sonnet_dataset, val_indices)

  sonnet_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=sonnet_dataset.collate_fn
  )
  val_dataloader = DataLoader(
    val_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=sonnet_dataset.collate_fn
  )

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args).to(device)

  optimizer = AdamW(model.parameters(), lr=args.lr)
  best_val_loss = float('inf')
  patience_counter = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / max(1, num_batches)

    model.eval()
    val_loss = 0
    val_batches = 0
    for batch in val_dataloader:
      b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')
      val_loss += loss.item()
      val_batches += 1
    val_loss = val_loss / max(1, val_batches)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, val loss :: {val_loss :.3f}.")
    print('Generating several output sonnets...')
    for batch in held_out_sonnet_dataset:
      prompt = batch[1]
      encoding = model.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)

      output = model.generate(
        encoding['input_ids'],
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
      )

      # ---- FIX: print prompt + continuation only (avoid duplicating the first 3 lines) ----
      prompt_len = encoding['input_ids'].shape[1]
      gen_ids = output[0][0]  # includes prompt
      cont_ids = gen_ids[prompt_len:].tolist()
      if cont_ids and cont_ids[-1] == model.tokenizer.eos_token_id:
        cont_ids = cont_ids[:-1]
      cont_text = model.tokenizer.decode(cont_ids)

      print(f'{prompt}{cont_text}\n\n')

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      save_model(model, optimizer, args, args.filepath)
    else:
      patience_counter += 1
      if patience_counter >= args.patience:
        print(f"Early stopping at epoch {epoch}: no val improvement for {args.patience} epochs.")
        break


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(
      encoding['input_ids'],
      temperature=args.temperature,
      top_p=args.top_p,
      repetition_penalty=args.repetition_penalty,
      no_repeat_ngram_size=args.no_repeat_ngram_size,
    )[0][0]

    decoded_output = model.tokenizer.decode(output.tolist())

    # Prefer good content: truncate extra lines but不要用空行硬凑
    lines = decoded_output.splitlines()
    if len(lines) > 14:
      lines = lines[:14]
    full_sonnet = "\n".join(lines) + "\n\n"
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--repetition_penalty", type=float, default=1.15,
                      help="Penalty for previously generated tokens (>1.0 reduces repetition).")
  parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                      help="Disallow repeating n-grams of this size.")

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--val_ratio", type=float, default=0.1,
                      help="Fraction of training data used for validation.")
  parser.add_argument("--patience", type=int, default=3,
                      help="Stop training when validation loss does not improve for N epochs.")
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


def unique_filepath(base):
  """If base path exists, append _v2, _v3, ... to avoid overwriting."""
  if not os.path.exists(base):
    return base
  stem, ext = os.path.splitext(base)
  ver = 2
  while os.path.exists(f'{stem}_v{ver}{ext}'):
    ver += 1
  return f'{stem}_v{ver}{ext}'


if __name__ == "__main__":
  args = get_args()
  base_filepath = f'{args.model_size}-{args.epochs}-{args.lr}-sonnet.pt'
  args.filepath = unique_filepath(base_filepath)
  args.sonnet_out = unique_filepath(
    f'predictions/sonnets-{args.model_size}-{args.epochs}-{args.lr}.txt'
  )
  seed_everything(args.seed)
  train(args)
  generate_submission_sonnets(args)