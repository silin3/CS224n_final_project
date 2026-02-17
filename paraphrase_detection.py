'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import logging
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from modules.lora import LoRALinear, apply_lora_to_gpt2

from optimizer import AdamW

# LoRA experiment modes:
# - none: full fine-tuning (default)
# - qv: LoRA-QV, apply LoRA to attention query and value projections only
# - all_attn: LoRA-AllAttn, apply LoRA to all attention projections (Q, K, V, O)
# - attn_mlp: LoRA-Attn+MLP, apply LoRA to attention projections and MLP layers

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


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

    lora_mode = getattr(args, 'lora_mode', 'none')
    if lora_mode != 'none':
      apply_lora_to_gpt2(
        self.gpt,
        lora_mode=lora_mode,
        lora_r=getattr(args, 'lora_r', 8),
        lora_alpha=getattr(args, 'lora_alpha', None),
      )
      # Freeze base model, only train LoRA params + classification head
      for param in self.gpt.parameters():
        param.requires_grad = False
      for name, param in self.gpt.named_parameters():
        if 'lora_' in name:
          param.requires_grad = True
      for param in self.paraphrase_detection_head.parameters():
        param.requires_grad = True
    else:
      # Full fine-tuning
      for param in self.gpt.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    ### YOUR CODE HERE
    #raise NotImplementedError
    gpt_out = self.gpt(input_ids, attention_mask)
    last_token = gpt_out['last_token']  # [batch_size, hidden_size]
    logits = self.gpt.hidden_state_to_token(last_token)[:, [3919, 8505]]  # no(0), yes(1)
    return logits


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
  logger = logging.getLogger('paraphrase')
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

  n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  n_total = sum(p.numel() for p in model.parameters())
  logger.info(f"Model params: trainable={n_trainable:,}, total={n_total:,}")

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    saved_flag = ""
    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)
      saved_flag = " [saved]"

    msg = f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}, best dev acc :: {best_dev_acc :.3f}{saved_flag}"
    print(msg)
    logger.info(msg)


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  logger = logging.getLogger('paraphrase')
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  saved_args = saved['args']
  model = ParaphraseGPT(saved_args)
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  # Log model info
  n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  n_total = sum(p.numel() for p in model.parameters())
  logger.info(f"[Test] Loaded model from {args.filepath}")
  logger.info(f"[Test] Model: {saved_args.model_size}, lora_mode={getattr(saved_args, 'lora_mode', 'none')}, "
              f"lora_r={getattr(saved_args, 'lora_r', 'N/A')}, lr={saved_args.lr}, epochs={saved_args.epochs}, "
              f"batch_size={getattr(saved_args, 'batch_size', 'N/A')}")
  logger.info(f"[Test] Params: trainable={n_trainable:,}, total={n_total:,}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  # Evaluate on dev set (has labels)
  dev_para_acc, dev_para_f1, dev_para_y_pred, dev_para_y_true, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  # Compute dev loss
  dev_loss = 0
  dev_batches = 0
  for batch in para_dev_dataloader:
    b_ids = batch['token_ids'].to(device)
    b_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].flatten().to(device)
    logits = model(b_ids, b_mask)
    dev_loss += F.cross_entropy(logits, labels, reduction='mean').item()
    dev_batches += 1
  dev_loss = dev_loss / dev_batches

  logger.info(f"[Test] Dev — acc: {dev_para_acc :.3f}, f1: {dev_para_f1 :.3f}, loss: {dev_loss :.3f}")
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}, f1 :: {dev_para_f1 :.3f}, loss :: {dev_loss :.3f}")

  # Generate test predictions (no labels available)
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  # Autograder BPE token ID：no=3919, yes=8505
  TOKEN_ID_NO, TOKEN_ID_YES = 3919, 8505
  dev_para_out = [TOKEN_ID_YES if p == 1 else TOKEN_ID_NO for p in dev_para_y_pred]
  test_para_out = [TOKEN_ID_YES if p == 1 else TOKEN_ID_NO for p in test_para_y_pred]

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for sent_id, pred in zip(dev_para_sent_ids, dev_para_out):
      f.write(f"{sent_id}, {pred} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for sent_id, pred in zip(test_para_sent_ids, test_para_out):
      f.write(f"{sent_id}, {pred} \n")

  logger.info(f"[Test] Predictions saved to {args.para_dev_out} and {args.para_test_out}")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--test_only", action='store_true', help="Skip training, only run test on saved model")

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  # LoRA options
  parser.add_argument("--lora_mode", type=str, default='none',
                      choices=['none', 'qv', 'all_attn', 'attn_mlp'],
                      help="LoRA experiment: none=full ft, qv=Q+V only, all_attn=Q+K+V+O, attn_mlp=all attn + MLP")
  parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
  parser.add_argument("--lora_alpha", type=float, default=None, help="LoRA alpha (default: lora_r)")

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


def setup_logger(args, lora_suffix):
  """Set up file + console logger for training."""
  os.makedirs('logs', exist_ok=True)
  log_file = f'logs/paraphrase{lora_suffix}.log'
  logger = logging.getLogger('paraphrase')
  logger.setLevel(logging.INFO)
  logger.handlers.clear()
  fh = logging.FileHandler(log_file, mode='a')
  fh.setLevel(logging.INFO)
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
  fh.setFormatter(fmt)
  ch.setFormatter(fmt)
  logger.addHandler(fh)
  logger.addHandler(ch)
  return logger


if __name__ == "__main__":
  args = get_args()
  args = add_arguments(args)  # Add d, l, num_heads before setting filepath
  lora_suffix = f"-lora-{args.lora_mode}" if args.lora_mode != 'none' else ""
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase{lora_suffix}.pt'  # Save path.
  # Use distinct prediction outputs per LoRA experiment
  if args.lora_mode != 'none':
    def add_lora_suffix(path, suffix):
      if '.' in path.rsplit('/', 1)[-1]:
        base, ext = path.rsplit('.', 1)
        return f"{base}{suffix}.{ext}"
      return f"{path}{suffix}"
    args.para_dev_out = add_lora_suffix(args.para_dev_out, lora_suffix)
    args.para_test_out = add_lora_suffix(args.para_test_out, lora_suffix)
  logger = setup_logger(args, lora_suffix)
  logger.info(f"Args: {vars(args)}")
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  if not args.test_only:
    train(args)
  test(args)
