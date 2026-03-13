# !/usr/bin/env python3


"""
This file contains our Dataset class for Quora paraphrase detection. You may want to modify this file to train on
additional sources of data, or if you change how the Quora dataset is processed (i.e. data augmentation, etc.).
"""

import csv
import random

import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    labels = torch.LongTensor([x[2] for x in all_data])
    # labels = ['yes' if label == 1 else 'no' for label in [x[2] for x in all_data]]
    # labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids']
    sent_ids = [x[3] for x in all_data]

    cloze_style_sents = [f'Question 1: "{s1}"\nQuestion 2: "{s2}\nAre these questions asking the same thing?\n' for
                         (s1, s2) in zip(sent1, sent2)]
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sent_ids': sent_ids
    }

    return batched_data


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    sent_ids = [x[2] for x in all_data]

    cloze_style_sents = [f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ' for (s1, s2) in
                         zip(sent1, sent2)]

    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids
    }

    return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        paraphrase_data.append((preprocess_string(record['sentence1']),
                                preprocess_string(record['sentence2']),
                                sent_id))

  else:
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          paraphrase_data.append((preprocess_string(record['sentence1']),
                                  preprocess_string(record['sentence2']),
                                  int(float(record['is_duplicate'])), sent_id))
        except:
          pass

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  return paraphrase_data


def _get_record_value(record, keys, default=None):
  for key in keys:
    if key in record and record[key] is not None and record[key] != '':
      return record[key]
  return default


def load_paws_data(paws_filename, split='train'):
  """Load PAWS/PAWS-X style data from a .csv file that is tab-delimited."""
  paws_data = []
  with open(paws_filename, 'r') as fp:
    for record in csv.DictReader(fp, delimiter='\t'):
      try:
        sent_id_raw = _get_record_value(record, ['id', 'pairID', 'sentence_pair_id'], default='')
        sent1_raw = _get_record_value(record, ['sentence1', 'sentence1_text', 'sent1'])
        sent2_raw = _get_record_value(record, ['sentence2', 'sentence2_text', 'sent2'])

        if sent1_raw is None or sent2_raw is None:
          continue

        sent_id = str(sent_id_raw).lower().strip() if sent_id_raw != '' else f'paws-{len(paws_data)}'
        sent1 = preprocess_string(sent1_raw)
        sent2 = preprocess_string(sent2_raw)

        if split == 'test':
          paws_data.append((sent1, sent2, sent_id))
        else:
          label_raw = _get_record_value(record, ['label', 'noisy_label', 'is_duplicate', 'paraphrase_label'])
          if label_raw is None:
            continue
          label = int(float(label_raw))
          paws_data.append((sent1, sent2, label, sent_id))
      except:
        pass

  print(f"Loaded {len(paws_data)} {split} examples from {paws_filename}")
  return paws_data


def load_mixed_paraphrase_data(quora_data, paws_data, paws_mix_ratio=0.3, seed=11711, max_paws_samples=None):
  """
  Mix Quora and PAWS train data.
  paws_mix_ratio controls how many PAWS samples to add relative to Quora size.
  e.g., ratio=0.3 means add about 0.3 * len(quora) PAWS examples.
  """
  mixed_data = list(quora_data)
  if paws_mix_ratio <= 0 or len(paws_data) == 0:
    print(f"Mixed data size: quora={len(quora_data)}, paws_used=0, total={len(mixed_data)}")
    return mixed_data

  target_paws = int(len(quora_data) * paws_mix_ratio)
  if max_paws_samples is not None:
    target_paws = min(target_paws, max_paws_samples)
  target_paws = min(target_paws, len(paws_data))

  rng = random.Random(seed)
  paws_selected = rng.sample(paws_data, target_paws) if target_paws < len(paws_data) else list(paws_data)

  # Prefix PAWS ids to avoid collisions with Quora ids.
  paws_selected = [(s1, s2, label, f'paws_{sid}') for (s1, s2, label, sid) in paws_selected]

  mixed_data.extend(paws_selected)
  rng.shuffle(mixed_data)
  print(f"Mixed data size: quora={len(quora_data)}, paws_used={len(paws_selected)}, total={len(mixed_data)}")
  return mixed_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data


class PrefixSonnetsDataset(Dataset):
  """Build multiple prefix->continuation LM examples from each sonnet."""

  def __init__(self, file_path, prefix_line_counts=(4, 6, 8), min_target_lines=4):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.examples = self._build_examples(file_path, prefix_line_counts, min_target_lines)

  def _load_sonnets(self, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]
    return [s.strip() for s in sonnets]

  def _build_examples(self, file_path, prefix_line_counts, min_target_lines):
    examples = []
    sonnets = self._load_sonnets(file_path)

    for sid, sonnet in enumerate(sonnets):
      lines = [line.strip() for line in sonnet.splitlines() if line.strip()]
      if len(lines) < min_target_lines + 1:
        continue

      full_text = '\n'.join(lines)
      for prefix_lines in prefix_line_counts:
        if prefix_lines <= 0 or prefix_lines >= len(lines):
          continue
        if len(lines) - prefix_lines < min_target_lines:
          continue
        prompt = '\n'.join(lines[:prefix_lines]).strip()
        examples.append((sid, prompt, full_text, prefix_lines))

    return examples

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx):
    return self.examples[idx]

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    prompts = [example[1] for example in all_data]
    full_texts = [example[2] for example in all_data]
    prefix_lines = [example[3] for example in all_data]

    full_encoding = self.tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True)
    prompt_encoding = self.tokenizer(prompts, padding=False, truncation=True)

    token_ids = torch.LongTensor(full_encoding['input_ids'])
    attention_mask = torch.LongTensor(full_encoding['attention_mask'])
    loss_mask = attention_mask.clone()

    prompt_lengths = []
    for i, prompt_ids in enumerate(prompt_encoding['input_ids']):
      prompt_len = min(len(prompt_ids), token_ids.size(1))
      prompt_lengths.append(prompt_len)
      loss_mask[i, :prompt_len] = 0

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'loss_mask': loss_mask,
      'prompt_lengths': torch.LongTensor(prompt_lengths),
      'prefix_lines': torch.LongTensor(prefix_lines),
      'sent_ids': idx
    }

    return batched_data
