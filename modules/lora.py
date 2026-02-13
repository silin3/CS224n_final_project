"""
LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

LoRA decomposes the weight update as: W' = W + (alpha/r) * B @ A
where A in R^{r x in_features}, B in R^{out_features x r}, r << min(in_features, out_features).
"""

import math
import torch
from torch import nn


class LoRALinear(nn.Module):
  """
  Wraps an nn.Linear layer with LoRA adaptation.
  Forward: output = linear(x) + (alpha / r) * (x @ A.T @ B.T)
  Original linear weights are frozen; only A and B are trained.
  """

  def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = None):
    super().__init__()
    self.linear = linear
    self.r = r
    self.alpha = alpha if alpha is not None else r

    in_features = linear.in_features
    out_features = linear.out_features

    # LoRA matrices: A (r, in_features), B (out_features, r)
    # Delta W = B @ A has shape (out_features, in_features)
    self.lora_A = nn.Parameter(torch.empty(r, in_features))
    self.lora_B = nn.Parameter(torch.empty(out_features, r))

    self._init_lora()
    self._freeze_original()

  def _init_lora(self):
    # A: Kaiming uniform
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    # B: zeros so that initial delta is zero
    nn.init.zeros_(self.lora_B)

  def _freeze_original(self):
    self.linear.weight.requires_grad = False
    if self.linear.bias is not None:
      self.linear.bias.requires_grad = False

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Base output
    out = self.linear(x)
    # LoRA: (alpha/r) * x @ A.T @ B.T
    # x: (batch, seq, in_features)
    # A.T: (in_features, r), x @ A.T: (batch, seq, r)
    # B.T: (r, out_features), (x @ A.T) @ B.T: (batch, seq, out_features)
    scale = self.alpha / self.r
    lora_out = (x @ self.lora_A.T @ self.lora_B.T) * scale
    return out + lora_out


def apply_lora_to_gpt2(gpt, lora_mode, lora_r=8, lora_alpha=None):
  """
  Apply LoRA to GPT-2 layers based on the experiment mode.

  Args:
    gpt: GPT2Model instance (with pretrained weights loaded)
    lora_mode: 'qv' | 'all_attn' | 'attn_mlp'
    lora_r: LoRA rank
    lora_alpha: LoRA scaling factor (default: lora_r)
  """
  num_layers = len(gpt.gpt_layers)
  for i in range(num_layers):
    layer = gpt.gpt_layers[i]
    attn = layer.self_attention

    if lora_mode == 'qv':
      # LoRA-QV: query and value only
      layer.self_attention.query = LoRALinear(attn.query, r=lora_r, alpha=lora_alpha)
      layer.self_attention.key = attn.key  # keep original
      layer.self_attention.value = LoRALinear(attn.value, r=lora_r, alpha=lora_alpha)
    elif lora_mode == 'all_attn':
      # LoRA-AllAttn: Q, K, V, O
      layer.self_attention.query = LoRALinear(attn.query, r=lora_r, alpha=lora_alpha)
      layer.self_attention.key = LoRALinear(attn.key, r=lora_r, alpha=lora_alpha)
      layer.self_attention.value = LoRALinear(attn.value, r=lora_r, alpha=lora_alpha)
      layer.attention_dense = LoRALinear(layer.attention_dense, r=lora_r, alpha=lora_alpha)
    elif lora_mode == 'attn_mlp':
      # LoRA-Attn+MLP: Q, K, V, O + MLP (interm_dense, out_dense)
      layer.self_attention.query = LoRALinear(attn.query, r=lora_r, alpha=lora_alpha)
      layer.self_attention.key = LoRALinear(attn.key, r=lora_r, alpha=lora_alpha)
      layer.self_attention.value = LoRALinear(attn.value, r=lora_r, alpha=lora_alpha)
      layer.attention_dense = LoRALinear(layer.attention_dense, r=lora_r, alpha=lora_alpha)
      layer.interm_dense = LoRALinear(layer.interm_dense, r=lora_r, alpha=lora_alpha)
      layer.out_dense = LoRALinear(layer.out_dense, r=lora_r, alpha=lora_alpha)
    else:
      raise ValueError(f"Unknown lora_mode: {lora_mode}")
