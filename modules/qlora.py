import math
import torch
import torch.nn.functional as F
from torch import nn


class QLoRALinear(nn.Module):
  """
  Wraps an nn.Linear layer with:
    1) frozen base weights,
    2) fake/group-wise low-bit quantization in forward,
    3) trainable LoRA adapters.

  Forward:
    output = linear_q(x) + (alpha / r) * (x @ A.T @ B.T)

  Notes:
  - Base weight is frozen.
  - Quantization is simulated on-the-fly from frozen fp weights.
  - Only lora_A and lora_B are trainable.
  """

  def __init__(
      self,
      linear: nn.Linear,
      r: int = 8,
      alpha: float = None,
      n_bits: int = 4,
      group_size: int = 64,
      quantize_bias: bool = False,
  ):
    super().__init__()
    if not isinstance(linear, nn.Linear):
      raise TypeError(f"QLoRALinear expects nn.Linear, got {type(linear)}")

    if r <= 0:
      raise ValueError(f"r must be positive, got {r}")
    if n_bits < 2 or n_bits > 8:
      raise ValueError(f"n_bits should usually be in [2, 8], got {n_bits}")
    if group_size <= 0:
      raise ValueError(f"group_size must be positive, got {group_size}")

    self.linear = linear
    self.r = r
    self.alpha = alpha if alpha is not None else r
    self.n_bits = n_bits
    self.group_size = group_size
    self.quantize_bias = quantize_bias

    in_features = linear.in_features
    out_features = linear.out_features

    # LoRA matrices: A (r, in_features), B (out_features, r)
    self.lora_A = nn.Parameter(torch.empty(r, in_features))
    self.lora_B = nn.Parameter(torch.empty(out_features, r))

    self._init_lora()
    self._freeze_original()

  def _init_lora(self):
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B)

  def _freeze_original(self):
    self.linear.weight.requires_grad = False
    if self.linear.bias is not None:
      self.linear.bias.requires_grad = False

  def _groupwise_quantize_dequantize(self, weight: torch.Tensor) -> torch.Tensor:
    """
    Simulate symmetric group-wise quantization and dequantization.

    Args:
      weight: (out_features, in_features)

    Returns:
      dequantized weight tensor with same shape as input.
    """
    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype

    # Pad last dimension so it is divisible by group_size
    pad_len = (self.group_size - (in_features % self.group_size)) % self.group_size
    if pad_len > 0:
      padded = F.pad(weight, (0, pad_len), mode="constant", value=0.0)
    else:
      padded = weight

    num_groups = padded.shape[1] // self.group_size
    w = padded.view(out_features, num_groups, self.group_size)

    qmax = (1 << (self.n_bits - 1)) - 1
    eps = 1e-8

    absmax = w.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    scale = absmax / qmax

    q = torch.round(w / scale).clamp(-qmax, qmax)
    w_deq = q * scale

    w_deq = w_deq.view(out_features, num_groups * self.group_size)
    if pad_len > 0:
      w_deq = w_deq[:, :in_features]

    return w_deq.to(device=device, dtype=dtype)

  def _quantized_weight(self) -> torch.Tensor:
    with torch.no_grad():
      return self._groupwise_quantize_dequantize(self.linear.weight)

  def _quantized_bias(self):
    if self.linear.bias is None:
      return None
    if not self.quantize_bias:
      return self.linear.bias

    with torch.no_grad():
      b = self.linear.bias
      qmax = (1 << (self.n_bits - 1)) - 1
      eps = 1e-8
      scale = b.abs().amax().clamp_min(eps) / qmax
      q = torch.round(b / scale).clamp(-qmax, qmax)
      return q * scale

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: (..., in_features)
    returns: (..., out_features)
    """
    q_weight = self._quantized_weight()
    q_bias = self._quantized_bias()
    base_out = F.linear(x, q_weight, q_bias)

    scale = self.alpha / self.r
    lora_out = (x @ self.lora_A.T @ self.lora_B.T) * scale

    return base_out + lora_out


def apply_qlora_to_gpt2(
    gpt,
    qlora_mode,
    qlora_r=8,
    qlora_alpha=None,
    n_bits=4,
    group_size=64,
    quantize_bias=False,
):
  """
  Apply QLoRA-style adapters to GPT-2 layers.

  Args:
    gpt: GPT2Model instance
    qlora_mode: 'qv' | 'all_attn' | 'attn_mlp'
    qlora_r: LoRA rank
    qlora_alpha: LoRA scaling factor (default: qlora_r)
    n_bits: fake quantization bit width
    group_size: group-wise quantization group size
    quantize_bias: whether to fake-quantize bias
  """
  num_layers = len(gpt.gpt_layers)

  def wrap(linear_module):
    return QLoRALinear(
      linear_module,
      r=qlora_r,
      alpha=qlora_alpha,
      n_bits=n_bits,
      group_size=group_size,
      quantize_bias=quantize_bias,
    )

  for i in range(num_layers):
    layer = gpt.gpt_layers[i]
    attn = layer.self_attention

    if qlora_mode == 'qv':
      layer.self_attention.query = wrap(attn.query)
      layer.self_attention.key = attn.key
      layer.self_attention.value = wrap(attn.value)

    elif qlora_mode == 'all_attn':
      layer.self_attention.query = wrap(attn.query)
      layer.self_attention.key = wrap(attn.key)
      layer.self_attention.value = wrap(attn.value)
      layer.attention_dense = wrap(layer.attention_dense)

    elif qlora_mode == 'attn_mlp':
      layer.self_attention.query = wrap(attn.query)
      layer.self_attention.key = wrap(attn.key)
      layer.self_attention.value = wrap(attn.value)
      layer.attention_dense = wrap(layer.attention_dense)
      layer.interm_dense = wrap(layer.interm_dense)
      layer.out_dense = wrap(layer.out_dense)

    else:
      raise ValueError(f"Unknown qlora_mode: {qlora_mode}")