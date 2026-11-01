import copy
import os
import json
import re
from timeit import default_timer as timer
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lora_output = self.dropout(x @ self.lora_A) @ self.lora_B
        return lora_output * self.scaling


def add_lora_to_multihead_attention(attention_module, r=4, alpha=1.0, dropout=0.0):
    embed_dim = attention_module.embed_dim

    device = next(attention_module.parameters()).device
    dtype = next(attention_module.parameters()).dtype

    for param in attention_module.parameters():
        param.requires_grad = False

    attention_module.lora_q = LoRALayer(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout).to(device=device, dtype=dtype)
    attention_module.lora_k = LoRALayer(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout).to(device=device, dtype=dtype)
    attention_module.lora_v = LoRALayer(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout).to(device=device, dtype=dtype)
    attention_module.lora_out_proj = LoRALayer(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout).to(device=device, dtype=dtype)

    attention_module._original_forward = attention_module.forward

    def lora_forward(query, key, value, key_padding_mask=None, need_weights=True,
                    attn_mask=None, average_attn_weights=True, is_causal=False):

        is_batched = query.dim() == 3
        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        if attention_module.in_proj_weight is not None:

            w_q, w_k, w_v = attention_module.in_proj_weight.chunk(3, dim=0)

            if attention_module.in_proj_bias is not None:
                b_q, b_k, b_v = attention_module.in_proj_bias.chunk(3, dim=0)
            else:
                b_q = b_k = b_v = None

            q = F.linear(query, w_q, b_q)
            k = F.linear(key, w_k, b_k)
            v = F.linear(value, w_v, b_v)

            q = q + attention_module.lora_q(query)
            k = k + attention_module.lora_k(key)
            v = v + attention_module.lora_v(value)
        else:

            q = F.linear(query, attention_module.q_proj_weight, attention_module.in_proj_bias)
            k = F.linear(key, attention_module.k_proj_weight, None)
            v = F.linear(value, attention_module.v_proj_weight, None)

            q = q + attention_module.lora_q(query)
            k = k + attention_module.lora_k(key)
            v = v + attention_module.lora_v(value)

        tgt_len, bsz, embed_dim = q.shape
        src_len = k.shape[0]

        head_dim = embed_dim // attention_module.num_heads
        scaling = float(head_dim) ** -0.5

        q = q.contiguous().view(tgt_len, bsz * attention_module.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * attention_module.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * attention_module.num_heads, head_dim).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = attn_output_weights * scaling

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, attention_module.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
            attn_output_weights = attn_output_weights.view(bsz * attention_module.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=attention_module.dropout, training=attention_module.training)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn_output = F.linear(attn_output, attention_module.out_proj.weight, attention_module.out_proj.bias)
        attn_output = attn_output + attention_module.lora_out_proj(attn_output)

        if not is_batched:
            attn_output = attn_output.squeeze(1)
            if need_weights:
                attn_output_weights = attn_output_weights.squeeze(1)

        if need_weights:

            attn_output_weights = attn_output_weights.view(bsz, attention_module.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            if not is_batched:
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            return attn_output, None

    attention_module.forward = lora_forward

    return attention_module


def apply_lora_to_model(model, r=4, alpha=1.0, dropout=0.0):

    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')

    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            add_lora_to_multihead_attention(module, r=r, alpha=alpha, dropout=dropout)

        else:

            apply_lora_to_model(module, r=r, alpha=alpha, dropout=dropout)

    return model
