from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F

from smolm_jp4.config import SmolmJp4Config


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10_000.0) -> torch.Tensor:
    """Precompute complex exponentials for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.to(x.device)
    x_rot = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rot.type_as(x)


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: SmolmJp4Config) -> None:
        super().__init__()
        self.w_gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w_up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w_down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class FullAttentionLayer(nn.Module):
    """Standard full-attention layer with GQA and RoPE."""

    def __init__(self, config: SmolmJp4Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # attention_mask: (batch, seq_len) with 1 for valid, 0 for padding
            # Expand to (batch, 1, 1, seq_len) for broadcasting
            if attention_mask.dim() == 2:
                attn_mask = attention_mask[:, None, None, :].to(dtype=attn_weights.dtype)
            else:
                attn_mask = attention_mask
            attn_weights = attn_weights + (1.0 - attn_mask) * torch.finfo(attn_weights.dtype).min

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        # Cache for inference
        new_cache = None
        if use_cache:
            new_cache = {"key": k, "value": v}

        return residual + attn_output, attn_weights if output_attentions else None, new_cache


class TransformerBlock(nn.Module):
    """A single transformer block: attention/GDN + FFN with pre-norm residuals.

    Supports optional Single-Pass mHC (Multi-Hyper-Connection) for improved
    training stability and expressiveness.
    """

    def __init__(self, config: SmolmJp4Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full_attention = config.is_full_attention_layer(layer_idx)
        self.use_mhc = config.use_mhc

        if self.is_full_attention:
            self.attn_or_gdn = FullAttentionLayer(config, layer_idx)
        else:
            from smolm_jp4.layers.gdn import GDNLayer
            self.attn_or_gdn = GDNLayer(config, layer_idx)

        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config)

        # Single-Pass mHC (optional)
        if self.use_mhc:
            from smolm_jp4.layers.mhc import SinglePassMHCFactor
            self.mhc = SinglePassMHCFactor(config.hidden_size, config.mhc_num_streams)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        prev_mhc_A: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None, torch.Tensor | None]:
        """Forward pass with optional mHC.

        Returns:
            hidden_states: output hidden states
            attn_weights: attention weights (if requested)
            new_cache: KV cache (if requested)
            mhc_A: mHC mixing coefficients (if mHC enabled)
        """
        mhc_A = None

        # Attention/GDN sublayer
        if self.is_full_attention:
            hidden_states, attn_weights, new_cache = self.attn_or_gdn(
                hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
        else:
            hidden_states, attn_weights, new_cache = self.attn_or_gdn(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

        # FFN sublayer
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        # Apply mHC after FFN (modifies the residual stream for next block)
        if self.use_mhc:
            mhc_out, mhc_A = self.mhc(hidden_states, prev_mhc_A)
            hidden_states = mhc_out

        return hidden_states, attn_weights, new_cache, mhc_A
