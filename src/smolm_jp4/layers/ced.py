from __future__ import annotations

import torch
import torch.nn as nn


class CEDKVProjection(nn.Module):
    """Causal Encoder-Decoder KV projection layer.

    In the CED architecture, decoder layers derive their KV cache from
    the encoder's final hidden state via layer-dependent projections,
    rather than computing KV from their own hidden states.

    Reference: DeepSeek-V4.1 Technical Report, Section 2.2

    For global attention layers in the decoder:
        C_l = H_{L/2} W_KV^l    (KV entries)
        Z_l = H_{L/2} W_Z^l     (compression weights)
    """

    def __init__(self, hidden_size: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # KV projection: hidden_size -> 2 * (num_kv_heads * head_dim)
        kv_dim = 2 * num_kv_heads * head_dim
        self.kv_proj = nn.Linear(hidden_size, kv_dim, bias=False)

        # Optional compression projection
        self.z_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, encoder_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_hidden: (batch, seq, hidden_size) - encoder's final hidden state

        Returns:
            k: (batch, seq, num_kv_heads, head_dim) - projected keys
            v: (batch, seq, num_kv_heads, head_dim) - projected values
        """
        kv = self.kv_proj(encoder_hidden)
        k, v = kv.chunk(2, dim=-1)

        batch_size, seq_len = encoder_hidden.shape[:2]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        return k, v


class CEDDecoderLayer(nn.Module):
    """Decoder layer that uses CED-style KV projection.

    This wraps an existing FullAttentionLayer to use encoder-derived KV
    instead of computing its own.
    """

    def __init__(self, attention_layer, kv_projection: CEDKVProjection) -> None:
        super().__init__()
        self.attention = attention_layer
        self.kv_proj = kv_projection

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask=None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        """Forward pass using encoder-derived KV."""
        batch_size, seq_len, _ = hidden_states.shape

        # Get KV from encoder hidden state
        k_new, v_new = self.kv_proj(encoder_hidden)

        # Apply RoPE to k
        from smolm_jp4.layers.gqa import apply_rotary_emb
        k_new = apply_rotary_emb(k_new, freqs_cis)

        # Get Q from current hidden states (using attention layer's norm and q_proj)
        residual = hidden_states
        normed = self.attention.norm(hidden_states)
        q = self.attention.q_proj(normed).view(
            batch_size, seq_len, self.attention.num_heads, self.attention.head_dim
        ).transpose(1, 2)
        q = apply_rotary_emb(q, freqs_cis)

        # Expand KV for GQA
        v = v_new.transpose(1, 2)
        k = k_new.transpose(1, 2)
        if self.attention.num_kv_groups > 1:
            k = k.repeat_interleave(self.attention.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.attention.num_kv_groups, dim=1)

        # Handle past_key_values (concatenate if present)
        if past_key_values is not None:
            past_k = past_key_values["key"]
            past_v = past_key_values["value"]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Scaled dot-product attention
        import math
        import torch.nn.functional as F
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attention.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_mask = attention_mask[:, None, None, :].to(dtype=attn_weights.dtype)
            else:
                attn_mask = attention_mask
            attn_weights = attn_weights + (1.0 - attn_mask) * torch.finfo(attn_weights.dtype).min

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        attn_weights = self.attention.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.attention.o_proj(attn_output)

        new_cache = None
        if use_cache:
            new_cache = {"key": k, "value": v}

        return residual + attn_output, attn_weights if output_attentions else None, new_cache
