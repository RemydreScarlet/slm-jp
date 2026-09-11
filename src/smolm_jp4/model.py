from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from smolm_jp4.config import SmolmJp4Config
from smolm_jp4.layers.gdn import GDNLayer
from smolm_jp4.layers.gqa import FullAttentionLayer, TransformerBlock, precompute_freqs_cis


class SmolmJp4Model(nn.Module):
    """Base transformer model for smolm-jp-4.

    Supports:
    - GDN + GQA hybrid attention
    - Single-Pass mHC (Multi-Hyper-Connection)
    - CED (Causal Encoder-Decoder) architecture
    - CSA2-lite (cross-layer KV sharing for GQA layers)
    """

    def __init__(self, config: SmolmJp4Config) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.head_dim, config.max_position_embeddings, config.rope_theta),
            persistent=False,
        )

        self.gradient_checkpointing = False

        # CED: KV projections for decoder layers
        if config.use_ced:
            from smolm_jp4.layers.ced import CEDKVProjection
            self.ced_kv_projections = nn.ModuleList()
            for i in range(config.num_hidden_layers):
                if config.is_ced_decoder_layer(i) and config.is_full_attention_layer(i):
                    # Decoder GQA layers need KV projection from encoder
                    proj = CEDKVProjection(
                        hidden_size=config.hidden_size,
                        num_kv_heads=config.num_key_value_heads,
                        head_dim=config.head_dim,
                    )
                    self.ced_kv_projections.append(proj)
                else:
                    self.ced_kv_projections.append(None)

        # CSA2-lite: shared KV cache for GQA layers
        if config.use_csa2_lite:
            # Track which GQA layers share KV
            self._csa2_primary_layer = None
            for i in range(config.num_hidden_layers):
                if config.is_full_attention_layer(i):
                    self._csa2_primary_layer = i
                    break
            self._csa2_shared_kv = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small standard deviation for residual layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> dict:
        hidden_states = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis[:input_ids.shape[1]]

        all_hidden_states = []
        all_attn_weights = []
        all_caches = [] if use_cache else None

        # mHC state tracking
        prev_mhc_A = None
        encoder_hidden = None  # For CED

        # CSA2-lite state
        csa2_shared_kv = None

        for layer_idx, layer in enumerate(self.layers):
            # CED: store encoder hidden state
            if self.config.use_ced and self.config.is_ced_encoder_layer(layer_idx):
                encoder_hidden = hidden_states
                # For the last encoder layer, also use it for CED decoder KV
                if self.config.is_ced_encoder_layer(layer_idx):
                    # Store for potential use by decoder
                    pass

            # CSA2-lite: check if we should use shared KV
            layer_past = None
            if use_cache and self.config.use_csa2_lite:
                if past_key_values:
                    layer_past = past_key_values[layer_idx]
                # For non-primary GQA layers, use shared KV
                if (self.config.is_full_attention_layer(layer_idx) and
                    layer_idx != self._csa2_primary_layer and
                    csa2_shared_kv is not None):
                    layer_past = csa2_shared_kv

            if self.gradient_checkpointing and self.training:
                result = checkpoint(
                    layer,
                    hidden_states,
                    freqs_cis,
                    attention_mask,
                    layer_past,
                    False,  # use_cache
                    False,  # output_attentions
                    prev_mhc_A,
                    use_reentrant=False,
                )
                hidden_states, attn_weights, new_cache, mhc_A = result
            else:
                hidden_states, attn_weights, new_cache, mhc_A = layer(
                    hidden_states,
                    freqs_cis=freqs_cis,
                    attention_mask=attention_mask,
                    past_key_values=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    prev_mhc_A=prev_mhc_A,
                    **kwargs,
                )

            # Update mHC state
            prev_mhc_A = mhc_A

            # CSA2-lite: update shared KV from primary layer
            if (self.config.use_csa2_lite and
                self.config.is_full_attention_layer(layer_idx) and
                layer_idx == self._csa2_primary_layer and
                new_cache is not None):
                csa2_shared_kv = new_cache

            if output_attentions:
                all_attn_weights.append(attn_weights)
            if use_cache:
                all_caches.append(new_cache)

        hidden_states = self.norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states if output_attentions else None,
            "attentions": all_attn_weights if output_attentions else None,
            "past_key_values": all_caches if use_cache else None,
        }


class SmolmJp4ForCausalLM(nn.Module):
    """Causal language model head for smolm-jp-4."""

    def __init__(self, config: SmolmJp4Config) -> None:
        super().__init__()
        self.config = config
        self.model = SmolmJp4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "attentions": outputs["attentions"],
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=512, temperature=0.8, top_p=0.95, **kwargs):
        """Simple greedy/temperature generation."""
        for _ in range(max_new_tokens):
            # Truncate to max_position_embeddings
            idx_cond = input_ids[:, -self.config.max_position_embeddings:]
            outputs = self(idx_cond, use_cache=True, **kwargs)
            logits = outputs["logits"][:, -1, :] / temperature

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids
