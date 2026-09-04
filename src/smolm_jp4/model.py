from __future__ import annotations

import torch
import torch.nn as nn

from smolm_jp4.config import SmolmJp4Config
from smolm_jp4.layers.gdn import GDNLayer
from smolm_jp4.layers.gqa import FullAttentionLayer, TransformerBlock, precompute_freqs_cis


class SmolmJp4Model(nn.Module):
    """Base transformer model for smolm-jp-4."""

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

        for layer in self.layers:
            if use_cache:
                past = past_key_values[layer.layer_idx] if past_key_values else None
            else:
                past = None

            hidden_states, attn_weights, new_cache = layer(
                hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

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
