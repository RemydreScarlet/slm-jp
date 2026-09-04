from __future__ import annotations

import torch
import torch.nn as nn
from fla.layers import GatedDeltaNet as FlaNetsGatedDeltaNet

from smolm_jp4.config import SmolmJp4Config


class GDNLayer(nn.Module):
    """Gated DeltaNet layer wrapping fla's GatedDeltaNet.

    This is a thin wrapper that configures fla's GatedDeltaNet for our architecture.
    When use_gate=False, we use a simple pre-norm residual block with GDN.
    """

    def __init__(self, config: SmolmJp4Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Pre-norm
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # GDN core (from fla library)
        self.gdn = FlaNetsGatedDeltaNet(
            hidden_size=config.hidden_size,
            expand_v=config.gdn_expand_v,
            head_dim=config.gdn_head_dim,
            num_heads=config.gdn_num_heads,
            num_v_heads=config.gdn_num_v_heads,
            mode=config.gdn_mode,
            use_gate=config.gdn_use_gate,
            use_short_conv=True,
            allow_neg_eigval=False,
            conv_size=config.gdn_conv_size,
            conv_bias=False,
            layer_idx=layer_idx,
            norm_eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states, attn_weights, new_cache = self.gdn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        return residual + hidden_states, attn_weights, new_cache
