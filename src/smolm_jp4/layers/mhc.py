from __future__ import annotations

import torch
import torch.nn as nn


class SinglePassMHCFactor(nn.Module):
    """Single-Pass Multi-Hyper-Connection (mHC).

    Maintains n residual streams between adjacent transformer blocks.
    The input-mixing coefficients A are shifted by one block to enable
    single-pass computation, halving activation memory traffic vs the
    original multi-pass mHC.

    Reference: DeepSeek-V4.1 Technical Report, Section 2.4.1

    The residual stream is updated as:
        X^{l+1} = B_l X^l + C_l F_l(A_{l-1} X^l)
        (A_l, B_l, C_l) = H(X^l)

    where:
        A_l: (batch, seq, n) - input mixing coefficients (shifted by 1 block)
        B_l: (batch, seq, n, n) - stream-to-stream mixing
        C_l: (batch, seq, n) - output mixing coefficients
        n: number of residual streams (default: 4)
    """

    def __init__(self, hidden_size: int, n_streams: int = 4) -> None:
        super().__init__()
        self.n_streams = n_streams
        self.hidden_size = hidden_size

        # Project normalized residual to predict coefficients
        # Using low-rank bottleneck for efficiency
        bottleneck = hidden_size // 4

        # A_l: 1 x n (input mixing, applied with 1-block delay)
        self.A_proj = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, n_streams, bias=False),
        )

        # B_l: n x n (stream-to-stream mixing)
        self.B_proj = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, n_streams * n_streams, bias=False),
        )

        # C_l: n x 1 (output mixing)
        self.C_proj = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, n_streams, bias=False),
        )

        # GatedNorm gate (elementwise self-gate on each stream)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, hidden_size, bias=False),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize projections near identity
        for proj in [self.A_proj, self.B_proj, self.C_proj]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
        # Initialize gate to output ~1
        # gate is Sequential(Linear(bias=False), SiLU, Linear(bias=False), Sigmoid)
        # gate[2] is the second Linear - initialize weights near zero so sigmoid outputs ~0.5
        nn.init.zeros_(self.gate[2].weight)

    def forward(
        self,
        x: torch.Tensor,
        prev_A: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq, hidden_size) - current hidden state
            prev_A: (batch, seq, n_streams) - mixing coefficients from previous block
                     If None (first block), uses uniform mixing.

        Returns:
            output: (batch, seq, hidden_size) - mixed output
            A: (batch, seq, n_streams) - current mixing coefficients
        """
        B, S, D = x.shape

        # Predict coefficients
        A = self.A_proj(x)  # (B, S, n)
        B_mat = self.B_proj(x).view(B, S, self.n_streams, self.n_streams)
        C = self.C_proj(x)  # (B, S, n)

        # Apply GatedNorm (elementwise self-gate)
        x_gated = x * self.gate(x)

        # Expand x to n streams: (B, S, 1, D) -> (B, S, n, D)
        x_expanded = x_gated.unsqueeze(2).expand(B, S, self.n_streams, D)

        # Single-Pass mixing: use prev_A (1-block delay) for input mixing
        if prev_A is None:
            # First block: use uniform mixing
            prev_A = torch.ones(B, S, self.n_streams, device=x.device, dtype=x.dtype) / self.n_streams

        # A_{l-1} X^l: weighted sum over streams using previous block's A
        # prev_A: (B, S, n) -> (B, S, n, 1)
        mixed_input = (prev_A.unsqueeze(-1) * x_expanded).sum(dim=2)  # (B, S, D)

        # B_l X^l: stream-to-stream mixing
        x_mixed = torch.einsum('bsnk,bskd->bsnd', B_mat, x_expanded)  # (B, S, n, D)

        # C_l F_l(A_{l-1} X^l): output mixing with the mixed input
        # C: (B, S, n) -> (B, S, n, 1)
        output = (C.unsqueeze(-1) * x_mixed).sum(dim=2)  # (B, S, D)

        return output, A


class SinglePassMHCWrapper(nn.Module):
    """Wrapper that applies Single-Pass mHC to any transformer block.

    This wraps the original residual connection, replacing the simple
    skip connection with the mHC multi-stream mechanism.
    """

    def __init__(self, hidden_size: int, n_streams: int = 4) -> None:
        super().__init__()
        self.mhc = SinglePassMHCFactor(hidden_size, n_streams)
        self.n_streams = n_streams

    def forward(
        self,
        x: torch.Tensor,
        sublayer_fn,
        prev_A: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq, hidden_size) - input hidden state
            sublayer_fn: callable that takes (x) and returns (output, attn_weights, cache)
            prev_A: (batch, seq, n_streams) - previous block's mixing coefficients

        Returns:
            output: (batch, seq, hidden_size) - output after mHC
            A: (batch, seq, n_streams) - current mixing coefficients
        """
        # Get mHC output (this becomes the new residual stream)
        mhc_out, A = self.mhc(x, prev_A)

        # Apply the sublayer to the mHC-mixed input
        sub_out, attn_weights, new_cache = sublayer_fn(mhc_out)

        return sub_out, A, attn_weights, new_cache
