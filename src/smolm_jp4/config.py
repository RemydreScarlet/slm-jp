from dataclasses import dataclass, field


@dataclass
class SmolmJp4Config:
    """Configuration for smolm-jp-4 model.

    Architecture: GDN (GatedDeltaNet) + GQA hybrid, with Single-Pass mHC,
    CED (Causal Encoder-Decoder), and CSA2-lite (cross-layer KV sharing).
    Dense model, no MoE.

    Key innovations from DeepSeek-V4.1:
    - Single-Pass mHC: Multi-stream residual connections for stability
    - CED: Split encoder/decoder to halve prefill computation
    - CSA2-lite: Cross-layer KV sharing for GQA layers
    """

    # --- Model size ---
    hidden_size: int = 1280
    num_hidden_layers: int = 28
    intermediate_size: int = 1920

    # --- Full-attention (GQA) ---
    num_attention_heads: int = 10
    num_key_value_heads: int = 2
    head_dim: int = 128  # hidden_size // num_attention_heads

    # --- GDN (GatedDeltaNet) ---
    gdn_head_dim: int = 128
    gdn_num_heads: int = 10
    gdn_num_v_heads: int = 10
    gdn_expand_v: float = 1.0
    gdn_use_gate: bool = False
    gdn_conv_size: int = 4
    gdn_mode: str = "chunk"

    # --- Block composition ---
    # GDN:full = 3:1 ratio. Full-attention at every 4th layer (0-indexed: 3,7,11,15,19).
    full_attention_interval: int = 4

    # --- Embedding ---
    vocab_size: int = 196_608
    max_position_embeddings: int = 32_768
    tie_word_embeddings: bool = True

    # --- Norm ---
    rms_norm_eps: float = 1e-5

    # --- Dropout ---
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # --- RoPE ---
    rope_theta: float = 10_000.0

    # --- Single-Pass mHC (Multi-Hyper-Connection) ---
    # From DeepSeek-V4.1: maintains n residual streams between blocks
    # Improves training stability and expressiveness
    use_mhc: bool = True
    mhc_num_streams: int = 4  # n=4 residual streams

    # --- CED (Causal Encoder-Decoder) ---
    # From DeepSeek-V4.1: split layers into encoder/decoder
    # Decoder KV is projected from encoder's final hidden state
    # Reduces prefill computation by ~50%
    use_ced: bool = True
    ced_split_ratio: float = 0.5  # fraction of layers used as encoder

    # --- CSA2-lite (Cross-Layer KV Sharing) ---
    # From DeepSeek-V4.1: share KV across GQA layers
    # Only applies to full-attention (GQA) layers
    use_csa2_lite: bool = True

    @property
    def num_gdn_layers(self) -> int:
        return self.num_hidden_layers - self.num_full_attn_layers

    @property
    def num_full_attn_layers(self) -> int:
        return sum(
            1 for i in range(self.num_hidden_layers)
            if i % self.full_attention_interval == self.full_attention_interval - 1
        )

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.full_attention_interval == self.full_attention_interval - 1

    @property
    def ced_split_point(self) -> int:
        """Layer index where encoder ends and decoder begins."""
        return int(self.num_hidden_layers * self.ced_split_ratio)

    def is_ced_encoder_layer(self, layer_idx: int) -> bool:
        """True if this layer is in the encoder portion of CED."""
        return layer_idx < self.ced_split_point

    def is_ced_decoder_layer(self, layer_idx: int) -> bool:
        """True if this layer is in the decoder portion of CED."""
        return layer_idx >= self.ced_split_point

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_key_value_heads <= self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0

        # GDN constraint: num_heads * head_dim == hidden_size (when use_gate=False)
        if not self.gdn_use_gate:
            assert self.gdn_num_heads * self.gdn_head_dim == self.hidden_size, (
                f"gdn_num_heads({self.gdn_num_heads}) * gdn_head_dim({self.gdn_head_dim}) "
                f"!= hidden_size({self.hidden_size})"
            )
        # GVA constraint
        if self.gdn_num_v_heads != self.gdn_num_heads:
            assert self.gdn_num_v_heads % self.gdn_num_heads == 0
