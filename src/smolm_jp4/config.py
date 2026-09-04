from dataclasses import dataclass


@dataclass
class SmolmJp4Config:
    """Configuration for smolm-jp-4 maximal 8GB model.

    Architecture: GDN (GatedDeltaNet) + GQA hybrid, no GR, no MoE.
    Tuned to maximally utilize RTX 2070 Super (8GB) with gradient checkpointing.
    624M params, 7.1GB peak at seq=2048 with AdamW (bf16).

    NOTE: vocab_size=196,608 (LLM-jp-4 tokenizer) is fixed and dominates the
    parameter budget (~252M for embedding alone).
    Sweep result: absolute max is 1408/24/2112 (661M, 7.47GB) but leaves
    only 0.3GB headroom. This config (1280/28/1920) leaves ~0.7GB margin.
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
