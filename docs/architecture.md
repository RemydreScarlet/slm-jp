# アーキテクチャ設計

smolm-jp-4 のアーキテクチャ詳細。Qwen3.8-Flash-Next の主要アイデアを 1B dense スケールに移植する。

## 設計方針

### 採用するコンポーネント

| コンポーネント | 状態 | 根拠 |
|---|---|---|
| GDN (Gated DeltaNet) + 通常 GQA ハイブリッド | **採用** | 論文 Tab.1 で GDN hybrid が 9 ベンチマーク中 8 で改善。4 層に 1 回が GQA、残り 3/4 が GDN |
| RoPE | **採用** (full-attention 層のみ) | NoPE variant は事後学習後の生成品質に悪影響 (無限生成しやすくなる) |
| MTP (Multi-Token Prediction) | **採用** (1層追加) | スペキュラティブデコーディング対応 |
| Gated Residual | **段階的導入** (保留) | §3.1 参照 |

### 削除したコンポーネント

| コンポーネント | 判断 | 理由 |
|---|---|---|
| n-gram embedding table | **削除** | 125B MoE では「300 tokens per active parameter」で管理される巨大記憶装置。1B dense ではテーブルだけで本体を上回り、パラメータ予算が破綻 |
| QSA / DSA (Sparse Attention) | **見送り** | 効果は主に 512K〜1M 長文脈で顕在化。32K〜64K ではメリット薄く、indexer 学習に Dense Warm-up ステージの追加コストに見合わない |
| MoE (ルーティング、複数エキスパート) | **削除 (dense 化)** | 当初の config.txt 自体が dense 化を明言 |

## パラメータ配分

```
vocab_size      = 196,608   # LLM-jp-4 トークナイザ (変更不可)
hidden_size     = 1,536
num_layers      = 28
num_heads       = 24
num_kv_heads    = 2         # GQA (head_dim=64)
head_dim        = 64

full_attn_interval = 4      # 28層中 7層が GQA、21層が GDN
linear_key_head_dim   = 192
linear_num_key_heads  = 8
linear_value_head_dim = 96
linear_num_value_heads = 16
linear_conv_kernel_dim = 4

intermediate_size = 3,072   # SwiGLU (hidden比 2.0倍)
tie_word_embeddings = true  # untied だと +302M で予算超過
```

### 概算パラメータ内訳 (GR 抜き)

| コンポーネント | 概算 | 比率 |
|---|---|---|
| トークン埋め込み (tied) | 302M | 30.8% |
| Full-attn 層 (GQA, 7層) | 36M | 3.6% |
| Linear-attn 層 (GDN, 21層) | 228M | 23.2% |
| FFN (全 28 層) | 396M | 40.4% |
| MTP | 19M | 2.0% |
| **合計** | **~981M** | **100%** |

> 注: 上記は Q/K/V/O 射影の行列形状から逆算した解析的概算値。norm・bias は含まれていない。実装後は必ず `model.num_parameters()` で実測すること。

### GR 追加時

Gated Residual (nr=4 フル版) 追加時は概算 +134M (合計 ~1.11B)。

## レイヤー構成

```
Layer 0:  GQA (full-attention + RoPE)
Layer 1:  GDN (linear-attention)
Layer 2:  GDN
Layer 3:  GDN
Layer 4:  GQA  ← 4層ごと
Layer 5:  GDN
Layer 6:  GDN
Layer 7:  GDN
...
Layer 24: GQA
Layer 25: GDN
Layer 26: GDN
Layer 27: GDN
```

判定式: `layer_idx % full_attn_interval == 0` なら GQA、それ以外は GDN。

## 各コンポーネント詳細

### GDN 層 (Linear Attention)

[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) の `fla.layers.GatedDeltaNet` を利用。

- **Delta Rule**: 過去の状態を GOP (Gated Outer Product) で更新
- **Conv カーネル**: `conv_kernel_dim=4` の局所情報圧縮
- **状態サイズ**: `num_key_heads × key_head_dim = 8 × 192 = 1,536` (= hidden_size)
- **計算复杂度**: O(N × d²) (通常の attention の O(N² × d) より高速)

```python
from fla.layers import GatedDeltaNet

gdn = GatedDeltaNet(
    hidden_size=1536,
    num_heads=16,             # value heads
    num_v_heads=16,
    key_dim=192,
    value_dim=96,
    conv_kernel_dim=4,
)
```

### GQA 層 (Full Attention)

通常の Multi-Head Attention with Grouped Query Attention。

- `num_heads=24`, `num_kv_heads=2` → GQA ratio = 12
- `head_dim=64`
- RoPE: `partial_rotary_factor=0.25`, `rope_theta=10_000_000`

### FFN (SwiGLU)

全 28 層に共通。

- `intermediate_size=3,072` (hidden の 2.0 倍)
- 活性化: SiLU (SwiGLU)

### MTP (Multi-Token Prediction)

1 層追加。 predictor ヘッドが 2 つ目のトークンを予測し、スペキュラティブデコーディングを支援。

```python
class SmolmJp4MTPLayer(nn.Module):
    def __init__(self, config):
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.prediction_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.mtp_num_hidden_layers)
        ])
```

## スケーリング参考値 (将来の 10B〜20B 版)

同じ設計比率 (vocab 固定、GDN:full=3:1、FFN 比 2.0 倍) でのラフな目安:

```
hidden=2560, layers=32  -> ~2.8B
hidden=3584, layers=36  -> ~5.8B
hidden=4096, layers=40  -> ~8.2B
```

10B〜20B 帯は hidden=4096〜5000 台、layers=40〜48 あたりが目安。**この規模になった時点で QSA/DSA 導入と Muon optimizer の本格採用を再検討する。**

## Gated Residual (GR) の段階的導入方針

論文の GR は、残差ストリームを nr=4 本に拡張し、read/write/GatedNorm を各層に挿入する機構。

| ステップ | 内容 | 目的 |
|---|---|---|
| 1 | GR なし・AdamW でベースライン学習 | パイプライン動作確認 |
| 2 | GatedNorm のみ追加 (残差拡張なし) | 安定性への効果を実測 |
| 3 | nr=4 のフル GR へ拡張 | 不安定性が問題になる場合のみ |

> ステップ 1 のベースライン学習が回った時点で、ステップ 2 の効果を実測して判断する。

---

## 350M モデルバリアント

ローカル GPU (RTX 2070 Super 8GB) で学習可能な 350M モデル。

### パラメータ配分 (実測 ~372M)

```
vocab_size      = 196,608   # LLM-jp-4 トークナイザ (固定)
hidden_size     = 1,024
num_layers      = 20
intermediate    = 1,536

# Full Attention (GQA)
num_heads       = 8
num_kv_heads    = 2
head_dim        = 128

# GDN
gdn_num_heads   = 8
gdn_num_v_heads = 8
gdn_head_dim    = 128
gdn_expand_v    = 1.0
gdn_use_gate    = False

# Block ratio
full_attention_interval = 4  # 20層中5層がGQA、15層がGDN
tie_word_embeddings     = True  # 必須
```

### 1B モデルとの比較

| 項目 | 1B | 350M |
|---|---|---|
| hidden_size | 1536 | 1024 |
| layers | 28 | 20 |
| heads | 24 | 8 |
| kv_heads | 2 | 2 |
| FFN intermediate | 3072 | 1536 |
| GDN heads | 24 | 8 |
| GDN v_heads | 48 | 8 |
| expand_v | 2.0 | 1.0 |
| 合計パラメータ | ~981M | ~372M |

### VRAM 予測 (RTX 2070 Super)

| 条件 | VRAM |
|---|---|
| Forward のみ | ~1.8 GB |
| Forward + Backward (seq=128) | ~2.2 GB |
| Forward + Backward (seq=2048, grad ckpt) | ~4-5 GB |
| AdamW 含む合計 | ~5.6 GB |

### 深刻な制約: vocab_size=196,608

LLM-jp-4 トークナイザの語彙数は 196,608。embedding 表は 196,608 × 1,024 × 2 bytes = **384 MB** (FP16)。
全パラメータ 372M のうち **54%** が embedding に消費される。

これは将来的に vocab を小さくする (例: 32K) か、SentencePiece の Unigram モードで
語彙を削減する検討も可能な余地がある。
