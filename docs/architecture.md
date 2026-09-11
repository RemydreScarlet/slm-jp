# アーキテクチャ設計

smolm-jp-4 のアーキテクチャ詳細。Qwen3.8-Flash-Next の主要アイデアを 1B dense スケールに移植する。

## 設計方針

### 採用するコンポーネント

| コンポーネント | 状態 | 根拠 |
|---|---|---|
| GDN (Gated DeltaNet) + 通常 GQA ハイブリッド | **採用** | 論文 Tab.1 で GDN hybrid が 9 ベンチマーク中 8 で改善。4 層に 1 回が GQA、残り 3/4 が GDN |
| RoPE | **採用** (full-attention 層のみ) | NoPE variant は事後学習後の生成品質に悪影響 (無限生成しやすくなる) |
| MTP (Multi-Token Prediction) | **採用** (1層追加) | スペキュラティブデコーディング対応 |
| **Single-Pass mHC** | **採用** | DeepSeek-V4.1: 残差ストリームを n=4 本に拡張。学習安定性+3.75pt改善 (Qwen3.8 Tab.5) |
| **CED (Causal Encoder-Decoder)** | **採用** | DeepSeek-V4.1: prefill 計算を約半分に削減。品質は据え置き |
| **CSA2-lite** | **採用** | DeepSeek-V4.1: GQA層間で KV キャッシュを共有 |

### 削除したコンポーネント

| コンポーネント | 判断 | 理由 |
|---|---|---|
| n-gram embedding table (Engram) | **見送り** | 1B ではパラメータ予算が厳しい (128M追加で埋め込み比率が44%に)。10B移行時に再検討 |
| QSA / DSA (Sparse Attention) | **見送り** | 効果は主に 512K〜1M 長文脈で顕在化。32K〜64K ではメリット薄く |
| MoE (ルーティング、複数エキスパート) | **削除 (dense 化)** | 当初の config.txt 自体が dense 化を明言 |
| Gated Residual (GR) | **mHC で代替** | Single-Pass mHC が GR の機能をカバー。+3.75pt 改善実績あり |

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

## DeepSeek-V4.1 由来の新機能

### Single-Pass mHC (Multi-Hyper-Connection)

**目的**: 残差ストリームを n=4 本に拡張し、学習安定性と表現力向上

**仕組み**:
```
X^{l+1} = B_l X^l + C_l F_l(A_{l-1} X^l)
(A_l, B_l, C_l) = H(X^l)
```

- A_l: (batch, seq, 4) - 入力混合係数（1ブロック遅延）
- B_l: (batch, seq, 4, 4) - ストリーム間混合
- C_l: (batch, seq, 4) - 出力混合
- GatedNorm: 要素ごとセルフゲート

**パラメータ追加**: ~57M (8.8%)
**効果**: Qwen3.8 で +3.75pt 平均スコア改善

### CED (Causal Encoder-Decoder)

**目的**: prefill 計算量を約半分に削減

**仕組み**:
- 28層を 14:14 に分割
- エンコーダ (層0-13): 通常のトランスフォーマー計算
- デコーダ (層14-27): グローバルKVをエンコーダ最終隠れ状態から射影
  - C_l = H_{L/2} W_KV^l
  - Z_l = H_{L/2} W_Z^l

**パラメータ追加**: ~32M (4.9%)
**利点**: エージェント用途（頻繁なツール呼び出し）でprefillコスト削減

### CSA2-lite (Cross-Layer KV Sharing)

**目的**: GQA層間でKVキャッシュを共有

**仕組み**:
- 7層のGQA層でKVを共有（層0がPrimary、残り6層がReuse）
- メモリ効率向上

**パラメータ追加**: なし
**制限**: GDN層には適用不可

## Gated Residual (GR) の扱い

**結論**: Single-Pass mHC で代替

GR は 1B スケールでは activation メモリ 4 倍増がボトルネックになる可能性があったが、
Single-Pass mHC は同等の改善効果（+3.75pt）をより効率的に実現する。

| 比較項目 | GR (nr=4) | Single-Pass mHC |
|---|---|---|
| パラメータ追加 | +134M | +57M |
| アクティベーション増 | 4倍 | 4倍 (Mega-mHC で2倍に削減可) |
| 実装の複雑さ | 高 | 中 |
| 安定性効果 | 大 | 大 |

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
