# 学習パイプライン

smolm-jp-4 の事前学習に使用するコーパス、ハイパーパラメータ、インフラ構成。

## 使用コーパス: LLM-jp Corpus v4.1

### 概要

LLM-jp-4 モデルシリーズの事前学習に使用されたコーパス。日本語・英語・コードから構成される。

- **配布元**: [NII GitLab](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4.1)
- **GitHub ミラー**: [brunoleomenezes/llm-jp-corpus-v4.1](https://github.com/brunoleomenezes/llm-jp-corpus-v4.1)
- **LLM-jp-4 での使用**: LLM-jp-4 全モデル (8B, 33B, 32B-A3B) の事前学習に使用

### ディレクトリ構成 (v4.1)

v4.1 は v4 から以下のサブコーパスが追加・修正されたもの。

```
llm-jp-corpus-v4.1/
├── en (18T tokens)
│   ├── en_fineweb-rescored (18T)
│   │   └── Qwen3-32B (Apache 2.0) で教育スコアを再付与
│   │       (元の Llama-3-70B 版は OSS ライセンスで公開不可のため)
│   └── en_megamath-web-pro-max-oss (67B)
│       └── gpt-oss-120b (Apache 2.0) で数学関連度スコア付与 + パラフレーズ
└── code (720B tokens)
    └── code_stack-v2 (720B)
        └── the Stack v2 (リポジトリごとのライセンス)
```

> トークン数は llm-jp-tokenizer v4 でカウント。

### v4 からの変更点 (v4.1 で追加されたもの)

| サブコーパス | トークン数 | 変更内容 |
|---|---|---|
| `en_fineweb-rescored` | 18T | Llama-3 依存を解消。Qwen3-32B でスコア再付与 |
| `en_megamath-web-pro-max-oss` | 67B | 新規追加。gpt-oss-120b で数学テキストを選別・パラフレーズ |
| `code_stack-v2` | 720B | the Stack v1 → v2 に更新 |

### v4 に含まれていたサブコーパス (使用しないもの)

LLM-jp-4 開発時に以下は使用されなかった (v4.1 にも含まれない):

| サブコーパス | 理由 |
|---|---|
| `en_fineweb` (元版) | Llama-3-70B-Instruct 依存 → OSS ライセンス公開不可 |
| `en_finemath` | Llama-3.1-70B-Instruct 依存 → OSS ライセンス公開不可 |
| `ja_sip_comprehensive_pdf/surya` | テキスト品質が不十分 |

### 全体構成 (v4 基準 + v4.1 追加分)

```
Total: ~19.5T tokens (v4) + ~18.7T tokens (v4.1 追加) = ~38T tokens

ja (688B)
├── ja_cc (223B)           # Common Crawl 日本語 (Uzushio フィルタ)
├── ja_fineweb-2 (236B)    # FineWeb-2 日本語部分
├── ja_patent (68B)        # 特許公報
├── ja_warp_pdf (58B)      # 国立国会図書館 WARP
├── ja_nwc2010 (16B)       # 日本語ウェブコーパス 2010
├── ja_sip (41B)           # SIP コーパス
├── ja_kaken (0.9B)        # 科研費
├── ja_kokkai_giji (0.8B)  # 国会会議録
├── ja_warp_html (0.8B)
├── ja_wiki (1B)           # Wikipedia
├── ja_aozorabunko (0.1B)  # 青空文庫
└── ja_e-gov (0.08B)

en (17.8T + 18T = ~35.8T)
├── en_fineweb (17.6T)       # v4 元版 (Llama-3 依存)
├── en_fineweb-rescored (18T) # v4.1 追加 (Qwen3 依存、OSS OK)
├── en_dolma (155B)           # books, pes2o, reddit, wiki
├── en_olmo (49B)             # algebraicstack, arxiv, openwebmath
├── en_finemath (10B)         # 数学 (Llama-3.1 依存)
├── en_mathpile (9B)
├── en_wiki (5B)
├── en_dolmino (1.5B)         # stackexchange
├── en_megamath-web-pro-max-oss (67B) # v4.1 追加
└── en_gsm8k (0.003B)

code (218B + 720B = ~938B)
├── code_stack (114B)         # the Stack v1
├── code_olmo-starcoder (104B)
└── code_stack-v2 (720B)      # v4.1 追加

ko (52B) / zh (789B)          # Wikipedia + FineWeb-2
```

### ライセンス

各サブコーパスのライセンスは [LLM-jp Corpus v4 README](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4/-/blob/main/README-ja.md) 参照。主に:

- 日本語サブコーパス: CC BY 4.0
- 英語 FineWeb: ODC-BY (Common Crawl terms of use 従う)
- コード: リポジトリごとのライセンス

> 本データセットは日本国内のサーバから配布。日本国外のサーバから再配布する場合は日本国著作権法が適用されない点に注意。

### コーパスの取得

```bash
# Git LFS が必要
git lfs install

# v4.1 のクローン (必要な部分のみ)
git clone --depth 1 https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4.1.git

# または必要なサブコーパスのみ取得
# (全量は 33TB+ なので、必要なものだけ選択)
```

## トークナイザ

LLM-jp-4 トークナイザ (llm-jp-tokenizer v4) を使用。

- **vocab_size**: 196,608
- **取得**: `AutoTokenizer.from_pretrained("llm-jp/llm-jp-4-8b-base")`
- **タイプ**: SentencePiece (Unigram モード) ベース
- **特徴**: 日本語・中国語・韓国語を Mistral から拡張した語彙

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-4-8b-base")
print(f"vocab_size: {tokenizer.vocab_size}")  # 196608
```

## トレーニングハイパーパラメータ

### Phase 1: AdamW ベースライン (GR なし)

| パラメータ | 値 | 備考 |
|---|---|---|
| optimizer | AdamW | Muon は見送り (§3.2) |
| learning_rate | 3e-4 | 1B スケールの標準値 |
| lr_scheduler | cosine with warmup | |
| warmup_steps | 2,000 | |
| max_steps | ~100,000 | 20B〜40B tokens / batch_size |
| batch_size | GPU 数に応じて調整 | FSDP で分散 |
| gradient_accumulation_steps | 目標 batch size に合わせて | |
| max_grad_norm | 1.0 | |
| weight_decay | 0.1 | |
| dtype | bfloat16 | |
| attention_dropout | 0.0 | |

### Phase 2: GR 導入後

Phase 1 の結果をベースに、GR の有無で比較実験。

### 学習データの準備

```python
# jsonl.gz ファイルからデータを読み込む例
import gzip
import json

def load_corpus(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            yield data["text"]

# データローダーの例 (packing を前提)
# 実際は datasets ライブラリや独自の DataPacker を使用
```

## インフラ構成

### 並列化戦略

| 方式 | 採用 | 理由 |
|---|---|---|
| FSDP (ZeRO Stage 2 相当) | **採用** | 1B dense の混合精度 Adam 状態は ~16GB。数枚 GPU で収まる |
| TP (Tensor Parallel) | 不要 | モデルサイズが小さい |
| PP (Pipeline Parallel) | 不要 | レイヤー数が少ない |
| DP (Data Parallel) | FSDP に内蔵 | |

### 推奨環境

```
GPU: 4〜8 × A100 80GB or H100 80GB
RAM: 256GB+
ストレージ: 500GB+ (コーパスのサブセット用)
```

### フレームワーク

```python
# pyproject.toml の依存関係
[project]
dependencies = [
    "torch>=2.4",
    "transformers>=4.45",
    "fla>=0.1",            # flash-linear-attention (GDN カーネル)
    "datasets",
    "accelerate",
    "wandb",
]
```

### FSDP 設定例

```python
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO Stage 2 相当
    mixed_precision_policy="bf16",
)
```

## 評価

### ベンチマーク

| ベンチマーク | 言語 | 用途 |
|---|---|---|
| llm-jp-eval | 日本語 | 日本語 LLM 標準ベンチマーク (JAQMC, JSQuAD, etc.) |
| LM Evaluation Harness | 英語 | Hellaswag, ARC, MMLU |
| 自前テスト | 日本語 | 32K/64K シーケンスでの loss 監視 |

### 評価スクリプト

```bash
# llm-jp-eval
python -m llm_jp_eval.run \
    --model smolm-jp-4-1b \
    --tasks all \
    --num-examples 100

# LM Evaluation Harness
lm_eval --model hf \
    --model_args pretrained=./checkpoints/latest \
    --tasks hellaswag,arc_easy,arc_challenge \
    --batch_size 32
```

## 学習スケジュール (目安)

| フェーズ | 内容 | 期間目安 |
|---|---|---|
| Phase 0 | アーキテクチャ実装・テスト | 2〜3 週間 |
| Phase 1 | AdamW ベースライン学習 (GR なし) | 1〜2 週間 (4×A100) |
| Phase 1.5 | ベースライン評価・分析 | 3 日 |
| Phase 2 | GR 導入実験 | 1〜2 週間 |
| Phase 3 | 最終モデル学習・評価 | 1〜2 週間 |
| Phase 4 | SFT (オプション) | 1 週間 |
