# コスト見積もり (実測ベース + 最新 Vast.ai 価格)

smolm-jp-4 の学習コスト。RTX 2070 Super (8GB) 実測。

## モデル概要

| 項目 | 値 |
|---|---|
| パラメータ数 | **624M** (8GB限界、grad ckpt有効) |
| アーキテクチャ | GDN+GQA hybrid (28層, hidden 1280) |
| 訓練データ | LLM-jp Corpus v4.1 (20B tokens) |
| 旧350Mモデル | 372M (1024/20/1536) - 比較用 |

## RTX 2070 Super 実測 (gradient checkpointing, bf16, AdamW)

| モデル | seq | s/step | tok/s | peak VRAM |
|---|---|---|---|---|
| 372M (1024/20/1536) | 1024 | 0.75 | 1,363 | 4.6GB |
| **624M (1280/28/1920)** | **1024** | **1.49** | **686** | **6.7GB** |
| **624M (1280/28/1920)** | **2048** | **2.97** | **688** | **7.15GB** |

## スケールアップ限界 (RTX 2070S, seq=2048, grad ckpt)

| hidden | layers | inter | params | peak | 結果 |
|---|---|---|---|---|---|
| 1024 | 20 | 1536 | 372M | 4.3GB | OK |
| 1024 | 32 | 1536 | 475M | 5.3GB | OK |
| 1152 | 24 | 1728 | 487M | 5.7GB | OK |
| **1280** | **28** | **1920** | **624M** | **7.08GB** | **採用** |
| 1280 | 30 | 1920 | 652M | 7.37GB | 限界 |
| 1408 | 24 | 2112 | 661M | 7.47GB | 絶対限界 |
| 1280 | 32 | 1920 | OOM | - | OOM |
| 1536 | 24 | 2304 | OOM | - | OOM |

絶対限界は 1408/24/2112 (661M, 7.47GB) だが headroom 0.3GB のみ。本採用は 1280/28/1920 (624M, 7.08GB) で 0.7GB 余裕。

## Vast.ai 最新価格 (2026年9月)

| GPU | VRAM | 単価 (on-demand) | 単価 (spot) |
|---|---|---|---|
| **RTX 3090** | 24GB | **$0.015/hr** | $0.014/hr |
| RTX 4090 | 24GB | $0.27/hr | **$0.13/hr** |

## 20B tokens のコスト (624Mモデル, seq=2048)

| GPU | 単価 | 所要時間 | **コスト** | 備考 |
|---|---|---|---|---|
| **RTX 3090** | $0.015/hr | ~168 日 | **$61** | 最安 |
| **RTX 4090 spot** | $0.13/hr | ~146 日 | **$456** |  |
| RTX 4090 on-demand | $0.27/hr | ~146 日 | $947 |  |
| ローカル 2070S | $0 | ~336 日 | $0 (+電気$300) |  |

比較: 旧372Mモデルなら RTX 3090 で $13/37日。624Mは約2倍遅く、コストも約5倍。

## 推奨

- **コスパ重視**: 372M (旧) → $13, 37日
- **性能重視**: 624M (新, 8GB限界) → $61, 168日
- ローカル2070Sのみは336日かかるため非推奨。Vast.ai RTX 3090が最安。

### Vast.ai の使い方

```bash
# https://cloud.vast.ai → Browse → RTX 3090
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install flash-linear-attention transformers datasets accelerate wandb
python -m smolm_jp4.train --config 624m --tokens 20b  # checkpoint必須
```
