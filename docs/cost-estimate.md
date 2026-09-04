# コスト見積もり (実測ベース + 最新 Vast.ai 価格)

smolm-jp-4 の350M モデル学習にかかるコスト。

## モデル概要

| 項目 | 値 |
|---|---|
| パラメータ数 | 372M |
| アーキテクチャ | GDN+GQA hybrid (20層) |
| 訓練データ | LLM-jp Corpus v4.1 (20B tokens) |

## Vast.ai 最新価格 (2026年9月)

| GPU | VRAM | 単価 (on-demand) | 単価 (spot) | TFLOPS (FP16) |
|---|---|---|---|---|
| **RTX 3090** | 24GB | **$0.015/hr** | $0.014/hr | ~35 |
| L4 | 22GB | $0.032/hr | $0.032/hr | ~30 |
| RTX 4090 | 24GB | $0.27/hr | **$0.13/hr** | ~82 |
| A100 PCIe | 80GB | $0.28/hr | $0.16/hr | ~312 |

## スループット推定

RTX 2070S 実測: 0.75s/step, 1,363 tok/s (seq=1024, batch=1)

GDN の chunk モードはメモリ帯域幅に依存するため、GPU 帯域幅比で推定:

| GPU | 帯域幅 | 推定速度 | 推定 throughput |
|---|---|---|---|
| RTX 2070S (実測) | 448 GB/s | 0.75s/step | 1,363 tok/s |
| **RTX 3090** | 936 GB/s | **~0.36s/step** | **~2,850 tok/s** |
| L4 | 300 GB/s | ~1.1s/step | ~930 tok/s |
| **RTX 4090** | 1,008 GB/s | **~0.33s/step** | **~3,100 tok/s** |

## 20B tokens のコスト

| GPU | 単価 | 所要時間 | **コスト** | 備考 |
|---|---|---|---|---|
| **RTX 3090** | $0.015/hr | ~37 日 | **$13** | 最安。在庫多め |
| L4 | $0.032/hr | ~48 日 | $37 | VRAM 22GB でギリギリ |
| **RTX 4090 spot** | $0.13/hr | ~33 日 | **$103** | コスパ良好 |
| RTX 4090 on-demand | $0.27/hr | ~33 日 | $214 | 安定 |
| A100 PCIe spot | $0.16/hr | ~33 日 | $127 | VRAM 余裕あり |

## 結論

**RTX 3090 が圧倒的に安い。$0.015/hr = 1日あたり $0.36。**

| 方案 | コスト | 期間 | 信頼性 |
|---|---|---|---|
| **RTX 3090 (推奨)** | **$13** | 37 日 | 中 (spot は切断リスク) |
| RTX 4090 spot | $103 | 33 日 | 中 |
| RTX 4090 on-demand | $214 | 33 日 | 高 |
| ローカル 2070S のみ | $0 | 170 日 | 最高 (自前) |

### 注意点

- RTX 3090 の $0.015/hr は Vast.ai marketplace の最安値。在庫は常に変動
- spot instance はプロバイダ都合で切断される可能性あり。**チェックポイントを頻繁に保存する設計が必須**
- 3090 は 24GB VRAM があるので、batch_size=2 にも増やせる (速度2倍)
- 長期レンタル (30日+) でプロバイダと交渉すると 5-15% 割引の場合あり

### Vast.ai の使い方

```bash
# 1. https://cloud.vast.ai にアクセス
# 2. "Browse" → Filter: RTX 3090, 24GB VRAM
# 3. 最安値のインスタンスを選択 (Reviews と Uptime を確認)
# 4. SSH 接続後:

pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install flash-linear-attention transformers datasets accelerate wandb

# 5. 学習開始 (チェックポイント自動保存あり)
python -m smolm_jp4.train --config 350m --tokens 20b
```
