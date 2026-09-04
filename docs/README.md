# smolm-jp-4

1Bクラス・日本語エージェント用途の dense 言語モデル。

Qwen3.8-Flash-Next (125B MoE) の GDN+GQA ハイブリッドアーキテクチャを、LLM-jp-4 コーパス・トークナイザで日本語モデルに大幅に缩小移植する。

## プロジェクト目標

| 項目 | 仕様 |
|---|---|
| パラメータ数 | ~1B (dense, MoE なし) |
| トークナイザ | LLM-jp-4 (vocab_size=196,608) |
| 対象用途 | エージェント (tool use, function calling) |
| コンテキスト長 | 32K〜64K |
| 言語 | 日本語中心、英語・中国語・韓国語・コード混合 |
| 将来スケール | 10B〜20B への拡張を考慮した設計 |

## ドキュメント一覧

| ファイル | 内容 |
|---|---|
| [README.md](./README.md) | このファイル (プロジェクト概要) |
| [architecture.md](./architecture.md) | アーキテクチャ詳細 (レイヤー構成、パラメータ配分) |
| [training.md](./training.md) | 学習パイプライン (コーパス、ハイパーパラメータ、 infra) |

## クイックスタート

```bash
# 1. リポジトリクローン
git clone <repo-url>
cd slm-jp-4

# 2. 依存関係インストール
pip install -e ".[dev]"

# 3. モデルサイズ確認
python -c "
from smolm_jp4.config import SmolmJp4Config
from smolm_jp4.model import SmolmJp4ForCausalLM
config = SmolmJp4Config()
model = SmolmJp4ForCausalLM(config)
print(f'Parameters: {model.num_parameters():,}')
"
```

## 参考文献

- [Qwen3.8-Flash-Next 技術レポート](https://arxiv.org/abs/2608.30320) (arXiv:2608.30320, 2026-08-31)
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) (ICLR 2025)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (GDN カーネル実装)
- [LLM-jp Corpus v4.1](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4.1)
- [LLM-jp-4 Tokenizer](https://github.com/llm-jp/llm-jp-tokenizer)
