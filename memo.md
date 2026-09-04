# smolm-jp-4: 設計・学習パイプライン方針まとめ

このドキュメントは、smolm-jp-4（1Bクラス・日本語エージェント用途の dense 言語モデル）の
アーキテクチャ設計とトレーニングパイプライン設計について、これまでの検討過程と結論を
整理したものです。別のAI/エンジニアが引き継いで作業する際のコンテキストとして使用してください。

> **詳細ドキュメント**: `docs/` ディレクトリに整理済み。
> - `docs/README.md`: プロジェクト概要・クイックスタート
> - `docs/architecture.md`: アーキテクチャ詳細 (レイヤー構成、パラメータ配分)
> - `docs/training.md`: 学習パイプライン (LLM-jp Corpus v4.1、ハイパーパラメータ、infra)

## 1. プロジェクトの出発点

- 元ネタは Qwen3.8-Flash-Next（125B total, 6B active, 51B n-gram embedding, MoE, hybrid
  GDN+QSA attention, Gated Residual, Muon optimizer）のアーキテクチャを縮小コピーしようとした
  draft config（`config.txt`、model_type=`qwen4_exp`）。
- 目的は、このアーキテクチャの主要アイデアを **LLM-jp-4 のコーパス・トークナイザ
  （vocab_size=196,608）を使う日本語モデルに、大幅に小さいスケール（当初2B→最終的に1B target）で
  移植する**こと。
- 想定用途：**エージェント用途**。文脈長は **32K〜64K** を主戦場と想定。ただし
  **将来 10B〜20B へスケールアップする可能性がある**という制約下で設計する。
- 学習リソースは **数枚のGPU（研究室・個人規模）**。大規模分散学習基盤は前提にしない。

論文の答え合わせ（Qwen3.8-Flash-Next技術レポート, arXiv:2608.30320, 2026-08-31, Qwen Team）を 2608_30320v1.pdf で確認済み。詳細は本ドキュメント末尾の「論文からの主要な知見」を参照。（現在はローカルファイル上にない。ウェブ検索やwget, curlによる取得が必要）

## 2. アーキテクチャ方針（結論）

### 2.1 削除したコンポーネントとその理由

| コンポーネント | 判断 | 理由 |
|---|---|---|
| n-gram embedding table | **削除** | 元モデルでは「300 tokens per active parameter」やvocab倍率（20×〜200×）で管理される、本体とは別枠の巨大な記憶装置。125B級MoEでは合理的なトレードオフだが、1B dense に同じ発想を持ち込むとテーブルだけで本体を何倍も上回り、パラメータ予算が破綻する。MoE expertを削って予算を捻出する前提の機構でもあり、dense小型モデルとは設計思想が根本的に合わない。 |
| QSA (Qwen Sparse Attention) / DSA (DeepSeek Sparse Attention) | **見送り（将来再検討）** | 効果は主に長文脈（論文の測定は512K〜1M）で顕在化する。想定文脈32K〜64Kではメリットが薄く、indexer学習に必要な Dense Warm-up ステージ（indexerだけ1000 step事前学習→backboneと合わせて8000 step sparse training、論文では計約200Bトークン規模）という追加の学習パイプライン工程のコストに見合わない。10B〜20Bへのスケールアップ時、かつ文脈長要件が64Kを大きく超える場合に再検討する。 |
| MoE（ルーティング、複数エキスパート） | **削除（dense化）** | 当初のconfig.txt自体がdense化を明言しており、この方針を継続。 |

### 2.2 採用したコンポーネント

- **GDN（Gated DeltaNet）+ 通常GQA のレイヤー単位ハイブリッド**：4層に1回が通常のGQA
  full-attention、残り3/4がGDN linear-attention。論文のGDN Hybridの構成をそのまま踏襲（QSA/indexer部分だけ除去）。
- **RoPE**：full-attention層にのみ適用（GDN層には不要）。論文でもNoPE variantは事後学習後の生成品質に悪影響（無限生成しやすくなる）と報告されており、RoPEは維持。
- **Gated Residual (GR) 相当の仕組み**：**段階的導入で保留中**。詳細は §3 参照。
- **MTP（Multi-Token Prediction）**：1層追加する方針を維持。

### 2.3 確定した1Bパラメータ配分

vocab_size は llm-jp-4 のトークナイザに合わせて 196,608 に固定（変更不可の制約）。

```
hidden_size = 1536
num_hidden_layers = 28
num_attention_heads = 24
num_key_value_heads = 2
head_dim = 64

full_attention_interval = 4   # 28層中7層がGQA、21層がGDN
linear_key_head_dim = 192
linear_num_key_heads = 8
linear_value_head_dim = 96
linear_num_value_heads = 16
linear_conv_kernel_dim = 4

intermediate_size = 3072      # SwiGLU, hidden比2.0倍
tie_word_embeddings = true    # 必須。untiedだと+302Mで予算超過
```

概算パラメータ内訳（合計 ~981M、GR抜き）:

| コンポーネント | 概算 | 比率 |
|---|---|---|
| トークン埋め込み（tied） | 302M | 30.8% |
| Full-attn層（GQA、7層） | 36M | 3.6% |
| Linear-attn層（GDN、21層） | 228M | 23.2% |
| FFN（全28層） | 396M | 40.4% |
| MTP | 19M | 2.0% |

GR (nr=4 フル版) を追加する場合は概算 +134M（合計 ~1.11B、詳細は§3参照）。

**注意**：上記はQ/K/V/O射影の行列形状から逆算した解析的概算値であり、norm・biasなどは
含まれていない。実装後は必ず実測パラメータ数で再検証すること。

### 2.4 スケーリング参考値（将来の10B〜20B版）

同じ設計比率（vocab固定、GDN:full=3:1、FFN比2.0倍）でのラフな目安：

```
hidden=2560, layers=32  -> ~2.8B
hidden=3584, layers=36  -> ~5.8B
hidden=4096, layers=40  -> ~8.2B
```

10B〜20B帯は hidden=4096〜5000台、layers=40〜48 あたりが目安。**この規模になった時点で
QSA/DSA導入とMuon optimizerの本格採用を再検討する。**

## 3. 未確定事項（次のアクションが必要）

### 3.1 Gated Residual (GR) の扱い

論文のGRは、残差ストリームをnr=4本に拡張し、read/write/GatedNormを各層（attn/mlpそれぞれ）に
挿入する機構。学習安定性への効果が非常に大きい（§3.3 stress testでMuon+GRは4倍学習率でも
loss spikeゼロ）一方、以下のコストがある：

- パラメータ増加：1B構成で +134M（許容範囲）
- **activationメモリが残差ストリーム分だけ約4倍**（数枚GPU環境ではこちらがボトルネックになりうる）
- 実装の複雑さ（read gate, write gate, per-branch RMSNormなど複数モジュール）

**現時点の暫定方針**：段階的導入。
1. まず GR なし・AdamW で学習パイプラインを一通り動かす
2. 次に GatedNorm 相当（式29、残差の拡張なし、gateのみ）を追加して安定性への効果を確認
3. それでも不安定性が問題になる場合のみ、nr=4のフルGRへ拡張する

→ **次のアクション**：①のベースライン学習が回った時点で、②の効果を実測して判断すること。

### 3.2 Optimizer（Muon vs AdamW）

**結論：現時点ではAdamWを採用し、Muonは見送る。**

理由：
- Muonの効果（4倍LRでの安定性、batch-size warmup不要など）は数兆トークン規模の長時間学習で
  顕在化する問題を解決するもの。1B・数十B〜100Bトークン規模の学習では、AdamWでも
  大きな不安定性が出にくく、Muon導入の効果検証自体のS/N比が悪い。
- 論文のMuon実装（Canzona）はTP/DP環境でのパラメータ再配置など、大規模分散学習を前提にした
  エンジニアリングを大量に含む。数枚GPU・FSDP/DDP という今回の構成では、その大半が不要な
  オーバーヘッドになる。
- NS反復・パラメータ選別（どの重みをMuon対象にするか）・fused parameter分割は規模に依らず
  必要な実装コストで、1Bでの早い実験サイクルという目的から外れるリスクがある。

**再検討条件**：10B〜20Bへのスケールアップが具体的に近づいた場合。その際は
「1Bのうちに小さくMuon実装をデバッグしておく」投資判断もあり得るが、GRなど他の変更と
同時に入れず、要素を1つずつ切り分けて導入すること。

## 4. 学習インフラ方針

- **並列化戦略**：ZeRO/FSDP相当のデータ並列のみで十分。1Bモデルの混合精度Adam状態は
  概算16GB程度で、数枚のGPUで無理なく収まる規模。Megatron-LM級の3D並列（TP+PP+DP）は
  オーバーエンジニアリング。
- **フレームワーク**：PyTorch FSDP（またはDDP）+ 既存のGDN/linear-attentionカーネル実装
  （`flash-linear-attention` (fla) ライブラリなど）を流用する方針。
- **参考実装**：Qwen3.8-Flash-Nextの実装は `transformers` ライブラリに統合されている
  （`qwen4_exp` 相当のアーキテクチャ名。正確なモジュール名は要確認）。ここから
  QSA/indexer関連クラスとn-gram embedding関連クラスを削除し、GDN層・GQA層・
  （将来的な）GR・MTP関連コードを流用する方針。vLLM/SGLangにも推論実装があるが、
  学習には直接関係しない。

## 5. 論文からの主要な知見（Qwen3.8-Flash-Next技術レポート要約）

- **GDN Hybrid の優位性**：full-attentionのみ、SWA hybrid、GDN hybrid の3種を比較した
  アブレーションで、GDN hybridが9ベンチマーク中8で改善、平均スコアも最高（Tab.1）。
- **QSA**：micro-block単位でtop-kを選ぶ軽量indexer方式。token budget K=2048、
  圧縮比r=4で最大512ブロック選択。1M文脈でprefill 7.6倍・decode 4.9倍高速化。
  短文脈での性能劣化はほぼ無い（Tab.2, Tab.3）が、その効果は文脈長に強く依存する。
- **n-gram embedding**：placement（層の位置）は2層目が最適、単一層で十分という結果
  （Tab.7）。vocabをスケールすると損失は単調に下がるが、下流ベンチマークは頭打ちになる
  （Tab.9）——loss改善とタスク性能改善が必ずしも一致しない好例として報告されている。
- **Gated Residual**：nr=4本の残差拡張＋elementwise gate。損失・ベンチマーク両面で改善
  するほか、学習安定性（loss spike, 勾配ノルムのoutlier抑制）への寄与が特に大きい
  （§2.2, §3.3, Fig.10-13）。
- **Muon optimizer**：行列パラメータのみに適用（埋め込み・出力ヘッド・MoEルーター・
  GRの低ランク射影はAdamWのまま）。GRとの組み合わせで学習率・バッチサイズを
  大きく取れるようになり、batch-size warmupが不要になった（§3.2）。

## 6. 参考ファイル

- `/mnt/user-data/uploads/config.txt`：最初のdraft config（縮小前、qwen4_exp系）
- `smolm-jp-4-1b.txt`（本セッションで生成、`/mnt/user-data/outputs/`）：1B target確定版config
- `/mnt/user-data/uploads/2608_30320v1.pdf`：Qwen3.8-Flash-Next技術レポート
  （arXiv:2608.30320, 2026-08-31, Qwen Team）

## 7. 350M モデル設計 (2026-09-03 時点)

### 7.1 背景

ローカル GPU が GTX 1080 (8GB) と RTX 2070 Super (8GB) のみ。1B モデルの事前学習は 8GB VRAM では
不可能（モデル重み+AdamW で 11-14GB 必要）ため、350M モデルに缩小。

### 7.2 設定

vocab_size=196,608 (LLM-jp-4 トークナイザ) は固定。embedding テーブルだけで ~201M と大きいため、
350M のうち約 54% が埋め込み层。トランスフォーマ本体は ~171M。

```
hidden_size = 1024
num_hidden_layers = 20
intermediate_size = 1536
num_attention_heads = 8
num_key_value_heads = 2 (GQA)
gdn_num_heads = 8
gdn_num_v_heads = 8 (expand_v=1.0)
tied_embeddings = true
max_position_embeddings = 32768
```

### 7.3 パラメータ内訳 (~372M)

| コンポーネント | 概算 | 比率 |
|---|---|---|
| トークン埋め込み (tied) | 201M | 54.1% |
| GDN 層 ×15 | 134M | 36.0% |
| Full-attn 層 ×5 (GQA) | 37M | 9.9% |
| FFN (SwiGLU, 全20層) | -- | (GDN/FFN 層に含む) |

### 7.4 VRAM 予測 (RTX 2070 Super)

- Forward のみ: ~1.8 GB
- Forward + Backward (seq=128): ~2.2 GB
- Forward + Backward (seq=2048, gradient ckpt): ~4-5 GB (推定)
- AdamW 含む合計: ~5.6 GB

→ 8GB に十分収まる

### 7.5 学習スケジュール

- 20B tokens / (batch=1 × seq=2048) ≈ 9.77M steps
- RTX 2070S: ~1,500 steps/h → 271 日 (ローカルのみ)
- RTX 4090 (Vast.ai): ~6,000 steps/h → 68 日 ($230-290)

### 7.6 コスト

| 方案 | コスト | 期間 |
|---|---|---|
| ローカルのみ | $0 (+ 電気代 ~$185) | ~9 ヶ月 |
| Vast.ai RTX 4090 spot | $230-290 | ~2-3 ヶ月 |
| 混合 (推奨) | $230-290 | ~3 ヶ月 |

詳細は `docs/cost-estimate.md` 参照。
