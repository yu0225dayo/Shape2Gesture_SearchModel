# Parts2Gesture

接触部位形状を介した、全体形状と両手把持ジェスチャの相互検索システム

## 研究背景

![Model overview](fig/model.png)

従来の物体把持検索・生成モデルは、物体の全体形状と把持姿勢の関係を学習している。つまり、学習されたモデルは訓練データに特化し、AGIの実現には膨大なデータ・計算コストが必要である。
しかし、バスケットとやかんのように形状が異なる物体でも把持姿勢が類似することに着目した。

本研究では、CLIPを参考に**把持姿勢と接触部位形状**の関係を学習し共通の潜在空間に埋め込むことで、汎用的なモデルの実現を目指す。このモデルを用いて、全体形状と把持姿勢の相互検索システムの構築を行う。
※接触部位形状の獲得はPointNetを用いて学習したモデルを用いる。
- PointNet: https://github.com/charlesq34/pointnet  
- CLIP: https://github.com/openai/CLIP

## 主な特徴

- ✅ **PointNetベース**: 順序不変な点群処理
- ✅ **マルチモーダル学習**: ジェスチャー + パーツ + ポイント
- ✅ **対照学習**: 相互検索システム(CLIPを参考に)

## プロジェクト構成

```
Parts2Gesture/
│
├── model_Contratstive_Parts2Gesture/              # 学習済みモデルディレクトリ
│   ├── pointnet_model_*.pth                       # パーツセグメンテーション
│   ├── contrastive_model_*.pth                    # ジェスチャー↔パーツ対比学習
│   └── parts2pts_model_*.pth                      # パーツ↔ポイント写像学習
│
└── utils_ContrastiveLearnig/                      # メイン実装モジュール
    │
    ├── model.py                                   # ニューラルネットワーク定義
    ├── dataset.py                                 # データセット読み込み
    ├── functions.py                               # 共有ユーティリティ関数
    ├── visualization.py                           # 3D可視化関数
    │
    ├── train.py                                   # モデル訓練スクリプト
    ├── calc_mIoU_partseg.py                       # パーツセグメンテーション評価
    ├── calc_mIoU_part2ges.py                      # パーツ→ジェスチャー評価
    │
    ├── show_pts2gesture.py                        # ポイント→ジェスチャー可視化
    ├── show_ges2pts.py                            # ジェスチャー→ポイント可視化
    └── show_cosinsim.py                           # コサイン類似度ヒートマップ
```

### モデル (model.py)

| モデル名 | 役割 | 入力 | 出力 |
|---------|------|------|------|
| **PointNetDenseCls** | パーツセグメンテーション | ポイントクラウド (B×2048×3) | 3クラス分類 (非接触/左手/右手) |
| **ContrastiveNet** | ジェスチャー↔パーツ対照学習 | 各手のジェスチャ × (パーツ + 全体特徴) | 類似度スコア行列 |
| **PartsToPtsNet** | パーツ↔ポイント対照学習 | 両手パーツ特徴 + 全体特徴 | 類似度スコア行列 |

## データセット詳細

### ディレクトリ構成

```
dataset/
├── train/                   # 訓練用データ（約1000サンプル）
│   ├── pts/                 # ポイントクラウド (CSV形式, 2048×3)
│   ├── pts_label/           # セグメンテーション教師信号 (CSV形式, 2048×1)
│   │                        # 0:背景, 1:左手接触, 2:右手接触
│   └── hands/               # 手ジェスチャー (CSV形式, 2×69)
│                            # 各手23個関節×3軸 
│
├── val/                     # 検証用データ（約200サンプル）
│   ├── pts/
│   ├── pts_label/
│   └── hands/
│
├── search/                  # 検索用データベース（参照セット）
│   ├── pts/
│   ├── pts_label/
│   └── hands/
```

## 使用方法

### 訓練

```bash
python train.py \
    --dataset path/to/dataset \
    --batchSize 16 \
    --nepoch 100 \
    --model path/to/model_dir  # オプション：既存モデルから続行
```

### 評価

```bash
# パーツセグメンテーション評価
python calc_mIoU_partseg.py \
    --model path/to/model \
    --dataset path/to/dataset

# パーツ→ジェスチャー評価
python calc_mIoU_part2ges.py \
    --model path/to/model \
    --dataset path/to/dataset
```

### 可視化（画面表示）
idxを指定し、任意のジェスチャ・全体形状から相互検索可能
```bash
# ポイント→ジェスチャー可視化
python show_pts2gesture.py \
    --model path/to/model \
    --dataset path/to/dataset \
    --idx 0

# ジェスチャー→ポイント可視化
python show_ges2pts.py \
    --model path/to/model \
    --dataset path/to/dataset \
    --idx 0

# コサイン類似度可視化
python show_cosinsim.py \
    --model path/to/model \
    --dataset path/to/dataset
```


## 必要環境
- Python 3.7
- PyTorch 1.9
- NumPy, Matplotlib, OpenCV, Pandas, tqdm

## インストール

```bash
pip install torch numpy matplotlib opencv-python pandas plyfile tqdm
```
