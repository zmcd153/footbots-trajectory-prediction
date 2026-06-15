# Football Multi-Agent Trajectory Prediction

俯瞰サッカー映像から選手・ボールを検出して競技場座標へ変換し、軌跡の前処理、ボール補完、保持者推定、将来軌跡予測、動画可視化までを行う研究用プロジェクトです。

本実装は、FootBotsおよびTranSPORTmerの時空間・集合注意の考え方を参考にした独自の簡略化実装です。原論文の公式実装または完全再現ではありません。

## 主な機能

- YOLOによる選手・ボール検出
- ByteTrack / BoT-SORTによる多物体追跡
- ホモグラフィによる画像座標から競技場座標への変換
- 軌跡の範囲制約、短軌跡除去、短区間補間
- 時間注意と社会注意を分離したTrajectory Transformer
- Set Transformerと双方向LSTMによるボール軌跡推定
- ボール軌跡補完とボール保持者推定
- 等速度予測および4モードの規則ベース予測
- 元映像または俯瞰競技場への予測結果描画

## システム構成

```text
俯瞰映像
  -> YOLO検出
  -> ByteTrack / BoT-SORT
  -> ホモグラフィ変換
  -> 軌跡前処理
  -> ボール補完・保持者推定
  -> Transformer / ボール予測 / 多モード予測
  -> CSV・可視化動画
```

## リポジトリ構成

```text
.
├── configs/              # ホモグラフィとチーム設定の例
├── data/                 # ローカル動画・追跡データ用（Git管理外）
├── dataset/              # YOLOデータセット用（Git管理外）
├── docs/                 # 詳細な日本語ドキュメント
├── runs/                 # 学習・推論結果用（Git管理外）
├── scripts/              # リポジトリ検証用スクリプト
├── src/                  # 実装本体
├── .gitignore
├── CITATION.cff
├── pyproject.toml
└── requirements.txt
```

## 必要環境

- Python 3.10以上
- Windows / Linux / macOS
- NVIDIA GPUは任意ですが、YOLOおよび軌跡モデルの学習にはCUDA環境を推奨します

## セットアップ

```bash
git clone https://github.com/<YOUR_ACCOUNT>/footbots-trajectory-prediction.git
cd footbots-trajectory-prediction

python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorchのCUDA版が必要な場合は、先に[PyTorch公式サイト](https://pytorch.org/get-started/locally/)の環境別コマンドでPyTorchを導入してください。

## データについて

動画、注釈データ、学習済み重みは容量および再配布条件のため、このリポジトリには含めません。各自で権利を確認したデータを`data/`または`dataset/`へ配置してください。

YOLOデータセットは次の構造を使用します。

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

クラス定義:

```text
0 player
1 ball
```

## クイックスタート

### 1. 動画から注釈用フレームを抽出

```bash
python -m src.extract_frames \
  --video data/match.mp4 \
  --out-dir data/annotation_frames \
  --every 30 \
  --max-frames 500
```

CVAT、Label Studio、Roboflowなどで修正し、YOLO形式で保存してください。

COCO事前学習モデルによる自動仮ラベル付きデータセットも作成できます。

```bash
python -m src.build_yolo_dataset \
  --videos data/match01.mp4,data/match02.mp4 \
  --dataset-dir dataset \
  --imgsz 1280
```

自動ラベルはそのまま正解データとして使用せず、必ず人手で確認してください。

### 2. YOLOデータセットを検証

```bash
python -m src.train_yolo \
  --dataset-dir dataset \
  --dry-run
```

### 3. 検出器を学習

```bash
python -m src.train_yolo \
  --dataset-dir dataset \
  --model yolov8s.pt \
  --imgsz 1280 \
  --epochs 80 \
  --batch 8 \
  --name soccer_topview
```

最良重みは通常、次へ保存されます。

```text
runs/detect/soccer_topview/weights/best.pt
```

### 4. ホモグラフィを設定

```bash
python -m src.export_calibration_frame \
  --video data/match.mp4 \
  --frame 0 \
  --out runs/calibration_frame.jpg
```

`configs/homography.example.json`をコピーし、映像内の競技場点と実競技場座標を設定します。4点以上の非共線対応点が必要です。

```bash
cp configs/homography.example.json homography.json
```

### 5. 選手・ボールを追跡

```bash
python -m src.track_video \
  --video data/match.mp4 \
  --weights runs/detect/soccer_topview/weights/best.pt \
  --homography homography.json \
  --tracker bytetrack.yaml \
  --out runs/raw_tracks.csv
```

出力CSVの主要列:

```text
frame,agent_id,agent_type,x,y,score,source_track_id,det_class
```

- `agent_id=0` / `agent_type=0`: ボール
- `agent_type=1,2`: 選手またはチーム種別
- `x,y`: 競技場上のメートル座標

### 6. 軌跡を前処理

```bash
python -m src.prepare_tracks \
  --tracks runs/raw_tracks.csv \
  --out runs/clean_tracks.csv \
  --min-length 30 \
  --max-gap 10
```

### 7. ボール軌跡と保持状態を補完

```bash
python -m src.tactical_features \
  --tracks runs/raw_tracks.csv \
  --out runs/possession_initial.csv

python -m src.complete_ball_track \
  --tracks runs/raw_tracks.csv \
  --possession runs/possession_initial.csv \
  --out runs/ball_completed.csv \
  --merge-out runs/tracks_with_ball.csv
```

### 8. 軌跡モデルを学習

選手を含む全エージェント:

```bash
python -m src.train_real \
  --tracks runs/clean_tracks.csv \
  --obs-steps 20 \
  --pred-steps 40 \
  --epochs 50 \
  --out runs/trajectory_transformer.pt
```

ボール専用モデル:

```bash
python -m src.train_ball \
  --tracks runs/tracks_with_ball.csv \
  --obs-steps 20 \
  --pred-steps 40 \
  --epochs 30 \
  --out runs/ball_set_bilstm.pt
```

### 9. 将来軌跡を推論

学習済みTransformer:

```bash
python -m src.predict_tracks \
  --checkpoint runs/trajectory_transformer.pt \
  --tracks runs/clean_tracks.csv \
  --out runs/predictions.csv
```

ボール:

```bash
python -m src.predict_ball \
  --checkpoint runs/ball_set_bilstm.pt \
  --tracks runs/tracks_with_ball.csv \
  --out runs/ball_predictions.csv
```

軽量な等速度基準:

```bash
python -m src.predict_linear \
  --tracks runs/clean_tracks.csv \
  --out runs/predictions_linear.csv
```

規則ベースの4モード予測:

```bash
python -m src.predict_multimodal \
  --tracks runs/tracks_with_ball.csv \
  --possession runs/possession_initial.csv \
  --out runs/predictions_multimodal.csv
```

### 10. 結果を可視化

競技場ビュー:

```bash
python -m src.render_field_video \
  --tracks runs/clean_tracks.csv \
  --predictions runs/predictions.csv \
  --out runs/predictions_field.mp4
```

元映像への描画:

```bash
python -m src.render_prediction_video \
  --video data/match.mp4 \
  --homography homography.json \
  --tracks runs/clean_tracks.csv \
  --predictions runs/predictions.csv \
  --out runs/predictions_overlay.mp4
```

多モード予測:

```bash
python -m src.render_multimodal_video \
  --video data/match.mp4 \
  --tracks runs/tracks_with_ball.csv \
  --predictions runs/predictions_multimodal.csv \
  --possession runs/possession_initial.csv \
  --draw-uncertainty \
  --out runs/predictions_multimodal.mp4
```

## 評価指標

軌道予測では、少なくとも次の指標を推奨します。

- ADE: 全予測時刻の平均位置誤差
- FDE: 最終予測時刻の位置誤差
- minADE / minFDE: 複数候補中の最小誤差
- Miss Rate: 許容誤差を超えた予測の割合

現在のコードは学習中のmasked ADEを出力しますが、論文レベルの比較には試合単位のデータ分割、等速度・LSTM等の基準モデル、アブレーション実験が必要です。

## 既知の制約

- 固定または準固定の俯瞰カメラを想定しています。
- ボール検出精度がパイプライン全体の主要な制約です。
- `MaskedTrackWindowDataset`は欠損値をゼロ埋めしますが、主体Transformerの注意機構には完全なpadding maskをまだ導入していません。
- 規則ベースのマルチモーダル確率は学習された確率ではありません。


## 参考研究

- FootBots: A Transformer-based Architecture for Motion Prediction in Soccer
- TranSPORTmer: A Holistic Approach to Trajectory Understanding in Multi-Agent Sports
- Set Transformer
- ByteTrack

