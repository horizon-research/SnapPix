
### 学習方法
* 共通

```
main.py \
--root_path PATH \ データセットへのpath
--video_path PATH \ 学習データのroot、root_pathからの相対パス
--annotation_path PATH \ アノテーションファイルへの相対パス
--result_path PATH \ 学習結果の保存先
--dataset NAME \
--n_classes N \ クラス数
--sample_duration N \ Nフレームを1つの入力とする
--learning_rate 1e-2 \ 学習率
--no_mean_norm \ 平均値での正規化をしない(露光パターンを学習させるため)
--optimizer sgd \
--lr_patience N \ N回calidationのlossが下がらなかった場合に学習率を1/10にする
--batch_size N \ バッチサイズ
--n_epochs 200 \
--resume_path PATH \ 続きから学習する場合、同一モデルかつ同一クラス数で初期値にしたい場合
--init_path PATH \ 異なるモデルもしくは異なるクラス数で初期値にしたい場合
--init_level N \ init_pathの重みをN個分みて同一名の重みのみコピー
--checkpoint N \ N epochごとにモデルを保存
```


* SVC2D(露光パターン最適化)

```
--model c2d_pt_exp
```

* SVC2D(学習済み露光パターンを使用)

```
--model c2d_pt \
--compress mask \
--mask_path MASK_PATH
```

```
--model c2d_pt_exp \
--resume_path PRE_TRAINED_MODEL_PATH \
--fixed_mask
```

* C3D

```
--model c3d
```

* Short

```
--model c2d \
--compress one
```


* Long

```
--model c2d \
--compress avg
```

* Low spatial resolution video

```
--model c3d \
--compress spatial \
--use_cv2 \ open cvでリサイズ
--spatial_compress_size N \ NxNを1x1にリサイズ
```

### 使用しているファイル
```
main.py  # main file
opts.py  # options
train.py  # trainの処理
test.py  # testの処理
validation.py  # validationの処理
spatial_transforms.py  # 空間方向での変換
spatio_temporal_transforms.py  # 時空間での変換
temporal_transforms.py  # 時間方向の変換
target_transforms.py  # ラベルの変換
model.py
models/
├── binarized_modules.py  # exposure
├── c2d.py
├── c2d_exp.py
├── c2d_pt.py  # SVC2D
├── c2d_pt_exp.py  # SVC2D
├── c3d.py
└── pattern_conv_modules.py  # Shift-Variant
dataset.py
datasets/
├── gtea.py
├── hmdb51.py
├── kth2.py
├── real.py  # 実実験用
├── something2.py
├── ucf101.py
└── ucf50.py
eval_*.py  # 評価用
utils.py

