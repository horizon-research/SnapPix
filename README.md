# 🚀 SnapPix

**Official Code for DAC'25**  
**📄 Paper Title:** *SnapPix: Efficient-Coding–Inspired In-Sensor Compression for Edge Vision*

---

## 📂 Dataset

### 🔗 Source  
We use [OpenMMLab Datasets](https://opendatalab.com/OpenMMLab) for all dataset used.📦

### ⚙️ Data Preprocessing  
Apply downsampling, inverse gamma correction, and grayscale conversion to dataset, here is an example for SSV2:

```bash
python preprocess_data.py ../../OpenDataLab___sthv2/raw/sthv2/sthv2/videos/ ssv2_processed/ --input_format .webm
```
See dataset/preprocessing.sh for more examples.

Process csv of SSV2:
```bash
# for finetuning dataset
python3 dataset/ssv2_list_process.py input.csv output.csv
# for pretraining dataset
python3 dataset/ssv2_list_process.py input.csv output.csv --pretrained
```

### 🧹 Generate K710 (Pretrain Dataset)  
Combining preprocessed K400 / K600 / K700 / SSV2 into one dataset:

```bash
python3 combine_pretrained.py dataset_lists/K710/train.csv dataset_lists/SSV2/train.csv \
mmdataset/k400_processed mmdataset/k600_processed mmdataset/k700_processed mmdataset/ssv2_processed combined_pretrain
```

Copy K400 / K600 / K700 to K710:
```bash
bash dataset/copy_k710.sh
```

---

## 🛠️ Environment Setup

Refer to the [VideoMAEv2 README](https://github.com/OpenGVLab/VideoMAEv2) for detailed environment installation instructions ✅

---

## 🧪 Decorrelated Pattern Training

Train using the decorrelation strategy:

```bash
python3 VideoMAEv2/run_decorrelation_training.py
```

🔍 A pretrained version is available at:  
`VideoMAEv2/decorrelation_training_wd0_norm_new`

---

## 🏋️️ Pretraining Scripts

Located in:  
`VideoMAEv2/scripts/pretrain_and_reconstruct`

Examples:  
- [`vits_pt.sh`](VideoMAEv2/scripts/pretrain_and_reconstruct/vits_pt.sh)  
- [`vitb_pt.sh`](VideoMAEv2/scripts/pretrain_and_reconstruct/vitb_pt.sh)

📌 Key Parameters:
- `OUTPUT_DIR`: Path to logs and checkpoints 📁  
- `DATA_PATH`: CSV list of data files 📄  
- `--data_root`: Dataset root (e.g., `/local_scratch/26477563/mmdataset/`) 🗂️

---

## 🎯 Finetuning Scripts

Found in `scripts/finetune/`

Examples:
- [`batch_k400.sh`](VideoMAEv2/scripts/finetune/batch_k400.sh)  
- [`ssv2_ptft.sh`](VideoMAEv2/scripts/finetune/ssv2_ptft.sh)  
- [`ucf_ptft.sh`](VideoMAEv2/scripts/finetune/ucf_ptft.sh)

🔧 Key Parameters:
- `OUTPUT_DIR`: Log/checkpoint directory  
- `DATA_PATH`: Dataset list path  
- `MODEL_PATH`: Path to pretrained model  
- `--data_root`: Dataset directory

---

## 📊 Evaluation Scripts

Evaluate on different datasets using:

- 📼 `scripts/K400_precise_val/` — *Kinetics-400*  
- 🎮 `scripts/SSV2_precise_val/` — *Something-Something V2*  
- 📹 `scripts/UCF_precise_val/` — *UCF-101*

---

## 🙏 Acknowledgements

A big thank you to:

- 🧠 **VideoMAEv2 authors** (Wang et al., CVPR 2023)  
  🔗 [VideoMAEv2 GitHub](https://github.com/OpenGVLab/VideoMAEv2)

- 🎥 **Action Recognition from a Single Coded Image**  
  📄 [IEEE Paper](https://ieeexplore.ieee.org/document/9105176)

We greatly appreciate the open-source / code-sharing contributions that made SnapPix possible 💡

