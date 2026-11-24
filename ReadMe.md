<div align="center">

# 🇰🇷 Korean Skin Condition Measurement Dataset

**The first AI dataset of Korean skin conditions & AI models for skin diagnosis**

[![Project Page](https://img.shields.io/badge/Project-Homepage-brightgreen)](https://leejeongho3214.github.io/NIA)
[![Dataset in AI-Hub](https://img.shields.io/badge/Dataset-AI--Hub-blue)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71645)
[![Paper - SRT 2026](https://img.shields.io/badge/Paper-SRT_2026_(SCIE)-b31b1b)](https://doi.org/10.1111/srt.70375)
[![Contact](https://img.shields.io/badge/Contact-Email-informational?logo=gmail)](mailto:72210297@dankook.ac.kr)

</div>

---

## 📢 Main Paper

> **Artificial Intelligence Based Skin Analysis Models for Predicting Visual Grades and Device Measured Physiological Values From Facial Images**
>
> Eunyoung Lee, **Jeongho Lee**, Nahee Kim, Junchae Na, Byungcheol Park, Sang-Il Choi
> *Skin Research and Technology*, Vol. 32, No. 9, e70375 (2026) · Open Access (CC BY 4.0)
> 📄 [https://doi.org/10.1111/srt.70375](https://doi.org/10.1111/srt.70375)

Built on the dataset and models in this repository, this work proposes an AI framework that predicts both **dermatologists' visual grades** and **device-measured physiological values** from facial images. It was validated on 1,099 Korean participants (aged 14–69) across three capture-device settings.

---

## 🆕 News

| Date | Update |
| --- | --- |
| **2026-09-01** | 🎉 Paper published in *Skin Research and Technology* (SCIE) — [10.1111/srt.70375](https://doi.org/10.1111/srt.70375) |
| **2026-03-21** | Re-uploaded the model checkpoint and dataset split download links |
| **2025-12-03** | Released model checkpoints and dataset split files; updated training and test code |

---

## 📌 Overview

- **The first AI dataset of Korean skin conditions**
- **Participants**: 1,100 men and women, from their teens to their 60s and older
- **Three capture devices**: digital camera (DSLR), tablet, and smartphone
- Multi-view facial images from **up to 7 angles**
- **Bounding boxes for 8 key facial regions** on every image
- Both **dermatologist visual assessments** and **precision instrument measurements**

---

## 🗂️ Dataset Composition

### 📷 Images

| Device | Angles | Notes |
| --- | --- | --- |
| Digital camera | 7 | Darkroom, with a face-fixing apparatus |
| Tablet | 3 | Controlled background and lighting |
| Smartphone | 3 | Controlled background and lighting |

### 🏷️ Labels

**① Dermatologist visual assessment (`class`)**

- Graded by 5 board-certified Korean dermatologists
- Items: pigmentation, lip dryness, pores, jawline sagging, wrinkles, and more
- The grade range differs by item

**② Precision instrument measurements (`regression`)**

- Devices validated in SCI-indexed publications and certified by the Korean Ministry of Food and Drug Safety (MFDS)
- Items: pores, pigmentation, wrinkles, hydration, and elasticity

### 🧪 Acquisition Protocol

- After face washing, participants acclimated in a **temperature- and humidity-controlled room** before capture
- Digital camera images were taken in a **darkroom** using a face-fixing apparatus

---

## 🧠 Skin Diagnosis AI Models

This repository releases the **ResNet-50** baseline of the paper (Table 6, visual-grade
classification on the DSLR test set). The CoAtNet-4 models of the paper are not part of this release.

### Architecture

- **Backbone**: ResNet-50 (torchvision, randomly initialised, no ImageNet pretraining), input 256×256
- The output size of the final fc layer equals the number of grades for the task
  (dryness 5, pigmentation 6, pore 6, sagging 6, wrinkle 7)
- A **separate model** is trained for each of the 5 visual-grade tasks

### Training Setup (the defaults of `tool/main.py`, which produced the released checkpoints)

| Item | Value |
| --- | --- |
| Optimizer | Adam (β = 0.9, 0.999, no weight decay), gradient clipping 1.0 |
| Learning rate | 1e-4, 5 linear warm-up epochs, cosine decay to 1e-6 |
| Epochs | max. 100, early stopping after 11 epochs without validation improvement |
| Batch size | 32, class-balanced `WeightedRandomSampler` (weights ∝ 1/√count) on the train split |
| Loss | class-balanced focal loss (γ = 2, β = 0.999) + 0.3 × smooth-L1(expected grade, GT grade) |
| Model selection | lowest validation MAE of the expected grade Σₖ softmaxₖ·k |
| Augmentation | `light` preset (affine, colour jitter, perspective, blur/sharpen, erasing; no flip), softened for dryness/pore/wrinkle; sampled once per image when the train split is loaded |
| Train / Val / Test | 7 : 1 : 2, split at the **participant level**, 4 random splits (seeds 1–4) |

The test split is never used during training.

---

## 🚀 Getting Started

### 1. Download Resources

| Resource | Link | Location |
| --- | --- | --- |
| Raw facial images & labels (AI-Hub) | [Link](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71645) | `dataset/img`, `dataset/label` |
| ResNet-50 checkpoints (seeds 1–4, `class`) | [Link](https://gofile.me/7wbhv/FpFhxyCZX) | `checkpoint/class/{1-4}st_resnet50/save_model/<task>/state_dict.bin` |
| Dataset split JSON | [Link](https://gofile.me/7wbhv/hpeVMNNOg) | `dataset/split` |

> 🔑 To get the **password** for the archives, please contact us by email → [72210297@dankook.ac.kr](mailto:72210297@dankook.ac.kr)
>
> The split JSON files contain participant-level 7:1:2 train/val/test splits for each facial sign, provided as four sets (**seeds 1–4**), for each device (`digital_camera`, `smart_pad`, `smart_phone`) and label type (`class`, `regression`).

> ℹ️ The raw data is hosted on [AI-Hub](https://www.aihub.or.kr). Access may require an AI-Hub account and a data-use application; please check AI-Hub's policy for overseas users.

### 2. Directory Structure

```
{$ROOT}
├── checkpoint
│   └── class
│       └── 1st_resnet50          # one folder per split seed (1st … 4st)
│           └── save_model
│               └── <task>/state_dict.bin
│
├── dataset
│   ├── img            # raw images from AI-Hub
│   ├── label          # AI-Hub labels (json)
│   ├── split          # train/val/test split json
│   └── cropped_img    # output of img_crop.py
│
└── tool
    ├── img_crop.py
    ├── main.py
    └── test.py
```

### 3. Preprocessing — Image Cropping

The CNN expects square inputs. Since the bboxes in the original JSON fit tightly around each region, they are made square before resizing:

a square window centred on the bbox (side = the longer bbox side) is cut out and resized to
256×256; where the window runs past the image border, the clipped crop is placed in the top-left
corner of a black square (zero padding) instead of being stretched.

```bash
python tool/img_crop.py
```

### 4. Training

The run name must start with the split seed (`1st_…` uses `dataset/split/*/1_*set_info.json`).
Checkpoints are written to `checkpoint/<mode>/<name>/`.

```bash
# Predict dermatologist visual grades (ResNet-50 recipe of the paper), seeds 1-4
for s in 1 2 3 4; do python tool/main.py --name ${s}st_resnet50 --mode class; done

# Predict instrument-measured values (not part of the released results)
python tool/main.py --name 1st_resnet50 --mode regression
```

### 5. Testing

```bash
for s in 1 2 3 4; do python tool/test.py --name ${s}st_resnet50 --mode class --equ 1; done
python tool/summarize.py --mode class --names 1st_resnet50 2st_resnet50 3st_resnet50 4st_resnet50
```

`test.py` scores each seed the way the paper does: the predicted grade is the expected grade
Σₖ softmaxₖ·k, MAE and Pearson r are computed on it, and the accuracies (%MAE = 0, ≤ 1, ≤ 2) on
the expected grade rounded to the nearest integer and clipped to the GT grade range. The
`[POOLED over tasks]` line pools all test images of all tasks of one seed; `summarize.py`
averages it over the four seeds. Use `--argmax` to score the argmax grade instead. For
regression, NMAE = MAE / (max GT − min GT).

Expected result with the released checkpoints (DSLR test set, mean of seeds 1–4, paper Table 6):

| Model | MAE | Pearson r | %MAE = 0 | %MAE ≤ 1 | %MAE ≤ 2 |
| --- | --- | --- | --- | --- | --- |
| ResNet-50 | 0.63 | 0.79 | 50.90 | 91.52 | 98.86 |

The table was produced with the crops used during training. About 1 in 5 of those crops differ
slightly from what `img_crop.py` produces (a few pixels of black margin at the window edge), so
crops regenerated with `img_crop.py` give marginally different numbers. Measured with the released
checkpoints on regenerated crops: MAE 0.63, r 0.79, %MAE = 0 50.96, ≤ 1 91.45, ≤ 2 98.84.

---

## 📚 Citation

If you use this dataset or code, please cite:

```bibtex
@article{lee2026skin,
  title   = {Artificial Intelligence Based Skin Analysis Models for Predicting
             Visual Grades and Device Measured Physiological Values From Facial Images},
  author  = {Lee, Eunyoung and Lee, Jeongho and Kim, Nahee and Na, Junchae
             and Park, Byungcheol and Choi, Sang-Il},
  journal = {Skin Research and Technology},
  volume  = {32},
  number  = {9},
  pages   = {e70375},
  year    = {2026},
  doi     = {10.1111/srt.70375}
}
```

---

## 📰 Related Publications

| Year | Venue | Link | Notes |
| --- | --- | --- | --- |
| 2026 | Skin Research and Technology (SCIE) | [Paper](https://doi.org/10.1111/srt.70375) | Main paper |
| 2025 | KIISE Transactions on Computing Practices (KTCP) | [Paper](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12252203) | Invited paper |
| 2024 | Korea Software Congress (KSC), KIISE | [Paper](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12041791) | 🏆 Outstanding Paper Award |
| 2024 | Korea Computer Congress (KCC), KIISE | [Paper](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11862094) | 🏅 Outstanding Presentation Award |

---

## 📬 Contact

> Ph.D. Student, Department of Computer Science, Dankook University
> **Jeongho Lee**
> 📧 [72210297@dankook.ac.kr](mailto:72210297@dankook.ac.kr)
