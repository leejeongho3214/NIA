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

### Architecture

- **Backbone**: ResNet-50
- The output size of the final fc layer equals the number of grades for the task
- A **separate model** is trained for each task (wrinkles, pores, dryness, etc.)

### Loss Function

- Cross-entropy overfits due to grade imbalance
- → We use **Focal Loss** or **Class-balanced Loss** instead

### Training Setup

| Item | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.005 |
| Epochs | 100 |
| Train / Val / Test | 8 : 1 : 1 |
| Split strategy | Stratified split based on grade distribution |

---

## 🚀 Getting Started

### 1. Download Resources

| Resource | Link | Location |
| --- | --- | --- |
| Raw facial images & labels (AI-Hub) | [Link](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71645) | `dataset/img`, `dataset/label` |
| Model checkpoints | [Link](https://gofile.me/7wbhv/FpFhxyCZX) | `checkpoint/` |
| Dataset split JSON | [Link](https://gofile.me/7wbhv/hpeVMNNOg) | `dataset/split` |

> 🔑 To get the **password** for the archives, please contact us by email → [72210297@dankook.ac.kr](mailto:72210297@dankook.ac.kr)
>
> The split JSON files contain random 8:1:1 splits stratified by grade for each facial sign, provided as four sets (**seeds 1–4**).

> ℹ️ The raw data is hosted on [AI-Hub](https://www.aihub.or.kr). Access may require an AI-Hub account and a data-use application; please check AI-Hub's policy for overseas users.

### 2. Directory Structure

```
{$ROOT}
├── checkpoint
│   ├── class
│   └── regression
│       └── 1st_cnn
│           └── save_model
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

The CNN expects square inputs. Since the bboxes in the original JSON fit tightly around each region, they must be resized to squares, using either:

- **Option 1**: a square crop centered on the bbox
- **Option 2**: zero-padding the bbox

```bash
python tool/img_crop.py
```

### 4. Training

```bash
# Predict dermatologist visual grades
python tool/main.py --name "CHECKPOINT_NAME" --mode class

# Predict instrument-measured values
python tool/main.py --name "CHECKPOINT_NAME" --mode regression
```

### 5. Testing

```bash
python tool/test.py --name "CHECKPOINT_NAME" --mode class
python tool/test.py --name "CHECKPOINT_NAME" --mode regression
```

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
