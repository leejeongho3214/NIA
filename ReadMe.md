<div align="center">

# 🇰🇷 한국인 피부상태 측정 데이터

**최초의 한국인 피부상태 AI 데이터셋 & 피부 진단 AI 모델**

[![Project Page](https://img.shields.io/badge/Project-Homepage-brightgreen)](https://leejeongho3214.github.io/NIA)
[![Dataset in AI-Hub](https://img.shields.io/badge/Dataset-AI--Hub-blue)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71645)
[![Paper - SRT 2026](https://img.shields.io/badge/Paper-SRT_2026_(SCIE)-b31b1b)](https://doi.org/10.1111/srt.70375)
[![Contact](https://img.shields.io/badge/Contact-Email-informational?logo=gmail)](mailto:72210297@dankook.ac.kr)

</div>

---

## 📢 대표 논문

> **Artificial Intelligence Based Skin Analysis Models for Predicting Visual Grades and Device Measured Physiological Values From Facial Images**
>
> Eunyoung Lee, **Jeongho Lee**, Nahee Kim, Junchae Na, Byungcheol Park, Sang-Il Choi
> *Skin Research and Technology*, Vol. 32, No. 9, e70375 (2026) · Open Access (CC BY 4.0)
> 📄 [https://doi.org/10.1111/srt.70375](https://doi.org/10.1111/srt.70375)

본 저장소의 데이터셋과 모델을 기반으로, 얼굴 이미지로부터 **전문의 육안 등급**과 **기기 측정 생리값**을 동시에 예측하는 AI 프레임워크를 제안한 연구입니다. 한국인 1,099명(14–69세), 3가지 촬영 장비 환경에서 검증되었습니다.

---

## 🆕 업데이트

| 날짜 | 내용 |
| --- | --- |
| **26/09/01** | 🎉 *Skin Research and Technology* (SCIE) 논문 게재 — [10.1111/srt.70375](https://doi.org/10.1111/srt.70375) |
| **26/03/21** | 모델 checkpoint 및 데이터셋 공유 링크 재업로드 |
| **25/12/03** | 모델 체크포인트 및 데이터셋 분할 파일 제공, 학습·테스트 코드 수정 |

---

## 📌 소개

- **최초의 한국인 피부상태 AI 데이터셋**
- **참여자**: 남녀 1,100명, 연령 10대~60대 이상
- **촬영 장비 3종**: 디지털 카메라(DSLR), 스마트패드, 스마트폰
- **최대 7가지 각도**의 다각도 얼굴 이미지
- 모든 이미지에 **8개 주요 얼굴 영역의 BBox** 포함
- **전문의 육안 평가 + 정밀 기기 측정값** 동시 제공

---

## 🗂️ 데이터 구성

### 📷 이미지

| 장비 | 촬영 각도 | 비고 |
| --- | --- | --- |
| 디지털 카메라 | 7 각도 | 암막실, 얼굴 고정 장치 사용 |
| 스마트패드 | 3 각도 | 배경·조명 조건 통제 |
| 스마트폰 | 3 각도 | 배경·조명 조건 통제 |

### 🏷️ 라벨링

**① 전문의 육안 평가 (`class`)**

- 국내 피부과 전문의 5인 참여
- 평가 항목: 색소침착, 입술건조도, 모공, 턱선처짐, 주름 등
- 항목별 등급 범위 상이

**② 정밀 측정 장비값 (`regression`)**

- SCI급 논문 및 식약처 인증 기반 장비 사용
- 측정 항목: 모공, 색소침착, 주름, 수분, 탄력

### 🧪 실험 환경

- 세면 후 **항온·항습실**에서 건조 → 촬영
- 디지털 카메라 촬영은 **암막실**에서 얼굴 고정 장치를 활용

---

## 🧠 피부 진단 AI 모델

### 모델 구조

- **Backbone**: ResNet-50
- 마지막 fc-layer 출력 크기 = 해당 task의 등급 수
- Task(주름, 모공, 건조도 등)별로 **분리된 모델**을 각각 학습

### 손실 함수

- Cross-Entropy는 등급 불균형으로 과적합이 발생
- → **Focal Loss** 또는 **Class-balanced Loss** 사용

### 학습 설정

| 항목 | 값 |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.005 |
| Epoch | 100 |
| Train / Val / Test | 8 : 1 : 1 |
| Split 방식 | 등급 분포를 고려한 stratified split |

---

## 🚀 시작하기

### 1. 리소스 내려받기

| 리소스 | 링크 | 배치 위치 |
| --- | --- | --- |
| 원본 얼굴 이미지 · 라벨 | [AI-Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71645) | `dataset/img`, `dataset/label` |
| 모델 checkpoint | [gofile.me/7wbhv/TaZgLsAag](https://gofile.me/7wbhv/TaZgLsAag) | `checkpoint/` |
| 데이터셋 분할 json | [gofile.me/7wbhv/cstOyfCWw](https://gofile.me/7wbhv/cstOyfCWw) | `dataset/split` |

> 🔑 압축 파일 **Password는 이메일로 문의**해 주세요. → [72210297@dankook.ac.kr](mailto:72210297@dankook.ac.kr)
>
> 분할 json은 각 facial sign별로 등급을 기준으로 8:1:1을 랜덤 분할한 결과이며, **Seed 1–4** 총 4세트로 구성되어 있습니다.

### 2. 폴더 구조

```
{$ROOT}
├── checkpoint
│   ├── class
│   └── regression
│       └── 1st_cnn
│           └── save_model
│
├── dataset
│   ├── img            # AI-Hub 원본 이미지
│   ├── label          # AI-Hub 라벨(json)
│   ├── split          # train/val/test 분할 json
│   └── cropped_img    # img_crop.py 출력
│
└── tool
    ├── img_crop.py
    ├── main.py
    └── test.py
```

### 3. 전처리 — 이미지 Crop

CNN 입력을 위해 정사각형 이미지가 필요합니다. 원본 json의 bbox는 영역에 딱 맞는 크기이므로 정사각형으로 재조정해야 합니다.

- **방법 1**: bbox 중심 기준 정사각형 crop
- **방법 2**: bbox에 zero-padding 추가

```bash
python tool/img_crop.py
```

### 4. 학습

```bash
# 전문의 육안 평가 등급 예측
python tool/main.py --name "저장할 체크포인트 이름" --mode class

# 기기 측정값 예측
python tool/main.py --name "저장할 체크포인트 이름" --mode regression
```

### 5. 테스트

```bash
python tool/test.py --name "저장된 체크포인트 이름" --mode class
python tool/test.py --name "저장된 체크포인트 이름" --mode regression
```

---

## 📚 Citation

본 데이터셋 또는 코드를 사용하실 경우 아래 논문을 인용해 주세요.

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

## 📰 관련 발표 논문

| 연도 | 학회/논문지 | 제목 링크 | 비고 |
| --- | --- | --- | --- |
| 2026 | Skin Research and Technology (SCIE) | [바로가기](https://doi.org/10.1111/srt.70375) | 대표 논문 |
| 2025 | 정보과학회 컴퓨팅의 실제 논문지 (KTCP) | [바로가기](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12252203) | — |
| 2024 | 정보과학회 KSC | [바로가기](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12041791) | 🏆 우수논문상 |
| 2024 | 정보과학회 KCC | [바로가기](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11862094) | 🏅 우수발표논문상 |

---

## 📬 문의

> 단국대학교 컴퓨터학과 박사과정
> **이정호 (Jeongho Lee)**
> 📧 [72210297@dankook.ac.kr](mailto:72210297@dankook.ac.kr)
