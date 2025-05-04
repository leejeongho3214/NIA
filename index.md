<p align="center">
  <img src="assets/dku.png" height="200", width = "200">
  <img src="assets/dku_hos.svg" height="200", width = "200">
  <img src="assets/iec.jpg" height="200", width = "200">
  <img src="assets/kairos.png" height="200", width = "200">
</p>

<h1 align="center">👋 한국인 피부상태 AI 데이터셋</h1>

이 프로젝트는 한국인을 대상으로 한 피부 이미지 데이터셋과 안면 피부 상태 평가를 위한 AI 모델을 제공합니다.  
10~60대 이상의 남녀 1,100명을 대상으로 수집한 다각도 이미지와 함께, 피부과 전문의의 육안 평가와 정밀 기기 측정값이 포함되어 있습니다.

---

## 📂 주요 링크

- 📊 [AI-Hub 데이터 다운로드](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71645)
- 📄 [논문 1 - 2024 KCC (DBpia)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11862094)
- 📄 [논문 2 - 2024 KSC (DBpia)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12041791)
- 📬 [이메일 문의](mailto:72210297@dankook.ac.kr)

---

## 🧠 간략 소개

- **장비**: 디지털 카메라, 스마트폰, 태블릿
- **촬영 각도**: 최대 7가지
- **BBox 라벨**: 얼굴 주요 부위 8개
- **평가 정보**:
  - 육안 평가 (전문의 5인)
  - 기기 측정값 (SCI 논문, 식약처 인증 장비)

---

## 💻 코드 실행 예시

```bash
python tool/main.py --name "my_checkpoint" --mode class       # 육안평가
python tool/main.py --name "my_checkpoint" --mode regression  # 기기 측정값
```

---

## 📊 결과 예시

<p align="center">
  <img src="assets/table1.png" width="700">
</p>

<p align="center">
  <img src="assets/figure2.png" width="700">
</p>

---

## 🛠 폴더 구조

```
project_root/
│
├── dataset/
│   ├── img/
│   ├── label/
│   └── cropped_img/
│
└── tool/
    ├── img_crop.py
    ├── main.py
    └── test.py
```

---

## 👤 Maintainer

- **이정호 (Jeongho Lee)**  
  단국대학교 컴퓨터학과 박사과정  
  📧 [72210297@dankook.ac.kr](mailto:72210297@dankook.ac.kr)
