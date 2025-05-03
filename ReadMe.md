# 한국인 피부상태 측정 데이터
[![Dataset in AI-Hub](https://img.shields.io/badge/Dataset%20in-AI--Hub-blue)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71645)
[![2024 KCC](https://img.shields.io/badge/Paper%201-2024_KCC-red)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11862094)
[![2024 KSC](https://img.shields.io/badge/Paper%202-2024_KSC-orange)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12041791)

## 소개
- 최초로 한국인을 대상으로 수집한 피부상태 측정 데이터
- 10~60대 이상의 남/여를 일정한 비율로 총 1,100명으로 구성
- 3개의 촬영장비(디지털 카메라, 스마트패드, 스마트폰)로 촬영
- 최대 7가지 촬영로 다각도 이미지를 촬영
- 모든 이미지마다 8가지 주요 영역 bbox를 제공
- 피부과 전문의 육안평가와 정밀 기기측정값을 제공

## 데이터 셋
### 이미지
- 디지털 카메라
    - 7가지 각도로 촬영
- 스마트 패드 & 폰
    - 3가지 각도로 촬영

### 라벨링
- 전문의 육안평가
    - 국내 피부과 전문의 교수 5인이 육안으로 색소침착, 입술건조도, 모공, 턱선 처짐, 주름을 평가
    - 항목마다 등급의 범위가 상이함
- 정밀 측정장비 기기값
    - SCI급 논문에 주로 인용되거나 식약처에서 인증받은 장비를 활용하여 모공, 색소침착, 주름, 수분, 탄력을 측정

### 실험환경
- 모든 참여자는 동일하게 세면한 뒤, 항온항습실에서 건조 후 촬영
- 디지털 카메라의 경우, 암막실에서 조명이 통제되고 자체 제작한 거치대에 얼굴을 고정하여 각도를 일정하게 변경해주며 촬영
- 나머지는 흰색 배경에서 의자에 앉아 촬영

### 관련 논문
- [우수발표논문 수상] 이정호 외 5명, "다중뷰 안면 영역열 이미지를 이용한 피부평가 AI"
- [우수논문 수상] 이정호 외 5명, "Transformer-CNN 기반 하이브리드 딥러닝 모델을 활용한 안면 피부 평가"
- 해외 저널 JEADV 준비중...

## 피부진단 AI 모델
### 모델
- ResNet-50을 사용
- 맨 마지막 fc-layer의 최종 출력값만 예측하고자 하는 등급의 범위만큼 길이로 수정
- 육안평가 task의 경우, 총 5개 모델을 선언하여 학습하였고 동일한 피부진단 항목의 이미지를 함께 학습
    - 주름: 이마, 미간, 눈가
    - 모공: 이마, 볼
    - 색소침착: 볼
    - 건조도: 입술
    - 처짐: 턱 <br><br>

<p align="middle">
    <img src="assets/figure1.png", width="2000" height="400">
</p>

### 손실 함수
- Cross-Entropy로 할 경우, 등급 불균형으로 인해 모델 과적합이 발생
- Focal Loss와 Class-balanced Loss를 사용

### 학습 방법
- Adam optimizer
- Learning rate 0.005
- 100 epoch
- Train/ Val/ Test를 8:1:1로 구성
- 육안평가의 경우, 등급별로 8:1:1 비율로 나눠주고 다시 합치는 방식으로 데이터 셋을 구성
- 기기 측정값의 경우, 측정값을 정렬한 뒤에 앞 순서 8개는 train, 뒤 순서는 1개씩 val, test로 나눠주고 다시 합침

### 결과
<p align="middle">
    <img src="assets/table1.png", width="=900" height="350">
</p>
<p align="middle">
    <img src="assets/figure2.png", width="1000" height="400">
</p>

## 코드
### 데이터 셋 제작
기본적으로 CNN 모델의 입력 이미지는 정사각형의 형태임. 입력으로 넣기 위해, resize를 바로 하게 되면 이미지 왜곡이 발생함. 그래서 아래 코드에서는 데이터셋에서 주어진 영역의 bbox의 센터값을 기준으로 정사각형으로 영역 이미지를 획득함

다른 방법으로는 주어진 영역 이미지(직사각형)에서 0-padding을 주어 정사각형을 만드는 것도 하나의 방법임.
```
python tool/img_crop.py
```

### 폴더
```
{$ROOT}
|-- dataset
|    ㄴㅡ img
|    ㄴㅡ label
|    ㄴㅡ cropped_img
|-- tool

```
### 학습
mode는 따로 입력하지 않으면 "육안평가"가 되고 mode를 regression을 입력하면 "정밀 기기측정값" 예측이 된다
```
python tool/main.py --name "체크포인트 이름" --mode "class" or "regression"
```

### 검증
```
python tool/test.py --name "앞서 저장한 체크포인트 이름" --mode "class" or "regression"
```

## 문의
단국대학교 컴퓨터학과 박사과정 이정호(72210297@dankook.ac.kr)에게 메일 보내주세요
