<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
</head>
<body style="font-family: sans-serif; line-height: 1.6; padding: 30px;">

  <!-- 상단 로고 -->
  <p align="center">
    <img src="assets/dku.png" height="200" width="200">
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <img src="assets/dku_hos.svg" height="200" width="200">
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <img src="assets/iec.jpg" height="200" width="200">
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <img src="assets/kairos.png" height="200" width="200">
  </p>

  <!-- 제목 -->
  <h1 align="center" style="color:#0066cc; font-size:42px; font-weight:bold;">
    👋 한국인 피부상태 AI 데이터셋
  </h1>

  <!-- 소개 -->
  <p align="center" style="font-size:18px;">
    이 프로젝트는 한국인을 대상으로 한 피부 이미지 데이터셋과 안면 피부 상태 평가를 위한 AI 모델을 제공합니다.<br>
    10~60대 이상의 남녀 1,100명을 대상으로 수집한 다각도 이미지와 함께, 피부과 전문의의 육안 평가와 정밀 기기 측정값이 포함되어 있습니다.
  </p>

  <hr>

  <!-- 주요 링크 -->
  <h2>📂 주요 링크</h2>
  <ul>
    <li>📊 <a href="https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71645">AI-Hub 데이터셋</a></li>
    <li>📄 <a href="https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11862094">정보과학회 2024 KCC (🏅 우수발표논문상)</a></li>
    <li>📄 <a href="https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12041791">정보과학회 2024 KSC (🏆 우수논문상)</a></li>
    <li>📬 <a href="mailto:72210297@dankook.ac.kr">이메일 문의</a></li>
  </ul>

  <hr>

  <!-- 간략 소개 -->
  <h2>🧠 간략 소개</h2>
  <ul>
    <li><strong>장비</strong>: 디지털 카메라, 스마트폰, 태블릿</li>
    <li><strong>촬영 각도</strong>: 최대 7가지</li>
    <li><strong>BBox 라벨</strong>: 얼굴 주요 부위 8개</li>
    <li><strong>평가 정보</strong>:
      <ul>
        <li>육안 평가 (전문의 5인)</li>
        <li>기기 측정값 (SCI 논문, 식약처 인증 장비)</li>
      </ul>
    </li>
  </ul>

  <hr>

  <!-- 코드 실행 예시 -->
  <h2>💻 코드 실행 예시</h2>
  <pre><code>
python tool/main.py --name "my_checkpoint" --mode class       # 육안평가
python tool/main.py --name "my_checkpoint" --mode regression  # 기기 측정값
  </code></pre>

  <hr>

  <!-- 결과 이미지 -->
  <h2>📊 결과 예시</h2>
  <p align="center">
    <img src="assets/table1.png" width="700"><br><br>
    <img src="assets/figure2.png" width="700">
  </p>

  <hr>

  <!-- 폴더 구조 -->
  <h2>🛠 폴더 구조</h2>
  <pre><code>
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
  </code></pre>

  <hr>

  <!-- 담당자 정보 -->
  <h2>👤 담당자</h2>
  <p>
    <strong>이정호 (Jeongho Lee)</strong><br>
    단국대학교 컴퓨터학과 박사과정<br>
    📧 <a href="mailto:72210297@dankook.ac.kr">72210297@dankook.ac.kr</a>
  </p>

</body>
</html>
