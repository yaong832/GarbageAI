# 쓰레기 분류 AI 프로젝트

이미지를 업로드하여 8가지 쓰레기 종류를 자동으로 분류하는 웹 애플리케이션입니다.

## 기능

- 🗑️ 8가지 쓰레기 종류 분류
  - 배터리 (Battery)
  - 생물학적 쓰레기 (Biological)
  - 골판지 (Cardboard)
  - 유리 (Glass)
  - 금속 (Metal)
  - 종이 (Paper)
  - 플라스틱 (Plastic)
  - 일반 쓰레기 (Trash)

- 🎨 모던한 웹 인터페이스
- 📊 전체 클래스 예측 확률 표시
- 📱 반응형 디자인

## 설치 방법

1. 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. 패키지 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 모델 학습

데이터셋을 준비한 후 모델을 학습시킵니다:

```bash
python gar.py
```

**데이터 구조:**
```
data/
└── garbage_dataset/
    ├── battery/
    ├── biological/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

각 폴더에 해당 클래스의 이미지 파일들을 넣어주세요.

### 2. 웹 애플리케이션 실행

```bash
python app.py
```

브라우저에서 `http://localhost:5000` 접속

## 프로젝트 구조

```
PythonProject/
├── app.py                 # Flask 웹 애플리케이션
├── gar.py                 # 모델 학습 스크립트
├── convert_model.py       # 모델 형식 변환 스크립트
├── requirements.txt       # 패키지 의존성
├── data/                  # 데이터셋 폴더
├── model/                 # 학습된 모델 저장
├── static/
│   └── uploads/          # 업로드된 이미지 저장
└── templates/
    ├── index.html        # 메인 페이지
    └── result.html       # 결과 페이지
```

## 기술 스택

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML, CSS, JavaScript
- **AI**: CNN (Convolutional Neural Network)

## 라이선스

이 프로젝트는 개인 사용 목적으로 제작되었습니다.

