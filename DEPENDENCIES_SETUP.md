# 의존성 설치 가이드

## 개요

이 프로젝트는 Flask 웹 애플리케이션과 TACO 탐지 모델을 통합하여 사용합니다. 필요한 모든 의존성을 설치하는 방법을 안내합니다.

## 필수 의존성

### 1. 기본 웹 애플리케이션
- Flask: 웹 프레임워크
- Werkzeug: WSGI 유틸리티
- TensorFlow/Keras: 딥러닝 모델

### 2. TACO 탐지 모델
- scikit-image: 이미지 처리
- pycocotools: COCO API
- imgaug: 이미지 증강
- matplotlib: 시각화

## 설치 방법

### 방법 1: requirements.txt 사용 (권장)

```bash
# 가상환경 활성화 (예: myproject)
c:/venvs/myproject/Scripts/activate

# 프로젝트 디렉토리로 이동
cd C:\projets\PythonProject

# 모든 의존성 설치
pip install -r requirements.txt
```

### 방법 2: 단계별 설치

```bash
# 1. 기본 웹 애플리케이션
pip install Flask==2.3.3 Werkzeug==2.3.7

# 2. 딥러닝 프레임워크
pip install tensorflow==2.13.0

# 3. 수치 연산
pip install "numpy>=1.24.0,<2.0.0" scipy>=1.9.0

# 4. 이미지 처리
pip install Pillow>=10.0.0 scikit-image>=0.20.0

# 5. TACO 탐지 모델
pip install pycocotools>=2.0.0 imgaug>=0.4.0

# 6. 기타
pip install pandas>=1.5.0 matplotlib>=3.5.0 requests>=2.28.0
```

## 중요: NumPy 버전 호환성

### 문제
- `imgaug`는 NumPy 2.x와 호환되지 않습니다
- NumPy 2.x에서 `np.sctypes`가 제거되어 `imgaug`가 오류를 발생시킵니다

### 해결
- **NumPy 1.x 사용 필수**: `numpy>=1.24.0,<2.0.0`
- 현재 `requirements.txt`에 이미 설정되어 있습니다

### NumPy 버전 확인
```bash
pip show numpy
```

### NumPy 다운그레이드 (필요시)
```bash
# NumPy 2.x가 설치된 경우
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall

# 또는 특정 버전
pip install numpy==1.26.3
```

## 설치 확인

### 1. 패키지 확인
```bash
pip list | findstr -i "flask tensorflow numpy scikit-image pycocotools imgaug"
```

### 2. Python에서 확인
```python
import flask
import tensorflow as tf
import numpy as np
import skimage
import pycocotools
import imgaug

print(f"Flask: {flask.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print(f"scikit-image: {skimage.__version__}")
```

### 3. TACO detector import 테스트
```python
# taco_detector.py가 있는 디렉토리에서
from taco_detector import TacoDetector
print("✅ TACO detector 모듈 import 성공")
```

## 일반적인 문제 해결

### 문제 1: NumPy 버전 오류
```
AttributeError: `np.sctypes` was removed in the NumPy 2.0 release
```

**해결**:
```bash
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

### 문제 2: pycocotools 설치 실패 (Windows)
```
error: Microsoft Visual C++ 14.0 is required
```

**해결**:
```bash
# 사전 컴파일된 wheel 사용
pip install pycocotools-windows
# 또는
pip install --only-binary :all: pycocotools
```

### 문제 3: TensorFlow 버전 호환성
```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.13.0
```

**해결**:
```bash
# 더 유연한 버전 지정
pip install "tensorflow>=2.13.0"
```

### 문제 4: imgaug 설치 실패
```
ERROR: Could not find a version that satisfies the requirement imgaug
```

**해결**:
```bash
# 최신 버전 설치
pip install imgaug --upgrade
# 또는 특정 버전
pip install imgaug==0.4.0
```

## 개발 환경 설정

### 가상환경 사용 (권장)

```bash
# 가상환경 생성
python -m venv c:/venvs/myproject

# 가상환경 활성화 (Windows)
c:/venvs/myproject/Scripts/activate

# 의존성 설치
pip install -r requirements.txt
```

### 현재 가상환경 확인
```bash
# 가상환경 활성화 스크립트 예시
@echo off
cd C:\projets\PythonProject
set FLASK_APP=app
set FLASK_DEBUG=true
c:/venvs/myproject/Scripts/activate
```

## 의존성 업데이트

### 모든 패키지 업데이트
```bash
pip install --upgrade -r requirements.txt
```

### 특정 패키지 업데이트
```bash
pip install --upgrade tensorflow
```

## 요약

| 패키지 | 용도 | 중요도 |
|--------|------|--------|
| Flask | 웹 프레임워크 | 필수 |
| TensorFlow | 딥러닝 모델 | 필수 |
| NumPy (1.x) | 수치 연산 | 필수 (버전 중요) |
| scikit-image | 이미지 처리 | 필수 (TACO) |
| pycocotools | COCO API | 필수 (TACO) |
| imgaug | 이미지 증강 | 필수 (TACO) |
| matplotlib | 시각화 | 선택 |
| pandas | 데이터 처리 | 선택 |

## 다음 단계

의존성 설치가 완료되면:
1. TACO 탐지 모델 테스트
2. 웹 애플리케이션 실행
3. 탐지 모드 통합

