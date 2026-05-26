# 쓰레기 분류 모델 훈련 가이드

## 개요

이 프로젝트는 **MobileNetV2 전이학습(Transfer Learning)**을 사용하여 쓰레기 분류 모델을 훈련합니다.

## 모델 훈련 방법

### 방법 1: 처음부터 모델 훈련 (`gar.py`) ⭐

**용도**: 새로운 모델을 처음부터 훈련하거나, 데이터셋이 크게 변경된 경우

**실행 방법**:
```bash
python gar.py
```

**특징**:
- MobileNetV2를 ImageNet 가중치로 초기화
- 2단계 학습 전략:
  1. **1단계 (15 에포크)**: 사전 학습된 레이어 고정, 분류 레이어만 학습
  2. **2단계 (15 에포크)**: 상위 레이어 일부 해제하여 Fine-tuning
- 총 30 에포크 (Early Stopping으로 조기 종료 가능)
- 학습률: 0.0001 → 0.00001 (2단계에서 감소)

**데이터 구조 요구사항**:
```
data/garbage_dataset/
├── battery/
├── biological/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

**출력**:
- `model/garbage_model.keras`: 훈련된 모델

---

### 방법 2: 기존 모델 추가 학습 (`gar_refine.py`) ⭐⭐

**용도**: 기존 모델을 더 많은 데이터로 개선하거나, 정확도를 향상시키고 싶은 경우

**실행 방법**:
```bash
python gar_refine.py
```

**특징**:
- 기존 모델을 로드하여 추가 학습
- 모든 레이어를 학습 가능하도록 설정 (Fine-tuning)
- 매우 낮은 학습률 (0.00001)로 미세 조정
- 기존 모델 자동 백업 (`garbage_model_backup.keras`)
- 20 에포크 (Early Stopping으로 조기 종료 가능)

**장점**:
- 기존 모델의 지식을 유지하면서 개선
- 더 빠른 수렴
- 과적합 위험 감소

---

## 훈련 설정

### 기본 설정 (`gar.py`)

```python
IMG_SIZE = 224          # 이미지 크기
BATCH_SIZE = 32         # 배치 크기
EPOCHS = 30             # 총 에포크 수
LEARNING_RATE = 0.0001  # 초기 학습률
VALIDATION_SPLIT = 0.2  # 검증 데이터 비율 (20%)
```

### 데이터 증강 (Data Augmentation)

**학습 데이터**:
- 회전: ±30도
- 이동: 가로/세로 ±30%
- 전단: ±20%
- 확대/축소: ±30%
- 좌우/상하 반전
- 밝기 조정: 80-120%

**검증 데이터**:
- 정규화만 수행 (증강 없음)

---

## 훈련 프로세스

### 1단계: 데이터 준비

```bash
# 데이터 구조 확인
data/garbage_dataset/
├── battery/        # 배터리 이미지들
├── biological/     # 생물학적 쓰레기 이미지들
├── cardboard/      # 골판지 이미지들
├── glass/          # 유리 이미지들
├── metal/          # 금속 이미지들
├── paper/          # 종이 이미지들
├── plastic/        # 플라스틱 이미지들
└── trash/          # 일반 쓰레기 이미지들
```

### 2단계: 모델 훈련

```bash
# 처음부터 훈련
python gar.py

# 또는 기존 모델 추가 학습
python gar_refine.py
```

### 3단계: 모델 확인

훈련이 완료되면:
- `model/garbage_model.keras`: 최종 모델
- 콘솔에 정확도 및 손실 정보 출력

---

## 모델 아키텍처

```
Input (224x224x3)
    ↓
MobileNetV2 (사전 학습, ImageNet 가중치)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(512, ReLU) + Dropout(0.5)
    ↓
BatchNormalization
    ↓
Dense(256, ReLU) + Dropout(0.3)
    ↓
Dense(8, Softmax)  # 8개 클래스
```

---

## 콜백 함수

### EarlyStopping
- 모니터: `val_loss`
- Patience: 7 (gar.py) / 10 (gar_refine.py)
- 최적 가중치 복원

### ModelCheckpoint
- 모니터: `val_accuracy`
- 최고 성능 모델만 저장

### ReduceLROnPlateau
- 모니터: `val_loss`
- 학습률 자동 감소 (정체 시)

---

## 훈련 결과 확인

훈련 완료 후 콘솔에 출력되는 정보:
- 학습 샘플 수
- 검증 샘플 수
- 최종 학습 정확도
- 최종 검증 정확도
- 최종 학습 손실
- 최종 검증 손실
- Top-3 정확도

---

## 주의사항

1. **데이터 불균형**: 각 클래스별 이미지 수가 비슷한지 확인
2. **GPU 사용**: GPU가 있으면 자동으로 사용 (TensorFlow)
3. **메모리**: 배치 크기를 조정하여 메모리 부족 방지
4. **백업**: `gar_refine.py`는 자동으로 기존 모델 백업

---

## 문제 해결

### 문제 1: "데이터 디렉토리를 찾을 수 없습니다"
**해결**: `data/garbage_dataset/` 폴더가 존재하는지 확인

### 문제 2: "모델 파일을 찾을 수 없습니다" (gar_refine.py)
**해결**: 먼저 `python gar.py`로 기본 모델을 훈련

### 문제 3: 메모리 부족
**해결**: `BATCH_SIZE`를 줄이기 (예: 32 → 16)

### 문제 4: 훈련이 너무 느림
**해결**: 
- GPU 사용 확인
- `EPOCHS` 수 줄이기
- 데이터 증강 범위 줄이기

---

## 새로운 데이터로 모델 개선하기

### 시나리오: 탐지 결과를 학습 데이터로 활용 (방안 7)

1. **데이터 수집**: TACO 탐지 결과를 `data/garbage_dataset/`에 저장
2. **클래스 매핑**: `class_mapper.py`로 새 클래스 처리
3. **모델 재훈련**: 
   ```bash
   # 기존 모델에 새 데이터로 추가 학습
   python gar_refine.py
   ```

### 자동화 워크플로우

```
1. 웹 앱에서 탐지 모드 사용
   ↓
2. 탐지 결과를 클래스별로 자동 저장
   ↓
3. 주기적으로 모델 재훈련
   python gar_refine.py
   ↓
4. 개선된 모델로 웹 앱 업데이트
```

---

## 성능 최적화 팁

1. **데이터 품질**: 잘못 분류된 이미지 제거
2. **데이터 증강**: 다양한 각도, 조명 조건의 이미지 추가
3. **하이퍼파라미터 튜닝**: 학습률, 배치 크기 조정
4. **정기적 재훈련**: 새로운 데이터로 주기적으로 모델 업데이트

---

## 요약

| 방법 | 스크립트 | 용도 | 학습 시간 |
|------|---------|------|----------|
| 처음부터 훈련 | `gar.py` | 새 모델 생성 | 길음 (~30 에포크) |
| 추가 학습 | `gar_refine.py` | 기존 모델 개선 | 짧음 (~20 에포크) |

**추천**: 처음에는 `gar.py`로 모델을 만들고, 이후에는 `gar_refine.py`로 점진적으로 개선

