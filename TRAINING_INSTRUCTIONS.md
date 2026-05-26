# 모델 학습 가이드

## 개요

업로드된 이미지들이 `data/garbage_dataset/` 폴더에 클래스별로 저장되어 있습니다. 이 데이터를 사용하여 모델을 학습시킬 수 있습니다.

## 학습 방법

### 방법 1: 처음부터 모델 학습 (gar.py)

**용도**: 새로운 모델을 처음부터 만들거나, 데이터가 크게 변경된 경우

**실행 방법**:
```bash
# 가상환경 활성화
c:/venvs/myproject/Scripts/activate

# 프로젝트 디렉토리로 이동
cd C:\projets\PythonProject

# 모델 학습 실행
python gar.py
```

**특징**:
- MobileNetV2 전이학습 사용
- 2단계 학습 전략:
  1. 1단계 (15 에포크): 사전 학습 레이어 고정, 분류 레이어만 학습
  2. 2단계 (15 에포크): 상위 레이어 일부 해제하여 Fine-tuning
- 총 30 에포크 (Early Stopping으로 조기 종료 가능)
- 학습률: 0.0001 → 0.00001 (2단계에서 감소)

**소요 시간**: GPU 사용 시 약 30분~1시간, CPU 사용 시 수 시간

---

### 방법 2: 기존 모델에 추가 학습 (gar_refine.py) ⭐ **권장**

**용도**: 기존 모델을 새로 수집한 데이터로 개선하고 싶은 경우

**실행 방법**:
```bash
# 가상환경 활성화
c:/venvs/myproject/Scripts/activate

# 프로젝트 디렉토리로 이동
cd C:\projets\PythonProject

# 기존 모델에 추가 학습
python gar_refine.py
```

**특징**:
- 기존 모델을 로드하여 추가 학습
- 모든 레이어를 학습 가능하도록 설정 (Fine-tuning)
- 매우 낮은 학습률 (0.00001)로 미세 조정
- 기존 모델 자동 백업 (`garbage_model_backup.keras`)
- 20 에포크 (Early Stopping으로 조기 종료 가능)

**소요 시간**: GPU 사용 시 약 10~20분, CPU 사용 시 1~2시간

**장점**:
- 기존 모델의 지식을 유지하면서 개선
- 더 빠른 수렴
- 과적합 위험 감소

---

## 단계별 가이드

### 1단계: 데이터 확인

학습 전에 데이터가 제대로 수집되었는지 확인:

```bash
# 각 클래스별 이미지 개수 확인
dir data\garbage_dataset\battery
dir data\garbage_dataset\plastic
# ... 등등
```

또는 Python으로 확인:
```python
import os
data_dir = 'data/garbage_dataset'
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{class_name}: {count}개")
```

### 2단계: 모델 학습 실행

#### 처음부터 학습하는 경우:
```bash
python gar.py
```

#### 기존 모델 개선하는 경우:
```bash
python gar_refine.py
```

### 3단계: 학습 진행 상황 확인

학습 중 콘솔에 다음과 같은 정보가 표시됩니다:

```
[데이터 로딩 중...]
[OK] 학습 샘플: 1,234장
[OK] 검증 샘플: 308장

[모델 구축 중 (MobileNetV2 전이학습)...]

Epoch 1/30
1234/1234 [==============================] - 45s 36ms/step - loss: 1.2345 - accuracy: 0.5678 - val_loss: 1.1234 - val_accuracy: 0.6123
...
```

### 4단계: 학습 완료 확인

학습이 완료되면:

```
[학습 완료!]
최종 학습 정확도: 0.9234
최종 검증 정확도: 0.9012
최종 학습 손실: 0.1234
최종 검증 손실: 0.1456
모델 저장 위치: model/garbage_model.keras
```

### 5단계: 웹 앱 재시작

학습된 모델을 사용하려면 웹 앱을 재시작:

```bash
# 웹 앱 중지 (Ctrl+C)
# 다시 시작
flask run
# 또는
python app.py
```

---

## 학습 설정 변경

### 학습 파라미터 조정

`gar.py` 또는 `gar_refine.py` 파일을 열어서 설정을 변경할 수 있습니다:

```python
# gar.py 또는 gar_refine.py에서
BATCH_SIZE = 32        # 배치 크기 (메모리에 따라 조정)
EPOCHS = 30            # 에포크 수
LEARNING_RATE = 0.0001 # 학습률
VALIDATION_SPLIT = 0.2 # 검증 데이터 비율 (20%)
```

### 메모리 부족 시

배치 크기를 줄이세요:
```python
BATCH_SIZE = 16  # 또는 8
```

---

## 데이터 불균형 해결

클래스별 이미지 개수가 크게 다르면:

1. **데이터 증강 강화**: `gar.py`에서 증강 범위 증가
2. **클래스 가중치 사용**: 손실 함수에 클래스 가중치 추가
3. **데이터 수집**: 부족한 클래스의 이미지 더 수집

---

## 자주 묻는 질문

### Q: 얼마나 많은 이미지가 필요하나요?

**A**: 클래스당 최소 50~100개 이상 권장. 더 많을수록 좋습니다.

### Q: 학습 시간이 얼마나 걸리나요?

**A**: 
- GPU 사용: 10~30분 (추가 학습) / 30분~1시간 (처음부터)
- CPU 사용: 1~2시간 (추가 학습) / 수 시간 (처음부터)

### Q: 언제 추가 학습을 해야 하나요?

**A**: 
- 새 이미지가 100개 이상 수집되었을 때
- 주기적으로 (예: 매주)
- 모델 정확도가 떨어졌을 때

### Q: 기존 모델이 덮어씌워지나요?

**A**: `gar_refine.py`는 자동으로 백업을 생성합니다:
- `model/garbage_model.keras` (새 모델)
- `model/garbage_model_backup.keras` (이전 모델)

### Q: 학습 중 오류가 발생하면?

**A**: 
- 메모리 부족: `BATCH_SIZE` 줄이기
- 파일 오류: 데이터 폴더 구조 확인
- 모델 로드 오류: 모델 파일 경로 확인

---

## 권장 워크플로우

### 초기 설정
1. 처음 모델 학습: `python gar.py`
2. 웹 앱 실행하여 이미지 수집

### 정기적 개선
1. 이미지 100개 이상 수집될 때마다
2. `python gar_refine.py` 실행
3. 웹 앱 재시작
4. 정확도 향상 확인

### 주기적 재학습
- 매주 또는 매월
- 데이터가 충분히 쌓였을 때
- 모델 성능이 저하되었을 때

---

## 학습 결과 확인

### 콘솔 출력 예시

```
[학습 완료!]
최종 학습 정확도: 0.9234 (92.34%)
최종 검증 정확도: 0.9012 (90.12%)
최종 학습 손실: 0.1234
최종 검증 손실: 0.1456
Top-3 정확도: 0.9876 (98.76%)
모델 저장 위치: model/garbage_model.keras
```

### 좋은 결과 기준

- **검증 정확도**: 85% 이상
- **검증 손실**: 0.3 이하
- **학습/검증 차이**: 5% 이하 (과적합 방지)

---

## 다음 단계

학습이 완료되면:

1. ✅ 웹 앱 재시작
2. ✅ 테스트 이미지로 정확도 확인
3. ✅ 필요시 추가 학습 반복
4. ✅ 정기적으로 모델 개선

---

## 빠른 시작 명령어

```bash
# 1. 가상환경 활성화
c:/venvs/myproject/Scripts/activate

# 2. 프로젝트 디렉토리로 이동
cd C:\projets\PythonProject

# 3. 기존 모델에 추가 학습 (권장)
python gar_refine.py

# 4. 학습 완료 후 웹 앱 재시작
flask run
```

