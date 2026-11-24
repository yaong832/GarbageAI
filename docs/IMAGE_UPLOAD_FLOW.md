# 이미지 업로드 시 동작 흐름

## 현재 구현 상태

### 기존 카테고리 (8개)
- `battery`, `biological`, `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`

---

## 시나리오 1: 기존 카테고리에 있는 경우 ✅

### 예시: 플라스틱 병 이미지 업로드

#### 단계별 동작:

```
1. 사용자가 이미지 업로드
   파일: "plastic_bottle.jpg"
   ↓
2. 파일 검증
   ✅ 파일 형식 확인 (jpg, png, jpeg, gif, bmp)
   ✅ 파일명 보안 처리 (secure_filename)
   ↓
3. static/uploads/에 저장
   경로: static/uploads/plastic_bottle.jpg
   용도: 웹에서 이미지 표시용
   ↓
4. 이미지 전처리
   - 크기 조정: 224x224 픽셀
   - 정규화: 0~255 → 0~1 범위
   ↓
5. 모델 예측
   입력: 전처리된 이미지 배열
   출력: 8개 클래스에 대한 확률
   
   예측 결과 예시:
   - plastic: 87.5%
   - glass: 8.2%
   - metal: 2.1%
   - paper: 1.0%
   - ... (나머지 클래스들)
   ↓
6. 최종 예측 결정
   predicted_class = "plastic" (가장 높은 확률)
   confidence = 87.5%
   ↓
7. 학습 데이터 저장 여부 확인
   신뢰도 체크: 87.5% >= 50.0% ✅
   → 저장 진행
   ↓
8. 학습 데이터로 저장
   대상 폴더: data/garbage_dataset/plastic/
   파일명: plastic_20241113_143052_123.jpg
   (형식: {class}_{timestamp}_{milliseconds}.{ext})
   ↓
9. 웹 결과 페이지 표시
   - 예측 클래스: plastic (플라스틱)
   - 신뢰도: 87.5%
   - 처리 방법 안내 표시
   - 전체 예측 결과 표시
   ↓
10. 완료
    ✅ static/uploads/plastic_bottle.jpg (웹 표시용)
    ✅ data/garbage_dataset/plastic/plastic_20241113_143052_123.jpg (학습용)
```

#### 콘솔 출력:
```
✅ 학습 데이터로 저장: data/garbage_dataset/plastic/plastic_20241113_143052_123.jpg (신뢰도: 87.5%)
```

#### 파일 구조:
```
프로젝트/
├── static/
│   └── uploads/
│       └── plastic_bottle.jpg          ← 웹 표시용
└── data/
    └── garbage_dataset/
        └── plastic/
            └── plastic_20241113_143052_123.jpg  ← 학습 데이터
```

---

## 시나리오 2: 신뢰도가 낮은 경우 (50% 미만) ⚠️

### 예시: 불명확한 이미지 업로드

#### 단계별 동작:

```
1-6. (시나리오 1과 동일)
   ↓
7. 학습 데이터 저장 여부 확인
   신뢰도 체크: 45.2% < 50.0% ❌
   → 저장하지 않음
   ↓
8. 웹 결과 페이지 표시
   - 예측 클래스: trash (일반 쓰레기) - 가장 높은 확률이지만 낮음
   - 신뢰도: 45.2%
   - 처리 방법 안내 표시
   ↓
9. 완료
   ✅ static/uploads/image.jpg (웹 표시용만)
   ❌ 학습 데이터로 저장하지 않음 (신뢰도 낮음)
```

#### 콘솔 출력:
```
⚠️ 신뢰도가 낮아 학습 데이터로 저장하지 않음: trash (45.2%)
```

#### 파일 구조:
```
프로젝트/
├── static/
│   └── uploads/
│       └── image.jpg          ← 웹 표시용만
└── data/
    └── garbage_dataset/
        └── (저장되지 않음)
```

---

## 시나리오 3: 기존 카테고리에 없는 경우 (향후 구현 예정) 🔮

### 현재 상태
**현재는 모든 예측이 8개 클래스 중 하나로만 나옵니다.**
- 모델이 학습된 클래스만 예측 가능
- 새로운 클래스는 자동으로 "trash" 또는 가장 유사한 클래스로 분류됨

### 향후 방안 7 구현 시 (TACO 탐지 모델 통합)

#### 예시: TACO에서 "Can" 클래스 탐지

```
1. 사용자가 이미지 업로드
   파일: "soda_can.jpg"
   ↓
2. TACO 탐지 모델 실행
   탐지 결과: "Can" (TACO 클래스)
   신뢰도: 82.3%
   ↓
3. 클래스 매핑 시도 (class_mapper.py)
   TACO 클래스: "Can"
   → 매핑 테이블 확인
   → "Can" → "metal" 매핑 발견 ✅
   ↓
4. 매핑된 클래스로 저장
   대상 폴더: data/garbage_dataset/metal/
   파일명: metal_20241113_143052_123.jpg
   ↓
5. 웹 결과 표시
   - 예측 클래스: metal (금속)
   - 원본 TACO 클래스: Can
   - 신뢰도: 82.3%
```

#### 예시: TACO에서 새로운 클래스 탐지 (매핑 불가능)

```
1. 사용자가 이미지 업로드
   파일: "unknown_object.jpg"
   ↓
2. TACO 탐지 모델 실행
   탐지 결과: "UnknownObject" (새로운 TACO 클래스)
   신뢰도: 75.0%
   ↓
3. 클래스 매핑 시도
   TACO 클래스: "UnknownObject"
   → 매핑 테이블에 없음 ❌
   → 새 클래스 후보로 분류
   ↓
4. 새 클래스 후보 저장
   대상 폴더: data/garbage_dataset/pending/unknownobject/
   파일명: unknownobject_20241113_143052_123.jpg
   상태: 승인 대기 중
   ↓
5. 웹 결과 표시
   - 예측 클래스: unknownobject (새 클래스 후보)
   - 알림: "새로운 클래스가 발견되었습니다. 관리자 페이지에서 확인하세요."
   ↓
6. 관리자 승인 페이지에서
   - 새 클래스 목록 확인
   - 승인/거부/이름 수정
   - 승인 시: data/garbage_dataset/unknownobject/로 이동
   - 거부 시: 기존 클래스로 재매핑
```

---

## 현재 코드의 동작 요약

### ✅ 구현된 기능

1. **이미지 업로드 및 저장**
   - `static/uploads/`에 저장 (웹 표시용)

2. **모델 예측**
   - 8개 클래스 중 하나로 예측
   - 신뢰도 계산

3. **학습 데이터 자동 저장**
   - 신뢰도 50% 이상인 경우
   - 예측된 클래스 폴더에 저장
   - 파일명: `{class}_{timestamp}_{milliseconds}.{ext}`

4. **웹 결과 표시**
   - 예측 클래스 및 신뢰도
   - 전체 예측 결과
   - 처리 방법 안내

### 🔮 향후 구현 예정 (방안 7)

1. **TACO 탐지 모델 통합**
   - 객체 탐지 및 클래스 식별

2. **새 클래스 처리**
   - `class_mapper.py`로 매핑
   - 새 클래스 후보 관리
   - 사용자 승인 시스템

3. **하이브리드 처리**
   - 탐지 → 매핑 → 저장
   - 매핑 불가능 시 승인 대기

---

## 실제 동작 예시

### 예시 1: 플라스틱 병 (기존 클래스, 높은 신뢰도)

**입력**: `plastic_bottle.jpg`

**처리**:
1. `static/uploads/plastic_bottle.jpg` 저장
2. 모델 예측: `plastic` (87.5%)
3. `data/garbage_dataset/plastic/plastic_20241113_143052_123.jpg` 저장 ✅

**결과 페이지**:
- 클래스: plastic (플라스틱)
- 신뢰도: 87.5%
- 처리 방법: 재활용 (플라스틱류)

---

### 예시 2: 불명확한 이미지 (기존 클래스, 낮은 신뢰도)

**입력**: `unclear_image.jpg`

**처리**:
1. `static/uploads/unclear_image.jpg` 저장
2. 모델 예측: `trash` (45.2%)
3. 신뢰도 50% 미만 → 저장하지 않음 ❌

**결과 페이지**:
- 클래스: trash (일반 쓰레기)
- 신뢰도: 45.2%
- 처리 방법: 일반 쓰레기

---

### 예시 3: 금속 캔 (향후 TACO 통합 시)

**입력**: `soda_can.jpg`

**처리** (향후):
1. `static/uploads/soda_can.jpg` 저장
2. TACO 탐지: `Can` (82.3%)
3. 매핑: `Can` → `metal`
4. `data/garbage_dataset/metal/metal_20241113_143052_123.jpg` 저장 ✅

**결과 페이지**:
- 클래스: metal (금속)
- 원본 TACO 클래스: Can
- 신뢰도: 82.3%
- 처리 방법: 재활용 (캔류)

---

## 설정 변경 가능 항목

### `app.py`에서 변경 가능:

```python
# 최소 신뢰도 임계값 (기본: 50%)
MIN_CONFIDENCE_FOR_SAVE = 50.0

# 예: 70% 이상만 저장하려면
MIN_CONFIDENCE_FOR_SAVE = 70.0

# 예: 모든 예측을 저장하려면
MIN_CONFIDENCE_FOR_SAVE = 0.0
```

---

## 데이터 수집 통계 확인

학습 데이터가 쌓이면:

```bash
# 각 클래스별 이미지 개수 확인
data/garbage_dataset/
├── battery/        (예: 150개)
├── biological/     (예: 200개)
├── cardboard/      (예: 180개)
├── glass/          (예: 170개)
├── metal/          (예: 190개)
├── paper/          (예: 210개)
├── plastic/        (예: 250개)  ← 가장 많이 수집됨
└── trash/          (예: 160개)
```

주기적으로 모델 재훈련:
```bash
python gar_refine.py
```

---

## 요약

| 상황 | 웹 표시 | 학습 데이터 저장 | 비고 |
|------|---------|-----------------|------|
| 기존 클래스, 신뢰도 ≥50% | ✅ | ✅ | 정상 저장 |
| 기존 클래스, 신뢰도 <50% | ✅ | ❌ | 신뢰도 낮아 저장 안 함 |
| 새 클래스 (향후) | ✅ | ⏳ | 승인 대기 후 저장 |

