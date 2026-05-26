# 새로운 클래스 처리 방안 (방안 7 구현 시)

## 문제 상황

- **기존 분류 모델 클래스**: battery, biological, cardboard, glass, metal, paper, plastic, trash (8개)
- **TACO 탐지 모델 클래스**: Can, Bottle, Plastic bag, Cup, Lid, Straw, Cigarette, Other 등 (다양함)
- **문제**: TACO에서 탐지된 클래스가 기존 분류 모델에 없을 수 있음

---

## 해결 방안

### 방안 A: 지능형 클래스 매핑 + 새 클래스 추가 ⭐⭐⭐⭐ (추천)

**개념**: 
1. TACO 클래스를 기존 클래스로 매핑 시도
2. 매핑 불가능한 경우 새 클래스로 추가
3. 사용자 확인 후 최종 결정

**동작 방식**:
```
TACO 탐지 결과
  ↓
클래스 매핑 시도
  ├─ 매핑 가능 → 기존 클래스로 저장
  └─ 매핑 불가능 → 새 클래스 후보로 분류
      ↓
사용자 확인 (웹 UI)
  ├─ 승인 → 새 클래스로 추가
  └─ 거부/수정 → 기존 클래스로 재매핑
```

**장점**:
- ✅ 자동화 + 사용자 제어
- ✅ 점진적 클래스 확장
- ✅ 데이터 품질 보장

**단점**:
- ⚠️ 사용자 확인 단계 필요

---

### 방안 B: 자동 클래스 매핑 테이블 ⭐⭐⭐

**개념**: TACO 클래스를 기존 클래스로 자동 매핑

**매핑 규칙**:
```python
TACO_TO_CLASSIFICATION_MAP = {
    # 명확한 매핑
    'Can': 'metal',
    'Bottle': 'plastic',
    'Plastic bag + wrapper': 'plastic',
    'Cup': 'plastic',
    'Glass bottle': 'glass',
    'Paper cup': 'paper',
    'Paper bag': 'paper',
    'Cardboard': 'cardboard',
    'Battery': 'battery',
    
    # 유사한 것들
    'Food waste': 'biological',
    'Disposable food container': 'plastic',
    
    # 불확실한 것들
    'Other': 'trash',
    'Unlabeled litter': 'trash',
    'Cigarette': 'trash',
    # ... 기타
}
```

**장점**:
- ✅ 완전 자동화
- ✅ 즉시 사용 가능
- ✅ 구현 간단

**단점**:
- ⚠️ 매핑 오류 가능성
- ⚠️ 새 클래스 추가 불가

---

### 방안 C: 신뢰도 기반 자동 처리 ⭐⭐⭐

**개념**: 신뢰도에 따라 자동 처리

**동작 방식**:
1. **높은 신뢰도 (>80%)**: 자동으로 기존 클래스에 매핑 또는 새 클래스 추가
2. **중간 신뢰도 (50-80%)**: 사용자 확인 요청
3. **낮은 신뢰도 (<50%)**: "unknown" 폴더에 저장, 나중에 검토

**장점**:
- ✅ 효율적인 자동화
- ✅ 불확실한 데이터 필터링

**단점**:
- ⚠️ 신뢰도 임계값 조정 필요

---

### 방안 D: 하이브리드 접근 (매핑 + 새 클래스) ⭐⭐⭐⭐⭐ (가장 추천)

**개념**: 자동 매핑 + 새 클래스 자동 추가 + 사용자 확인

**동작 방식**:

#### 1단계: 자동 매핑 시도
```python
# 명확한 매핑이 있는 경우
if taco_class in KNOWN_MAPPINGS:
    target_class = KNOWN_MAPPINGS[taco_class]
    confidence = 'high'
```

#### 2단계: 새 클래스 후보 식별
```python
# 매핑 불가능한 경우
if taco_class not in KNOWN_MAPPINGS:
    # 새 클래스로 추가 (임시)
    new_class = taco_class.lower().replace(' ', '_')
    confidence = 'medium'
    requires_review = True
```

#### 3단계: 사용자 확인 (웹 UI)
- 새 클래스 후보 목록 표시
- 사용자가 승인/거부/수정 가능
- 승인된 클래스는 자동으로 폴더 생성

#### 4단계: 데이터 저장
```python
# 승인된 클래스로 저장
if approved:
    save_to_class(approved_class)
else:
    save_to_review_folder()
```

**장점**:
- ✅ 자동화 + 유연성
- ✅ 점진적 클래스 확장
- ✅ 데이터 품질 보장
- ✅ 사용자 제어

**단점**:
- ⚠️ 초기 설정 필요

---

### 방안 E: "unknown" 클래스 활용 ⭐⭐

**개념**: 매핑 불가능한 것은 "unknown" 폴더에 저장

**동작 방식**:
1. 매핑 가능 → 기존 클래스
2. 매핑 불가능 → `data/garbage_dataset/unknown/` 폴더
3. 주기적으로 검토하여 새 클래스 생성

**장점**:
- ✅ 간단한 구현
- ✅ 데이터 손실 없음

**단점**:
- ⚠️ 수동 검토 필요
- ⚠️ "unknown" 폴더 관리 필요

---

### 방안 F: 유사도 기반 자동 분류 ⭐⭐⭐

**개념**: TACO 클래스와 기존 클래스의 유사도를 계산하여 자동 매핑

**동작 방식**:
1. TACO 클래스 이름과 기존 클래스 이름 유사도 계산
2. 유사도가 높으면 자동 매핑
3. 낮으면 새 클래스 후보로 분류

**장점**:
- ✅ 지능형 자동화
- ✅ 새 클래스 자동 식별

**단점**:
- ⚠️ 유사도 계산 로직 필요
- ⚠️ 오매핑 가능성

---

## 추천 구현: 방안 D (하이브리드 접근)

### 구현 구조

```python
# 1. 클래스 매핑 테이블
KNOWN_MAPPINGS = {
    'Can': 'metal',
    'Bottle': 'plastic',
    'Plastic bag + wrapper': 'plastic',
    # ... 기타
}

# 2. 새 클래스 관리
NEW_CLASSES = []  # 승인 대기 중인 새 클래스

# 3. 처리 로직
def process_detection_result(taco_class, confidence):
    # 매핑 시도
    if taco_class in KNOWN_MAPPINGS:
        return KNOWN_MAPPINGS[taco_class], 'mapped'
    
    # 새 클래스 후보
    new_class = normalize_class_name(taco_class)
    return new_class, 'new_candidate'
```

### 웹 UI 추가 기능

1. **새 클래스 승인 페이지**
   - 새로 발견된 클래스 목록
   - 승인/거부/수정 버튼
   - 클래스 이름 변경 가능

2. **클래스 관리 페이지**
   - 현재 클래스 목록
   - 클래스 통계
   - 클래스 삭제/병합 기능

---

## 구현 예시 코드 구조

### 1. 클래스 매핑 모듈
```python
# class_mapper.py
class ClassMapper:
    def __init__(self):
        self.known_mappings = {...}
        self.pending_classes = []
    
    def map_class(self, taco_class):
        # 매핑 로직
        pass
    
    def add_new_class(self, taco_class, user_approved_name):
        # 새 클래스 추가
        pass
```

### 2. 데이터 수집 모듈
```python
# data_collector.py
class DataCollector:
    def collect_from_detection(self, detections, image):
        for detection in detections:
            taco_class = detection['class']
            mapped_class, status = mapper.map_class(taco_class)
            
            if status == 'new_candidate':
                # 사용자 확인 대기
                save_to_pending(mapped_class)
            else:
                # 즉시 저장
                save_to_class(mapped_class)
```

### 3. 웹 UI 관리 페이지
```html
<!-- admin.html -->
- 새 클래스 승인 목록
- 클래스별 이미지 개수
- 클래스 통계
```

---

## 단계별 구현 계획

### Phase 1: 기본 매핑
1. 클래스 매핑 테이블 생성
2. 자동 매핑 로직 구현
3. 매핑 불가능한 경우 "unknown" 폴더에 저장

### Phase 2: 새 클래스 관리
1. 새 클래스 후보 식별
2. 웹 UI에 승인 페이지 추가
3. 승인된 클래스 자동 폴더 생성

### Phase 3: 고급 기능
1. 클래스 통계 대시보드
2. 클래스 병합 기능
3. 자동 클래스 정리

---

## 선택 가이드

### 빠른 구현이 필요한 경우
→ **방안 B** (자동 클래스 매핑 테이블)

### 유연성이 중요한 경우
→ **방안 D** (하이브리드 접근) ⭐ 추천

### 완전 자동화가 필요한 경우
→ **방안 C** (신뢰도 기반 자동 처리)

### 간단한 시작이 필요한 경우
→ **방안 E** (unknown 클래스 활용)

