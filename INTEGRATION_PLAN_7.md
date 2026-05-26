# 방안 7 구현 계획: 탐지 결과를 학습 데이터로 활용

## 목표

TACO 탐지 모델로 탐지한 결과를 자동으로 수집하여 분류 모델의 학습 데이터로 활용

## 핵심 문제 해결: 새로운 클래스 처리

### 해결 전략: 하이브리드 접근 (방안 D)

1. **자동 매핑**: TACO 클래스를 기존 클래스로 매핑
2. **새 클래스 식별**: 매핑 불가능한 경우 새 클래스 후보로 분류
3. **사용자 확인**: 웹 UI에서 승인/거부/수정
4. **자동 저장**: 승인된 클래스로 자동 저장

---

## 구현 단계

### Phase 1: 기본 구조

#### 1.1 클래스 매핑 모듈 (`class_mapper.py`)
- ✅ TACO → 분류 클래스 매핑
- ✅ 새 클래스 후보 관리
- ✅ 사용자 정의 매핑 저장

#### 1.2 데이터 수집 모듈
- 탐지 결과에서 객체 영역 추출
- 클래스 매핑 적용
- 이미지 저장

#### 1.3 웹 UI 확장
- 새 클래스 승인 페이지
- 클래스 관리 페이지

---

### Phase 2: 워크플로우

```
1. 사용자가 이미지 업로드
   ↓
2. TACO 탐지 모델로 객체 탐지
   ↓
3. 각 객체별로:
   a. 클래스 매핑 시도
   b. 매핑 성공 → 해당 클래스 폴더에 저장
   c. 매핑 실패 → 새 클래스 후보로 분류
   ↓
4. 새 클래스 후보는 웹 UI에 표시
   ↓
5. 사용자가 승인/거부/수정
   ↓
6. 승인된 클래스는 자동으로 폴더 생성 및 저장
```

---

## 데이터 저장 구조

```
data/garbage_dataset/
├── battery/              # 기존 클래스
├── plastic/              # 기존 클래스
├── metal/                # 기존 클래스
├── ...                   # 기존 클래스들
├── can/                  # 새로 추가된 클래스 (예시)
├── bottle/               # 새로 추가된 클래스 (예시)
├── pending/              # 승인 대기 중인 이미지
│   ├── new_class_1/
│   └── new_class_2/
└── unknown/              # 매핑 불가능한 낮은 신뢰도 이미지
```

---

## 클래스 매핑 전략

### 자동 매핑 규칙

1. **명확한 매핑** (즉시 저장)
   - Can → metal
   - Bottle → plastic
   - Glass bottle → glass
   - 등

2. **유사도 기반 매핑** (신뢰도 확인)
   - 클래스 이름 유사도 계산
   - 신뢰도 > 0.8: 자동 저장
   - 신뢰도 0.5-0.8: 사용자 확인
   - 신뢰도 < 0.5: unknown 폴더

3. **새 클래스 후보** (사용자 확인 필요)
   - 매핑 불가능한 경우
   - 정규화된 이름으로 새 클래스 생성
   - 사용자 승인 대기

---

## 웹 UI 추가 기능

### 1. 새 클래스 승인 페이지 (`/admin/pending-classes`)

```html
- 승인 대기 중인 클래스 목록
- 각 클래스별:
  * TACO 원본 클래스 이름
  * 제안된 클래스 이름
  * 이미지 개수
  * 샘플 이미지 미리보기
  * 승인/거부/이름 수정 버튼
```

### 2. 클래스 관리 페이지 (`/admin/classes`)

```html
- 현재 모든 클래스 목록
- 클래스별 이미지 개수
- 클래스 통계
- 클래스 삭제/병합 기능
```

### 3. 결과 페이지에 새 클래스 알림

```html
- 새 클래스가 발견되면 알림 표시
- "관리자 페이지에서 확인" 링크
```

---

## 구현 예시 코드

### 데이터 수집 함수

```python
def collect_detection_data(detections, image_path, mapper):
    """탐지 결과를 학습 데이터로 수집"""
    image = Image.open(image_path)
    results = []
    
    for i, detection in enumerate(detections):
        # 객체 영역 추출
        box = detection['box']
        cropped = image.crop((box['x1'], box['y1'], box['x2'], box['y2']))
        
        # 클래스 매핑
        taco_class = detection['class_name']
        mapped_class, status, needs_review = mapper.map_class(
            taco_class, 
            detection['confidence']
        )
        
        # 저장 경로 결정
        if status == 'mapped' and not needs_review:
            # 즉시 저장
            save_path = save_to_class(cropped, mapped_class)
        elif status == 'new_candidate' or needs_review:
            # 승인 대기
            save_path = save_to_pending(cropped, mapped_class, taco_class)
        else:
            # unknown
            save_path = save_to_unknown(cropped)
        
        results.append({
            'original_class': taco_class,
            'mapped_class': mapped_class,
            'status': status,
            'needs_review': needs_review,
            'saved_path': save_path
        })
    
    return results
```

---

## 장점

1. **자동화**: 대부분의 경우 자동으로 처리
2. **확장성**: 새 클래스 자동 추가 가능
3. **품질 보장**: 사용자 확인으로 잘못된 데이터 방지
4. **지속적 개선**: 더 많은 데이터 수집으로 모델 개선

---

## 주의사항

1. **데이터 품질**: 잘못 매핑된 데이터가 학습에 포함되지 않도록 주의
2. **클래스 불균형**: 특정 클래스의 데이터가 너무 많아지지 않도록 모니터링
3. **저장 공간**: 이미지가 계속 누적되므로 주기적 정리 필요
4. **성능**: 대량의 이미지 처리 시 성능 고려

---

## 다음 단계

1. `class_mapper.py` 구현 완료
2. 데이터 수집 모듈 구현
3. 웹 UI 관리 페이지 추가
4. 테스트 및 개선

