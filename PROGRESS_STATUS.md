# 프로젝트 진행 상황 및 TODO

## 📊 현재 진행 상황

### ✅ 완료된 작업 (Phase 1: 기본 분류 시스템)

#### 1. 기본 웹 애플리케이션
- ✅ Flask 기반 쓰레기 분류 웹 앱 (`app.py`)
- ✅ 이미지 업로드 기능
- ✅ Keras 모델 로드 및 예측
- ✅ 결과 페이지 (예측 클래스, 신뢰도, 처리 방법 안내)
- ✅ 8개 클래스 분류: battery, biological, cardboard, glass, metal, paper, plastic, trash

#### 2. 학습 데이터 자동 수집 시스템
- ✅ 이미지 업로드 시 자동으로 학습 데이터로 저장
- ✅ 신뢰도 50% 이상인 경우만 저장 (설정 가능)
- ✅ 클래스별 폴더에 자동 분류 저장
- ✅ 파일명: `{class}_{timestamp}_{milliseconds}.{ext}` 형식

#### 3. 모델 훈련 시스템
- ✅ `gar.py`: 처음부터 모델 훈련 (MobileNetV2 전이학습)
- ✅ `gar_refine.py`: 기존 모델 추가 학습
- ✅ 2단계 학습 전략 (고정 → Fine-tuning)
- ✅ Early Stopping, Model Checkpoint, Learning Rate Scheduling

#### 4. 새 클래스 처리 준비
- ✅ `class_mapper.py`: 클래스 매핑 모듈
- ✅ TACO → 분류 클래스 매핑 테이블
- ✅ 새 클래스 후보 관리 시스템
- ✅ 사용자 정의 매핑 저장/로드

#### 5. 문서화
- ✅ `TRAINING_GUIDE.md`: 모델 훈련 가이드
- ✅ `IMAGE_UPLOAD_FLOW.md`: 이미지 업로드 동작 흐름
- ✅ `INTEGRATION_PLAN_7.md`: 방안 7 구현 계획
- ✅ `NEW_CLASS_HANDLING.md`: 새 클래스 처리 방안
- ✅ `INTEGRATION_OPTIONS.md`: 통합 방안 옵션

---

## 🚧 진행 중 / 다음 단계 (Phase 2: TACO 탐지 모델 통합)

### Phase 2-1: TACO 탐지 모델 통합 준비

#### 1. TACO 탐지 모델 래퍼 생성
- ⏳ `taco_detector.py`: TACO 탐지 모델 래퍼 클래스
- ⏳ 탐지 모델 로드 및 초기화
- ⏳ 이미지에서 객체 탐지 및 마스크 생성
- ⏳ 탐지 결과 파싱 (클래스, 바운딩 박스, 신뢰도)

#### 2. 의존성 설치
- ⏳ TACO 프로젝트 의존성 확인
- ⏳ `requirements.txt` 업데이트
- ⏳ NumPy 버전 호환성 해결 (imgaug 호환)

#### 3. 탐지 모델 가중치 준비
- ⏳ COCO 사전 학습 가중치 다운로드
- ⏳ TACO 모델 학습 또는 기존 모델 사용

---

### Phase 2-2: 웹 애플리케이션 확장

#### 1. 탐지 모드 추가
- ⏳ 웹 UI에 모드 선택 추가 (분류 / 탐지)
- ⏳ 탐지 결과 표시 페이지 (`result_detection.html`)
- ⏳ 바운딩 박스 및 마스크 시각화
- ⏳ 탐지된 객체별 클래스 표시

#### 2. 탐지 결과 처리
- ⏳ 탐지 결과를 분류 데이터로 변환
- ⏳ 객체 영역 크롭 및 저장
- ⏳ `class_mapper.py`를 통한 클래스 매핑
- ⏳ 새 클래스 후보 식별 및 저장

#### 3. 하이브리드 모드 (선택사항)
- ⏳ 탐지 → 각 객체 영역 분류
- ⏳ 두 모델 결과 통합

---

### Phase 2-3: 새 클래스 관리 시스템

#### 1. 관리자 페이지
- ⏳ `/admin/pending-classes`: 승인 대기 클래스 목록
- ⏳ 새 클래스 승인/거부/수정 UI
- ⏳ 클래스별 샘플 이미지 미리보기
- ⏳ 클래스 통계 대시보드

#### 2. 클래스 관리 기능
- ⏳ `/admin/classes`: 전체 클래스 목록 및 통계
- ⏳ 클래스 삭제/병합 기능
- ⏳ 클래스별 이미지 개수 표시

#### 3. 자동화
- ⏳ 승인된 클래스 자동 폴더 생성
- ⏳ 거부된 클래스 기존 클래스로 재매핑

---

### Phase 2-4: 자동화 및 최적화

#### 1. 자동 모델 재훈련
- ⏳ 주기적 모델 재훈련 스케줄링 (예: 매주)
- ⏳ 새 데이터 임계값 도달 시 자동 재훈련
- ⏳ 재훈련 알림 시스템

#### 2. 데이터 품질 관리
- ⏳ 중복 이미지 제거
- ⏳ 잘못 분류된 이미지 검토 시스템
- ⏳ 데이터 불균형 모니터링

#### 3. 성능 최적화
- ⏳ 이미지 캐싱
- ⏳ 배치 처리 최적화
- ⏳ 모델 로드 최적화

---

## 📋 상세 TODO 리스트

### 우선순위 높음 (High Priority)

- [ ] **TACO 탐지 모델 래퍼 생성** (`taco_detector.py`)
  - [ ] TACO detector 모듈 통합
  - [ ] 탐지 모델 로드 함수
  - [ ] 이미지 탐지 함수
  - [ ] 결과 파싱 함수

- [ ] **의존성 설치 및 호환성 해결**
  - [ ] TACO requirements 확인
  - [ ] NumPy/imgaug 호환성 해결
  - [ ] requirements.txt 통합

- [ ] **탐지 모드 웹 UI 추가**
  - [ ] 모드 선택 UI (분류/탐지)
  - [ ] 탐지 결과 페이지
  - [ ] 바운딩 박스 시각화

- [ ] **탐지 결과 → 학습 데이터 변환**
  - [ ] 객체 영역 크롭
  - [ ] class_mapper를 통한 매핑
  - [ ] 클래스별 저장

### 우선순위 중간 (Medium Priority)

- [ ] **새 클래스 승인 시스템**
  - [ ] 관리자 페이지 UI
  - [ ] 승인/거부 로직
  - [ ] 클래스 통계

- [ ] **자동 모델 재훈련**
  - [ ] 스케줄링 시스템
  - [ ] 자동 실행 스크립트

### 우선순위 낮음 (Low Priority)

- [ ] **하이브리드 모드**
- [ ] **데이터 품질 관리**
- [ ] **성능 최적화**

---

## 🎯 다음 단계 (즉시 시작 가능)

### 1단계: TACO 탐지 모델 통합 준비

```bash
# 1. TACO 프로젝트 구조 확인
cd C:\Taco
# detector 모듈 확인

# 2. taco_detector.py 생성
# - TACO 모델 로드
# - 탐지 함수 구현

# 3. requirements.txt 통합
# - TACO 의존성 추가
# - NumPy 버전 조정
```

### 2단계: 웹 앱에 탐지 모드 추가

```python
# app.py에 추가
@app.route('/predict-detection', methods=['POST'])
def predict_detection():
    # TACO 탐지 실행
    # 결과 처리
    # 학습 데이터로 저장
```

### 3단계: 새 클래스 관리 UI

```python
# app.py에 추가
@app.route('/admin/pending-classes')
def admin_pending_classes():
    # 승인 대기 클래스 목록
```

---

## 📁 현재 파일 구조

```
C:\projets\PythonProject\
├── app.py                    ✅ 기본 웹 앱 (학습 데이터 자동 저장 포함)
├── class_mapper.py           ✅ 클래스 매핑 모듈
├── gar.py                    ✅ 모델 훈련 스크립트
├── gar_refine.py             ✅ 모델 추가 학습 스크립트
├── requirements.txt          ⚠️ TACO 의존성 추가 필요
├── data/
│   └── garbage_dataset/      ✅ 학습 데이터 저장 폴더
│       ├── battery/
│       ├── biological/
│       ├── ...
│       └── trash/
├── model/
│   └── garbage_model.keras   ✅ 훈련된 모델
├── static/
│   └── uploads/              ✅ 업로드된 이미지
├── templates/
│   ├── index.html            ✅ 메인 페이지
│   └── result.html           ✅ 결과 페이지
└── 문서/
    ├── TRAINING_GUIDE.md     ✅
    ├── IMAGE_UPLOAD_FLOW.md  ✅
    ├── INTEGRATION_PLAN_7.md ✅
    └── ...
```

---

## 🔄 워크플로우 (현재)

### 현재 동작 흐름

```
사용자 이미지 업로드
    ↓
static/uploads/에 저장 (웹 표시용)
    ↓
Keras 모델로 분류
    ↓
예측 결과 표시
    ↓
신뢰도 ≥50%?
    ├─ Yes → data/garbage_dataset/{class}/에 저장 ✅
    └─ No → 저장하지 않음
```

### 목표 워크플로우 (Phase 2 완료 후)

```
사용자 이미지 업로드
    ↓
모드 선택 (분류 / 탐지)
    ↓
[분류 모드]
    → Keras 모델로 분류
    → 결과 표시
    → 학습 데이터 저장

[탐지 모드]
    → TACO 모델로 탐지
    → 객체별 바운딩 박스 표시
    → 각 객체 영역 크롭
    → class_mapper로 매핑
    → 학습 데이터 저장
    → 새 클래스 발견 시 승인 대기
```

---

## ⚠️ 주의사항

1. **NumPy 버전 호환성**
   - imgaug는 NumPy 2.x와 호환되지 않음
   - NumPy 1.x 사용 필요

2. **모델 가중치**
   - TACO 모델 학습 또는 기존 가중치 필요
   - COCO 사전 학습 가중치 다운로드 필요

3. **데이터 저장 공간**
   - 이미지가 계속 누적됨
   - 주기적 정리 필요

---

## 📈 성공 지표

- [ ] 탐지 모드 정상 작동
- [ ] 새 클래스 자동 식별
- [ ] 승인 시스템 작동
- [ ] 자동 모델 재훈련
- [ ] 모델 정확도 향상

---

## 🎉 완료 기준

Phase 2 완료 시:
- ✅ 탐지 모드 사용 가능
- ✅ 탐지 결과를 학습 데이터로 자동 수집
- ✅ 새 클래스 승인 시스템 작동
- ✅ 자동 모델 재훈련 가능

---

**마지막 업데이트**: 2024-11-13
**현재 단계**: Phase 1 완료, Phase 2 시작 준비

