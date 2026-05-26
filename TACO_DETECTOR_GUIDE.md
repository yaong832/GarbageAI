# TACO 탐지 모델 래퍼 사용 가이드

## 개요

`taco_detector.py`는 TACO 프로젝트의 Mask R-CNN 탐지 모델을 PythonProject에서 쉽게 사용할 수 있도록 래핑한 모듈입니다.

## 주요 기능

1. **모델 로드**: 학습된 TACO 모델 가중치 로드
2. **객체 탐지**: 이미지에서 쓰레기 객체 탐지
3. **결과 파싱**: 탐지 결과를 사용하기 쉬운 형식으로 변환
4. **객체 크롭**: 탐지된 객체 영역 자동 크롭

## 사용 방법

### 1. 기본 사용

```python
from taco_detector import create_detector

# 탐지 모델 생성
detector = create_detector(
    model_path='path/to/mask_rcnn_taco.h5',  # 모델 가중치 파일
    class_map='map_10.csv',  # 클래스 맵 파일
    detection_min_confidence=0.5  # 최소 신뢰도
)

# 이미지 탐지
results = detector.detect('image.jpg')

# 결과 확인
print(f"탐지된 객체 수: {results['num_detections']}")
for detection in results['detections']:
    print(f"- {detection['class_name']}: {detection['confidence']:.2%}")
    print(f"  바운딩 박스: {detection['bbox']}")
```

### 2. 객체 크롭 및 저장

```python
# 탐지 및 크롭
cropped_images = detector.detect_and_crop(
    'image.jpg',
    output_dir='output/cropped',  # 저장 디렉토리
    min_confidence=0.5
)

# 크롭된 이미지 정보
for img_info in cropped_images:
    print(f"클래스: {img_info['class_name']}")
    print(f"신뢰도: {img_info['confidence']:.2%}")
    print(f"저장 경로: {img_info['saved_path']}")
```

### 3. 고급 사용

```python
from taco_detector import TacoDetector

# 직접 인스턴스 생성
detector = TacoDetector(
    model_path='path/to/model.h5',
    class_map_path='path/to/map_10.csv',
    detection_min_confidence=0.6
)

# 모델 로드
detector.load_model()

# 탐지 수행
results = detector.detect('image.jpg', return_masks=True)

# 마스크 정보 포함
for detection in results['detections']:
    if 'mask' in detection:
        mask = detection['mask']  # 마스크 배열
```

## 반환 형식

### `detect()` 메서드 반환값

```python
{
    'detections': [
        {
            'class_name': 'Can',  # 클래스 이름
            'class_id': 1,  # 클래스 ID
            'confidence': 0.85,  # 신뢰도 (0.0 ~ 1.0)
            'bbox': [100, 50, 200, 150],  # [y1, x1, y2, x2]
            'bbox_xyxy': [50, 100, 150, 200],  # [x1, y1, x2, y2]
            'mask': np.array(...)  # 마스크 (return_masks=True인 경우)
        },
        ...
    ],
    'num_detections': 3,  # 탐지된 객체 개수
    'image_shape': (480, 640),  # (height, width)
    'raw_results': {...}  # 원본 결과 (디버깅용)
}
```

### `detect_and_crop()` 메서드 반환값

```python
[
    {
        'class_name': 'Can',
        'confidence': 0.85,
        'bbox': [100, 50, 200, 150],
        'cropped_image': np.array(...),  # 크롭된 이미지 배열
        'saved_path': 'output/cropped/Can_20241113_143052_123_0.jpg'  # 저장 경로
    },
    ...
]
```

## 클래스 맵

TACO는 다양한 원본 클래스를 더 적은 수의 클래스로 매핑합니다.

예시 (`map_10.csv`):
- `Drink can` → `Can`
- `Clear plastic bottle` → `Bottle`
- `Garbage bag` → `Plastic bag + wrapper`
- 등등...

## 모델 경로

### 기본 모델 경로
```
C:\Taco\detector\models\mask_rcnn_taco.h5
```

### 모델이 없는 경우
1. COCO 사전 학습 가중치 다운로드
2. TACO 데이터셋으로 모델 학습
3. 학습된 모델 가중치 사용

## 주의사항

1. **TACO 프로젝트 경로**: 기본값은 `C:\Taco`입니다. 다른 경로를 사용하려면 `taco_dir` 파라미터를 지정하세요.

2. **의존성**: TACO detector 모듈이 필요합니다. 다음이 설치되어 있어야 합니다:
   - TensorFlow/Keras
   - NumPy
   - Pillow
   - pycocotools
   - imgaug (NumPy 1.x 필요)

3. **모델 가중치**: 모델 파일이 없으면 탐지를 수행할 수 없습니다.

## 오류 해결

### ImportError: TACO detector 모듈을 import할 수 없습니다
- TACO 프로젝트 경로 확인
- `taco_detector.py`의 `TACO_DIR` 변수 수정

### FileNotFoundError: 모델 파일을 찾을 수 없습니다
- 모델 경로 확인
- 모델 학습 필요

### NumPy 버전 오류
- NumPy 1.x 사용 (2.x는 imgaug와 호환되지 않음)

## 다음 단계

이제 `taco_detector.py`를 사용하여:
1. 웹 앱에 탐지 모드 추가
2. 탐지 결과를 학습 데이터로 변환
3. 새 클래스 자동 식별

