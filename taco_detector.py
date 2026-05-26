"""
TACO 탐지 모델 래퍼
TACO 프로젝트의 Mask R-CNN 모델을 사용하여 쓰레기 객체 탐지를 수행합니다.
"""

import os
import sys
import numpy as np
from PIL import Image
import csv

# TACO 프로젝트 경로 추가 (detector 모듈 import를 위해)
TACO_DIR = r'C:\Taco'
if TACO_DIR not in sys.path:
    sys.path.insert(0, TACO_DIR)
    sys.path.insert(0, os.path.join(TACO_DIR, 'detector'))

try:
    from detector.model import MaskRCNN
    from detector.config import Config
    from detector import utils
except ImportError as e:
    print(f"⚠️ TACO detector 모듈을 import할 수 없습니다: {e}")
    print(f"TACO 디렉토리 경로를 확인하세요: {TACO_DIR}")
    MaskRCNN = None
    Config = None
    utils = None


class TacoDetector:
    """TACO 탐지 모델 래퍼 클래스"""
    
    def __init__(self, 
                 model_path=None,
                 class_map_path=None,
                 taco_dir=None,
                 detection_min_confidence=0.5):
        """
        Args:
            model_path: 학습된 모델 가중치 파일 경로
            class_map_path: 클래스 매핑 CSV 파일 경로 (예: map_10.csv)
            taco_dir: TACO 프로젝트 디렉토리 경로
            detection_min_confidence: 탐지 최소 신뢰도 (0.0 ~ 1.0)
        """
        if MaskRCNN is None:
            raise ImportError("TACO detector 모듈을 로드할 수 없습니다.")
        
        self.taco_dir = taco_dir or TACO_DIR
        self.model_path = model_path
        self.class_map_path = class_map_path
        self.detection_min_confidence = detection_min_confidence
        self.model = None
        self.config = None
        self.class_names = None
        self.class_map = None
        
        # 클래스 맵 로드
        if class_map_path:
            self.load_class_map(class_map_path)
        
        # 모델 초기화
        if model_path:
            self.load_model(model_path)
    
    def load_class_map(self, class_map_path):
        """클래스 매핑 파일 로드"""
        if not os.path.exists(class_map_path):
            # TACO 디렉토리 기준으로 시도
            taco_map_path = os.path.join(self.taco_dir, 'detector', 'taco_config', class_map_path)
            if os.path.exists(taco_map_path):
                class_map_path = taco_map_path
            else:
                raise FileNotFoundError(f"클래스 맵 파일을 찾을 수 없습니다: {class_map_path}")
        
        self.class_map = {}
        with open(class_map_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    self.class_map[row[0]] = row[1]
        
        print(f"✅ 클래스 맵 로드 완료: {len(self.class_map)}개 매핑")
        return self.class_map
    
    def create_config(self, num_classes=None):
        """탐지용 설정 생성"""
        if num_classes is None:
            # 클래스 맵에서 고유한 클래스 개수 계산
            if self.class_map:
                unique_classes = set(self.class_map.values())
                num_classes = len(unique_classes) + 1  # +1 for background
            else:
                num_classes = 2  # 기본값: background + 1 class
        
        class TacoInferenceConfig(Config):
            NAME = "taco"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = num_classes
            DETECTION_MIN_CONFIDENCE = self.detection_min_confidence
            USE_OBJECT_ZOOM = False
        
        self.config = TacoInferenceConfig()
        return self.config
    
    def load_model(self, model_path=None):
        """탐지 모델 로드"""
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            raise ValueError("모델 경로가 지정되지 않았습니다.")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # 설정 생성
        if self.config is None:
            self.create_config()
        
        # 모델 생성
        print(f"🔄 TACO 탐지 모델 로드 중: {self.model_path}")
        self.model = MaskRCNN(mode="inference", config=self.config, model_dir=os.path.dirname(self.model_path))
        
        # 가중치 로드
        try:
            self.model.load_weights(self.model_path, self.model_path, by_name=True)
            print(f"✅ 모델 로드 완료!")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
        
        # 클래스 이름 설정
        if self.class_map:
            # 매핑된 클래스 이름 목록 생성
            unique_classes = sorted(set(self.class_map.values()))
            self.class_names = ['BG'] + unique_classes  # BG = Background
        else:
            self.class_names = ['BG', 'Litter']  # 기본값
        
        return self.model
    
    def detect(self, image_path_or_array, return_masks=True):
        """
        이미지에서 객체 탐지 수행
        
        Args:
            image_path_or_array: 이미지 파일 경로 또는 numpy 배열
            return_masks: 마스크 반환 여부
        
        Returns:
            dict: 탐지 결과
                - 'detections': 탐지된 객체 리스트
                    - 'class_name': 클래스 이름
                    - 'class_id': 클래스 ID
                    - 'confidence': 신뢰도 (0.0 ~ 1.0)
                    - 'bbox': 바운딩 박스 [y1, x1, y2, x2]
                    - 'mask': 마스크 (선택사항)
                - 'num_detections': 탐지된 객체 개수
                - 'image_shape': 이미지 크기 (height, width)
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        # 이미지 로드
        if isinstance(image_path_or_array, str):
            image = np.array(Image.open(image_path_or_array))
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array
        else:
            raise TypeError("이미지는 파일 경로(str) 또는 numpy 배열이어야 합니다.")
        
        # 이미지가 RGB가 아니면 변환
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # 탐지 수행
        results = self.model.detect([image], verbose=0)[0]
        
        # 결과 파싱
        detections = []
        num_detections = len(results['rois'])
        
        for i in range(num_detections):
            class_id = results['class_ids'][i]
            confidence = float(results['scores'][i])
            bbox = results['rois'][i].tolist()  # [y1, x1, y2, x2]
            
            # 클래스 이름 가져오기
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"Class_{class_id}"
            
            detection = {
                'class_name': class_name,
                'class_id': int(class_id),
                'confidence': confidence,
                'bbox': bbox,
                'bbox_xyxy': [bbox[1], bbox[0], bbox[3], bbox[2]],  # [x1, y1, x2, y2] 형식
            }
            
            # 마스크 추가 (선택사항)
            if return_masks and 'masks' in results:
                mask = results['masks'][:, :, i]
                detection['mask'] = mask
            
            detections.append(detection)
        
        return {
            'detections': detections,
            'num_detections': num_detections,
            'image_shape': image.shape[:2],  # (height, width)
            'raw_results': results  # 원본 결과 (디버깅용)
        }
    
    def detect_and_crop(self, image_path_or_array, output_dir=None, min_confidence=None):
        """
        이미지에서 객체를 탐지하고 각 객체 영역을 크롭하여 저장
        
        Args:
            image_path_or_array: 이미지 파일 경로 또는 numpy 배열
            output_dir: 크롭된 이미지 저장 디렉토리 (None이면 저장하지 않음)
            min_confidence: 최소 신뢰도 (None이면 기본값 사용)
        
        Returns:
            list: 크롭된 이미지 정보 리스트
                - 'class_name': 클래스 이름
                - 'confidence': 신뢰도
                - 'bbox': 바운딩 박스
                - 'cropped_image': 크롭된 이미지 배열
                - 'saved_path': 저장된 경로 (저장한 경우)
        """
        if min_confidence is None:
            min_confidence = self.detection_min_confidence
        
        # 탐지 수행
        results = self.detect(image_path_or_array, return_masks=False)
        
        # 이미지 로드
        if isinstance(image_path_or_array, str):
            image = np.array(Image.open(image_path_or_array))
            image_path = image_path_or_array
        else:
            image = image_path_or_array
            image_path = None
        
        # RGB 변환
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        cropped_images = []
        
        for i, detection in enumerate(results['detections']):
            if detection['confidence'] < min_confidence:
                continue
            
            # 바운딩 박스 추출
            y1, x1, y2, x2 = detection['bbox']
            y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
            
            # 이미지 경계 확인
            h, w = image.shape[:2]
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            
            # 크롭
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue
            
            cropped_info = {
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox'],
                'cropped_image': cropped,
                'saved_path': None
            }
            
            # 저장 (선택사항)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{detection['class_name']}_{timestamp}_{i}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                Image.fromarray(cropped).save(filepath)
                cropped_info['saved_path'] = filepath
            
            cropped_images.append(cropped_info)
        
        return cropped_images


def create_detector(model_path=None, class_map='map_10.csv', **kwargs):
    """
    TACO 탐지 모델 인스턴스 생성 (편의 함수)
    
    Args:
        model_path: 모델 가중치 파일 경로
        class_map: 클래스 맵 파일명 또는 경로
        **kwargs: TacoDetector 추가 인자
    
    Returns:
        TacoDetector: 탐지 모델 인스턴스
    """
    # 기본 경로 설정
    taco_dir = kwargs.get('taco_dir', TACO_DIR)
    
    # 클래스 맵 경로
    if not os.path.isabs(class_map):
        class_map_path = os.path.join(taco_dir, 'detector', 'taco_config', class_map)
    else:
        class_map_path = class_map
    
    # 모델 경로
    if model_path is None:
        # 기본 모델 경로 시도
        default_model_path = os.path.join(taco_dir, 'detector', 'models', 'mask_rcnn_taco.h5')
        if os.path.exists(default_model_path):
            model_path = default_model_path
        else:
            print(f"⚠️ 기본 모델 파일을 찾을 수 없습니다: {default_model_path}")
            print("모델 경로를 직접 지정하거나 모델을 학습시켜주세요.")
    
    detector = TacoDetector(
        model_path=model_path,
        class_map_path=class_map_path,
        **kwargs
    )
    
    if model_path:
        detector.load_model()
    
    return detector


# 사용 예시
if __name__ == '__main__':
    # 테스트
    try:
        detector = create_detector(
            model_path=None,  # 모델 경로 지정 필요
            class_map='map_10.csv',
            detection_min_confidence=0.5
        )
        print("✅ TACO 탐지 모델 초기화 완료")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")

