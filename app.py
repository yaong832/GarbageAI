"""
쓰레기 분류 웹 애플리케이션
"""

import os
import sys

# Windows cmd(cp949)에서 이모지 print 시 즉시 종료 방지
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from datetime import datetime
import shutil
import json
from PIL import Image, ImageDraw, ImageFont

# TACO 탐지 모델 (선택적 로드)
# NumPy 2.x 환경에서는 imgaug 호환성 문제로 로드하지 않음
TACO_DETECTOR_AVAILABLE = False
create_detector = None
ClassMapper = None

try:
    import numpy as np
    # NumPy 버전 체크 (1.x만 imgaug와 호환)
    np_version = np.__version__.split('.')
    np_major = int(np_version[0])
    np_minor = int(np_version[1]) if len(np_version) > 1 else 0
    
    # NumPy 1.x인 경우에만 TACO 탐지 모델 로드 시도
    if np_major == 1:
        try:
            from taco_detector import create_detector
            from class_mapper import ClassMapper
            TACO_DETECTOR_AVAILABLE = True
            print("✅ TACO 탐지 모듈 로드 완료")
        except ImportError as e:
            print(f"⚠️ TACO 탐지 모듈을 로드할 수 없습니다: {e}")
            print("⚠️ 탐지 기능은 사용할 수 없습니다. 분류 모드만 사용 가능합니다.")
        except Exception as e:
            print(f"⚠️ TACO 탐지 모듈 로드 중 오류: {e}")
            print("⚠️ 탐지 기능은 사용할 수 없습니다. 분류 모드만 사용 가능합니다.")
    else:
        print(f"⚠️ NumPy {np.__version__} 버전은 imgaug와 호환되지 않습니다.")
        print("⚠️ 탐지 기능은 사용할 수 없습니다. 분류 모드만 사용 가능합니다.")
        print("⚠️ NumPy 1.x로 다운그레이드하거나 Python 3.11 이하를 사용하세요.")
except ImportError:
    # NumPy가 없는 경우 (거의 없음)
    print("⚠️ NumPy가 설치되지 않았습니다.")
except Exception as e:
    print(f"⚠️ NumPy 버전 확인 중 오류: {e}")

# Flask 앱 초기화
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'garbage_classification_secret_key_2024')

# 설정
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
IMG_SIZE = 224
UPLOAD_FOLDER = os.path.join('static', 'uploads')
DATA_DIR = os.path.join('data', 'garbage_dataset')  # 학습 데이터 저장 경로
MODEL_PATH_KERAS = os.path.join('model', 'garbage_model.keras')
MODEL_PATH_H5 = os.path.join('model', 'garbage_model.h5')
MIN_CONFIDENCE_FOR_SAVE = 50.0  # 학습 데이터로 저장할 최소 신뢰도 (%)

CLASSES = ['battery', 'biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_NAMES_KO = {
    'battery': '배터리',
    'biological': '생물학적 쓰레기',
    'cardboard': '골판지',
    'glass': '유리',
    'metal': '금속',
    'paper': '종이',
    'plastic': '플라스틱',
    'trash': '일반 쓰레기'
}

# 각 쓰레기 종류별 처리 방법 안내
DISPOSAL_GUIDE = {
    'battery': {
        'method': '재활용 (전용 수거함)',
        'description': '배터리는 전용 수거함에 버려주세요. 일반 쓰레기와 섞이면 화재 위험이 있습니다.',
        'tips': [
            '배터리 전용 수거함에 배출',
            '배터리 양극을 테이프로 감싸서 방전 방지',
            '대형마트, 주민센터 등에 수거함 설치'
        ],
        'icon': '🔋'
    },
    'biological': {
        'method': '음식물 쓰레기',
        'description': '음식물 쓰레기 전용 봉투에 담아 배출하거나, 퇴비화하여 활용하세요.',
        'tips': [
            '음식물 쓰레기 전용 봉투 사용',
            '물기를 충분히 제거 후 배출',
            '가정용 퇴비통 활용 권장'
        ],
        'icon': '🍃'
    },
    'cardboard': {
        'method': '재활용 (종이류)',
        'description': '골판지는 종이류 재활용으로 분리 배출하세요. 깨끗하게 펼쳐서 묶어주세요.',
        'tips': [
            '이물질 제거 후 배출',
            '비닐, 테이프 등 제거',
            '비가 오는 날은 실내 보관 후 배출'
        ],
        'icon': '📦'
    },
    'glass': {
        'method': '재활용 (유리류)',
        'description': '유리병은 깨끗이 씻어서 재활용품으로 배출하세요. 깨진 유리는 일반 쓰레기입니다.',
        'tips': [
            '내용물을 깨끗이 제거',
            '라벨 제거 후 배출',
            '깨진 유리는 신문지에 싸서 일반쓰레기로'
        ],
        'icon': '🍶'
    },
    'metal': {
        'method': '재활용 (캔류)',
        'description': '금속 캔은 내용물을 비우고 깨끗이 씻어 재활용품으로 배출하세요.',
        'tips': [
            '내용물 완전히 비우기',
            '깨끗이 씻어서 배출',
            '압착하여 부피 줄이기'
        ],
        'icon': '🥫'
    },
    'paper': {
        'method': '재활용 (종이류)',
        'description': '종이는 재활용품으로 분리 배출하세요. 비닐 코팅된 종이는 일반 쓰레기입니다.',
        'tips': [
            '비닐 코팅 종이는 일반쓰레기',
            '이물질 제거 후 배출',
            '신문지, 책자 등은 묶어서 배출'
        ],
        'icon': '📄'
    },
    'plastic': {
        'method': '재활용 (플라스틱류)',
        'description': '플라스틱은 내용물을 비우고 깨끗이 씻어 재활용품으로 배출하세요.',
        'tips': [
            '내용물 완전히 비우기',
            '라벨 제거 후 배출',
            '깨끗이 씻어서 배출',
            '부피가 큰 경우 압착'
        ],
        'icon': '♻️'
    },
    'trash': {
        'method': '일반 쓰레기',
        'description': '재활용이 불가능한 일반 쓰레기는 종량제 봉투에 담아 배출하세요.',
        'tips': [
            '종량제 봉투 사용',
            '재활용 가능 여부 재확인',
            '음식물 찌꺼기 제거 후 배출'
        ],
        'icon': '🗑️'
    }
}

# 업로드 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)  # 학습 데이터 폴더 생성
DETECTION_RESULTS_FOLDER = os.path.join('static', 'detection_results')
os.makedirs(DETECTION_RESULTS_FOLDER, exist_ok=True)  # 탐지 결과 이미지 저장 폴더
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# TACO 탐지 모델 초기화 (선택적)
taco_detector = None
class_mapper = None
if TACO_DETECTOR_AVAILABLE:
    try:
        # 클래스 매퍼 초기화
        class_mapper = ClassMapper()
        print("✅ 클래스 매퍼 초기화 완료")
        
        # 탐지 모델은 필요할 때 로드 (모델 파일이 있을 때만)
        # taco_detector = create_detector(...)  # 나중에 필요할 때 로드
    except Exception as e:
        print(f"⚠️ TACO 탐지 모델 초기화 실패: {e}")
        TACO_DETECTOR_AVAILABLE = False

# 모델 로드
if os.path.exists(MODEL_PATH_KERAS):
    MODEL_PATH = MODEL_PATH_KERAS
elif os.path.exists(MODEL_PATH_H5):
    MODEL_PATH = MODEL_PATH_H5
else:
    raise FileNotFoundError(
        f"모델 파일을 찾을 수 없습니다.\n"
        f"확인한 경로: {MODEL_PATH_KERAS}, {MODEL_PATH_H5}\n"
        "먼저 'python gar.py'를 실행하여 모델을 학습시켜주세요."
    )

model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ 모델 로드 완료: {MODEL_PATH}")

# 유틸리티 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image_for_training(filepath, predicted_class, confidence, original_filename):
    """
    예측된 이미지를 학습 데이터로 저장
    
    Args:
        filepath: 원본 이미지 파일 경로
        predicted_class: 예측된 클래스 이름
        confidence: 예측 신뢰도 (%)
        original_filename: 원본 파일명
    
    Returns:
        저장된 파일 경로 또는 None (저장하지 않은 경우)
    """
    # 신뢰도가 낮으면 저장하지 않음
    if confidence < MIN_CONFIDENCE_FOR_SAVE:
        print(f"⚠️ 신뢰도가 낮아 학습 데이터로 저장하지 않음: {predicted_class} ({confidence}%)")
        return None
    
    try:
        # 클래스 폴더 경로
        class_folder = os.path.join(DATA_DIR, predicted_class)
        os.makedirs(class_folder, exist_ok=True)
        
        # 새 파일명 생성 (중복 방지)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지
        file_ext = os.path.splitext(original_filename)[1]
        new_filename = f"{predicted_class}_{timestamp}{file_ext}"
        new_filepath = os.path.join(class_folder, new_filename)
        
        # 이미지 복사
        shutil.copy2(filepath, new_filepath)
        
        print(f"✅ 학습 데이터로 저장: {new_filepath} (신뢰도: {confidence}%)")
        return new_filepath
    except Exception as e:
        print(f"❌ 학습 데이터 저장 실패: {str(e)}")
        return None


# 라우트
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('파일이 선택되지 않았습니다.')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('파일이 선택되지 않았습니다.')
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash('지원하지 않는 파일 형식입니다. 이미지 파일(jpg, png, jpeg, gif, bmp)만 업로드 가능합니다.')
        return redirect(request.url)

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 이미지 전처리
        img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 예측
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASSES[predicted_idx]
        confidence = round(100 * np.max(predictions), 2)

        # 전체 예측 결과
        all_predictions = {
            CLASSES[i]: round(100 * predictions[0][i], 2)
            for i in range(len(CLASSES))
        }
        sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)

        # 한국어 이름 추가
        label_ko = CLASS_NAMES_KO.get(predicted_class, predicted_class)
        predictions_with_ko = [
            (class_name, percent, CLASS_NAMES_KO.get(class_name, class_name))
            for class_name, percent in sorted_predictions
        ]

        # 처리 방법 안내 정보
        disposal_info = DISPOSAL_GUIDE.get(predicted_class, {
            'method': '일반 쓰레기',
            'description': '적절한 방법으로 처리해주세요.',
            'tips': [],
            'icon': '🗑️'
        })

        # 학습 데이터로 저장 (신뢰도가 충분한 경우)
        saved_for_training = save_image_for_training(
            filepath, 
            predicted_class, 
            confidence, 
            filename
        )
        
        # 저장 여부를 템플릿에 전달 (선택사항)
        training_saved = saved_for_training is not None

        return render_template('result.html',
                             filename=filename,
                             label=predicted_class,
                             label_ko=label_ko,
                             confidence=confidence,
                             all_predictions=predictions_with_ko,
                             disposal_info=disposal_info,
                             training_saved=training_saved)
    except Exception as e:
        flash(f'예측 중 오류가 발생했습니다: {str(e)}')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display-pending/<class_name>/<filename>')
def display_pending_image(class_name, filename):
    """pending 폴더의 이미지 표시"""
    pending_path = os.path.join(DATA_DIR, 'pending', class_name, filename)
    if os.path.exists(pending_path):
        from flask import send_file
        return send_file(pending_path)
    else:
        flash('이미지를 찾을 수 없습니다.')
        return redirect(url_for('admin_pending_classes'))

def draw_bounding_boxes(image_path, detections, output_path):
    """
    이미지에 바운딩 박스와 레이블을 그려서 저장
    
    Args:
        image_path: 원본 이미지 경로
        detections: 탐지 결과 리스트
        output_path: 저장할 경로
    """
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # 색상 팔레트
        colors = [
            (255, 0, 0),    # 빨강
            (0, 255, 0),    # 초록
            (0, 0, 255),    # 파랑
            (255, 255, 0),  # 노랑
            (255, 0, 255),  # 마젠타
            (0, 255, 255),  # 시안
        ]
        
        for i, detection in enumerate(detections):
            # 바운딩 박스 좌표 [x1, y1, x2, y2]
            x1, y1, x2, y2 = detection['bbox_xyxy']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 색상 선택
            color = colors[i % len(colors)]
            
            # 바운딩 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 레이블 텍스트
            label = f"{detection['class_name']} {detection['confidence']:.1%}"
            
            # 텍스트 배경
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # 텍스트 크기 계산
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 텍스트 배경 그리기
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], 
                         fill=color, outline=color)
            
            # 텍스트 그리기
            draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
        
        # 저장
        img.save(output_path)
        return True
    except Exception as e:
        print(f"❌ 바운딩 박스 그리기 실패: {e}")
        return False

@app.route('/predict-detection', methods=['POST'])
def predict_detection():
    """탐지 모드: 이미지에서 여러 객체 탐지"""
    if not TACO_DETECTOR_AVAILABLE:
        flash('탐지 기능을 사용할 수 없습니다. TACO 탐지 모델이 로드되지 않았습니다.')
        return redirect(url_for('home'))
    
    if 'file' not in request.files:
        flash('파일이 선택되지 않았습니다.')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('파일이 선택되지 않았습니다.')
        return redirect(url_for('home'))
    
    if not allowed_file(file.filename):
        flash('지원하지 않는 파일 형식입니다. 이미지 파일(jpg, png, jpeg, gif, bmp)만 업로드 가능합니다.')
        return redirect(url_for('home'))
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 탐지 모델 로드 (지연 로드)
        global taco_detector
        if taco_detector is None:
            # 모델 경로 확인
            model_path = os.path.join('C:', 'Taco', 'detector', 'models', 'mask_rcnn_taco.h5')
            if not os.path.exists(model_path):
                flash('탐지 모델 파일을 찾을 수 없습니다. 모델을 학습시키거나 경로를 확인해주세요.')
                return redirect(url_for('home'))
            
            try:
                taco_detector = create_detector(
                    model_path=model_path,
                    class_map='map_10.csv',
                    detection_min_confidence=0.5
                )
                print("✅ TACO 탐지 모델 로드 완료")
            except Exception as e:
                flash(f'탐지 모델 로드 실패: {str(e)}')
                return redirect(url_for('home'))
        
        # 탐지 수행
        results = taco_detector.detect(filepath, return_masks=False)
        
        # 탐지 결과 처리
        detections = []
        saved_cropped_images = []
        
        # 원본 이미지 로드 (크롭용)
        original_img = Image.open(filepath)
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        img_array = np.array(original_img)
        
        for i, detection in enumerate(results['detections']):
            # 클래스 매핑
            taco_class = detection['class_name']
            if class_mapper:
                mapped_class, status, needs_review = class_mapper.map_class(
                    taco_class, 
                    detection['confidence']
                )
            else:
                mapped_class = taco_class
                status = 'mapped'
                needs_review = False
            
            detections.append({
                'taco_class': taco_class,
                'mapped_class': mapped_class,
                'class_name': mapped_class,  # 표시용
                'confidence': detection['confidence'],
                'bbox': detection['bbox'],
                'bbox_xyxy': detection['bbox_xyxy'],
                'status': status,
                'needs_review': needs_review
            })
            
            # 학습 데이터로 저장 (신뢰도가 충분한 경우)
            if detection['confidence'] >= (MIN_CONFIDENCE_FOR_SAVE / 100.0):
                try:
                    # 바운딩 박스로 직접 크롭
                    x1, y1, x2, y2 = detection['bbox_xyxy']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 이미지 경계 확인
                    h, w = img_array.shape[:2]
                    x1 = max(0, min(x1, w))
                    x2 = max(0, min(x2, w))
                    y1 = max(0, min(y1, h))
                    y2 = max(0, min(y2, h))
                    
                    if x2 > x1 and y2 > y1:
                        # 크롭
                        cropped = img_array[y1:y2, x1:x2]
                        
                        if cropped.size > 0:
                            # 저장 경로 결정
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            file_ext = os.path.splitext(filename)[1]
                            
                            # 새 클래스 후보인 경우 pending 폴더에 저장
                            if status in ['new_candidate', 'pending']:
                                pending_folder = os.path.join(DATA_DIR, 'pending', mapped_class)
                                os.makedirs(pending_folder, exist_ok=True)
                                cropped_filename = f"{mapped_class}_{timestamp}_{i}{file_ext}"
                                cropped_path = os.path.join(pending_folder, cropped_filename)
                            else:
                                # 기존 클래스인 경우 바로 저장
                                class_folder = os.path.join(DATA_DIR, mapped_class)
                                os.makedirs(class_folder, exist_ok=True)
                                cropped_filename = f"{mapped_class}_{timestamp}_{i}{file_ext}"
                                cropped_path = os.path.join(class_folder, cropped_filename)
                            
                            Image.fromarray(cropped).save(cropped_path)
                            saved_cropped_images.append(cropped_path)
                            print(f"✅ 탐지 객체 저장: {cropped_path} (상태: {status})")
                except Exception as e:
                    print(f"⚠️ 객체 크롭/저장 실패: {e}")
        
        # 바운딩 박스가 그려진 이미지 생성
        result_image_filename = f"detected_{filename}"
        result_image_path = os.path.join(DETECTION_RESULTS_FOLDER, result_image_filename)
        draw_bounding_boxes(filepath, detections, result_image_path)
        
        return render_template('result_detection.html',
                             filename=filename,
                             result_image=result_image_filename,
                             detections=detections,
                             num_detections=len(detections),
                             saved_count=len(saved_cropped_images))
    
    except Exception as e:
        flash(f'탐지 중 오류가 발생했습니다: {str(e)}')
        import traceback
        print(traceback.format_exc())
        return redirect(url_for('home'))

# 관리자 페이지 라우트
@app.route('/admin')
def admin_home():
    """관리자 메인 페이지"""
    return render_template('admin_home.html')

@app.route('/admin/pending-classes')
def admin_pending_classes():
    """승인 대기 중인 새 클래스 목록"""
    if not class_mapper:
        flash('클래스 매퍼가 초기화되지 않았습니다.')
        return redirect(url_for('home'))
    
    pending = class_mapper.get_pending_classes()
    
    # 각 클래스별 샘플 이미지 찾기
    pending_with_samples = []
    for p in pending:
        # pending 폴더에서 샘플 이미지 찾기
        pending_folder = os.path.join(DATA_DIR, 'pending', p['normalized_name'])
        sample_images = []
        if os.path.exists(pending_folder):
            images = [f for f in os.listdir(pending_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            sample_images = images[:3]  # 최대 3개만
        
        pending_with_samples.append({
            **p,
            'sample_images': sample_images,
            'sample_count': len(sample_images)
        })
    
    return render_template('admin_pending_classes.html', 
                         pending_classes=pending_with_samples)

@app.route('/admin/approve-class', methods=['POST'])
def admin_approve_class():
    """새 클래스 승인"""
    if not class_mapper:
        flash('클래스 매퍼가 초기화되지 않았습니다.')
        return redirect(url_for('admin_pending_classes'))
    
    taco_class = request.form.get('taco_class')
    approved_name = request.form.get('approved_name', '').strip()
    
    if not taco_class:
        flash('클래스 이름이 제공되지 않았습니다.')
        return redirect(url_for('admin_pending_classes'))
    
    try:
        # 승인
        if approved_name:
            success = class_mapper.approve_new_class(taco_class, approved_name)
        else:
            success = class_mapper.approve_new_class(taco_class)
        
        if success:
            # pending 폴더의 이미지를 새 클래스 폴더로 이동
            pending_info = None
            for p in class_mapper.get_pending_classes():
                if p['taco_class'] == taco_class:
                    pending_info = p
                    break
            
            if pending_info:
                pending_folder = os.path.join(DATA_DIR, 'pending', pending_info['normalized_name'])
                if os.path.exists(pending_folder):
                    # 승인된 클래스 이름 결정
                    final_name = approved_name if approved_name else pending_info['normalized_name']
                    final_name = class_mapper.normalize_class_name(final_name)
                    target_folder = os.path.join(DATA_DIR, final_name)
                    os.makedirs(target_folder, exist_ok=True)
                    
                    # 파일 이동
                    for filename in os.listdir(pending_folder):
                        src = os.path.join(pending_folder, filename)
                        dst = os.path.join(target_folder, filename)
                        shutil.move(src, dst)
                    
                    # 빈 폴더 삭제
                    try:
                        os.rmdir(pending_folder)
                    except:
                        pass
            
            flash(f'클래스 "{taco_class}"가 승인되었습니다.')
        else:
            flash(f'클래스 승인에 실패했습니다.')
    except Exception as e:
        flash(f'오류가 발생했습니다: {str(e)}')
        import traceback
        print(traceback.format_exc())
    
    return redirect(url_for('admin_pending_classes'))

@app.route('/admin/reject-class', methods=['POST'])
def admin_reject_class():
    """새 클래스 거부 (기존 클래스로 매핑)"""
    if not class_mapper:
        flash('클래스 매퍼가 초기화되지 않았습니다.')
        return redirect(url_for('admin_pending_classes'))
    
    taco_class = request.form.get('taco_class')
    mapped_to = request.form.get('mapped_to', 'trash')
    
    if not taco_class:
        flash('클래스 이름이 제공되지 않았습니다.')
        return redirect(url_for('admin_pending_classes'))
    
    try:
        # 거부 및 매핑
        success = class_mapper.reject_new_class(taco_class, mapped_to)
        
        if success:
            # pending 폴더의 이미지를 매핑된 클래스 폴더로 이동
            pending_info = None
            for p in class_mapper.get_pending_classes():
                if p['taco_class'] == taco_class:
                    pending_info = p
                    break
            
            if pending_info:
                pending_folder = os.path.join(DATA_DIR, 'pending', pending_info['normalized_name'])
                if os.path.exists(pending_folder):
                    target_folder = os.path.join(DATA_DIR, mapped_to)
                    os.makedirs(target_folder, exist_ok=True)
                    
                    # 파일 이동
                    for filename in os.listdir(pending_folder):
                        src = os.path.join(pending_folder, filename)
                        dst = os.path.join(target_folder, filename)
                        shutil.move(src, dst)
                    
                    # 빈 폴더 삭제
                    try:
                        os.rmdir(pending_folder)
                    except:
                        pass
            
            flash(f'클래스 "{taco_class}"가 "{mapped_to}"로 매핑되었습니다.')
        else:
            flash(f'클래스 거부에 실패했습니다.')
    except Exception as e:
        flash(f'오류가 발생했습니다: {str(e)}')
        import traceback
        print(traceback.format_exc())
    
    return redirect(url_for('admin_pending_classes'))

@app.route('/admin/classes')
def admin_classes():
    """클래스 관리 및 통계 페이지"""
    if not class_mapper:
        flash('클래스 매퍼가 초기화되지 않았습니다.')
        return redirect(url_for('home'))
    
    # 모든 클래스 목록
    all_classes = class_mapper.get_all_classes()
    
    # 클래스별 통계
    stats = class_mapper.get_class_statistics(DATA_DIR)
    
    # 클래스 정보 리스트
    classes_info = []
    for class_name in all_classes:
        classes_info.append({
            'name': class_name,
            'name_ko': CLASS_NAMES_KO.get(class_name, class_name),
            'image_count': stats.get(class_name, 0),
            'icon': DISPOSAL_GUIDE.get(class_name, {}).get('icon', '🗑️')
        })
    
    # 총 이미지 수
    total_images = sum(stats.values())
    
    return render_template('admin_classes.html',
                         classes=classes_info,
                         total_images=total_images,
                         total_classes=len(all_classes))

if __name__ == '__main__':
    import threading
    import time
    import webbrowser

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    def open_browser():
        time.sleep(1.5)
        url = 'http://127.0.0.1:5000'
        print(f"\n브라우저를 엽니다: {url}")
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    print("\n" + "=" * 50)
    print("쓰레기 분류 Flask 서버를 시작합니다...")
    print("서버 주소: http://127.0.0.1:5000")
    print("=" * 50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
