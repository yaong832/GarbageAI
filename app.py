"""
쓰레기 분류 웹 애플리케이션
"""

import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Flask 앱 초기화
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'garbage_classification_secret_key_2024')

# 설정
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
IMG_SIZE = 224
UPLOAD_FOLDER = os.path.join('static', 'uploads')
MODEL_PATH_KERAS = os.path.join('model', 'garbage_model.keras')
MODEL_PATH_H5 = os.path.join('model', 'garbage_model.h5')

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
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

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

        return render_template('result.html',
                             filename=filename,
                             label=predicted_class,
                             label_ko=label_ko,
                             confidence=confidence,
                             all_predictions=predictions_with_ko,
                             disposal_info=disposal_info)
    except Exception as e:
        flash(f'예측 중 오류가 발생했습니다: {str(e)}')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
