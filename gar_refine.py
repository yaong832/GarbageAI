"""
기존 모델을 로드하여 추가 학습(정제)하는 스크립트
더 높은 정확도를 위해 미세 조정을 수행합니다.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 설정
DATA_DIR = os.path.join('data', 'garbage_dataset')
MODEL_SAVE_PATH = os.path.join('model', 'garbage_model.keras')
BACKUP_MODEL_PATH = os.path.join('model', 'garbage_model_backup.keras')
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20  # 추가 학습 에포크
LEARNING_RATE = 0.00001  # 매우 낮은 학습률로 미세 조정
VALIDATION_SPLIT = 0.2

# 모델 파일 확인
if not os.path.exists(MODEL_SAVE_PATH):
    print(f"[오류] 모델 파일을 찾을 수 없습니다: {MODEL_SAVE_PATH}")
    print("먼저 'python gar.py'를 실행하여 기본 모델을 학습시켜주세요.")
    exit(1)

# 데이터 디렉토리 확인
if not os.path.exists(DATA_DIR):
    print(f"[오류] 데이터 디렉토리를 찾을 수 없습니다: {DATA_DIR}")
    exit(1)

print("=" * 60)
print("모델 추가 학습 (정제)")
print("=" * 60)
print(f"기존 모델: {MODEL_SAVE_PATH}")
print(f"데이터셋: {DATA_DIR}")
print(f"이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
print(f"배치 크기: {BATCH_SIZE}")
print(f"추가 학습 에포크: {EPOCHS}")
print(f"학습률: {LEARNING_RATE}")
print("=" * 60)

# 기존 모델 백업
if os.path.exists(MODEL_SAVE_PATH):
    import shutil
    shutil.copy2(MODEL_SAVE_PATH, BACKUP_MODEL_PATH)
    print(f"\n[백업 완료] 기존 모델을 백업했습니다: {BACKUP_MODEL_PATH}")

# 기존 모델 로드 (optimizer 제외)
print("\n[모델 로드 중...]")
try:
    # optimizer를 제외하고 모델 구조와 가중치만 로드
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)
    print("[OK] 모델 로드 완료! (optimizer 제외)")
except Exception as e:
    print(f"[경고] 모델 로드 중 문제 발생: {e}")
    print("[시도] 일반 방식으로 모델 로드...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print("[OK] 모델 로드 완료!")

# 모든 레이어를 학습 가능하도록 설정 (미세 조정)
print("\n[모든 레이어를 학습 가능하도록 설정...]")
for layer in model.layers:
    layer.trainable = True

# 더 낮은 학습률로 재컴파일
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

# 강화된 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.7, 1.3],
    channel_shift_range=0.2,
    validation_split=VALIDATION_SPLIT
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

# 데이터 로더 생성
print("\n[데이터 로딩 중...]")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"[OK] 학습 샘플: {train_generator.samples:,}장")
print(f"[OK] 검증 샘플: {validation_generator.samples:,}장")

# 콜백 설정
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  # 더 많은 patience로 더 많이 학습
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # 더 공격적인 학습률 감소
        patience=5,
        min_lr=1e-8,
        verbose=1
    )
]

# 추가 학습 시작
print("\n[추가 학습 시작...]")
print("-" * 60)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callbacks,
    verbose=1
)

# 최종 모델 저장
model.save(MODEL_SAVE_PATH)

# 결과 출력
print("\n" + "=" * 60)
print("[추가 학습 완료!]")
print("=" * 60)
print(f"총 학습 에포크: {len(history.history['accuracy'])}")
print(f"최종 학습 정확도: {history.history['accuracy'][-1]:.4f}")
print(f"최종 검증 정확도: {history.history['val_accuracy'][-1]:.4f}")
print(f"최종 학습 손실: {history.history['loss'][-1]:.4f}")
print(f"최종 검증 손실: {history.history['val_loss'][-1]:.4f}")

if 'val_top_3_accuracy' in history.history:
    print(f"Top-3 정확도: {history.history['val_top_3_accuracy'][-1]:.4f}")

print(f"\n모델 저장 위치: {MODEL_SAVE_PATH}")
print(f"백업 모델 위치: {BACKUP_MODEL_PATH}")
print("=" * 60)

