"""
쓰레기 분류 모델 학습 스크립트
전이학습과 개선된 아키텍처를 사용하여 정확도를 향상시킵니다.
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 설정
DATA_DIR = os.path.join('data', 'garbage_dataset')
MODEL_SAVE_PATH = os.path.join('model', 'garbage_model.keras')
IMG_SIZE = 224  # 전이학습에 최적화된 크기
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001  # 전이학습에 적합한 낮은 학습률
VALIDATION_SPLIT = 0.2
CLASSES = ['battery', 'biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_CLASSES = len(CLASSES)

# 데이터 디렉토리 확인
if not os.path.exists(DATA_DIR):
    print(f"[오류] 데이터 디렉토리를 찾을 수 없습니다: {DATA_DIR}")
    print("\n데이터 구조:")
    print("  data/garbage_dataset/")
    print("    ├── battery/")
    print("    ├── biological/")
    print("    ├── cardboard/")
    print("    ├── glass/")
    print("    ├── metal/")
    print("    ├── paper/")
    print("    ├── plastic/")
    print("    └── trash/")
    exit(1)

print("=" * 60)
print("쓰레기 분류 모델 학습 (개선된 버전)")
print("=" * 60)
print(f"데이터셋: {DATA_DIR}")
print(f"이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
print(f"배치 크기: {BATCH_SIZE}")
print(f"에포크: {EPOCHS}")
print(f"클래스 수: {NUM_CLASSES}")
print(f"모델: MobileNetV2 (전이학습)")
print("=" * 60)

# 강화된 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
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

# 전이학습 모델 구축
print("\n[모델 구축 중 (MobileNetV2 전이학습)...]")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# 사전 학습된 가중치 고정 (처음 몇 에포크 동안)
base_model.trainable = False

# 커스텀 분류 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print("\n[모델 구조:]")
model.summary()

# 콜백 설정
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
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
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# 1단계: 사전 학습된 레이어 고정 상태로 학습
print("\n[1단계 학습 시작 (사전 학습 레이어 고정)...]")
print("-" * 60)

history1 = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=15,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callbacks,
    verbose=1
)

# 2단계: Fine-tuning - 상위 레이어 일부 해제
print("\n[2단계 Fine-tuning 시작 (상위 레이어 해제)...]")
base_model.trainable = True

# 상위 레이어만 학습 가능하도록 설정
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 더 낮은 학습률로 재컴파일
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE * 0.1),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print("-" * 60)
history2 = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS - 15,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callbacks,
    verbose=1,
    initial_epoch=15
)

# 최종 모델 저장
model.save(MODEL_SAVE_PATH)

# 결과 출력
print("\n" + "=" * 60)
print("[학습 완료!]")
print("=" * 60)

# 최종 결과는 history2가 있으면 history2, 없으면 history1 사용
if 'history2' in locals() and len(history2.history.get('accuracy', [])) > 0:
    final_history = history2
    print(f"1단계 학습 에포크: {len(history1.history['accuracy'])}")
    print(f"2단계 학습 에포크: {len(history2.history['accuracy'])}")
else:
    final_history = history1
    print(f"총 학습 에포크: {len(history1.history['accuracy'])}")

print(f"최종 학습 정확도: {final_history.history['accuracy'][-1]:.4f}")
print(f"최종 검증 정확도: {final_history.history['val_accuracy'][-1]:.4f}")
print(f"최종 학습 손실: {final_history.history['loss'][-1]:.4f}")
print(f"최종 검증 손실: {final_history.history['val_loss'][-1]:.4f}")

if 'val_top_3_accuracy' in final_history.history:
    print(f"Top-3 정확도: {final_history.history['val_top_3_accuracy'][-1]:.4f}")

print(f"모델 저장 위치: {MODEL_SAVE_PATH}")
print("=" * 60)
