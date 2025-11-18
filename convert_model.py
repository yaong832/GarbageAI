"""
기존 .h5 모델을 .keras 형식으로 변환하는 스크립트
재학습 없이 모델 형식만 변환합니다.
"""

import tensorflow as tf
import os

# 모델 경로 설정
MODEL_PATH_H5 = os.path.join('model', 'garbage_model.h5')
MODEL_PATH_KERAS = os.path.join('model', 'garbage_model.keras')

print("=" * 50)
print("모델 형식 변환 스크립트")
print("=" * 50)

# 기존 .h5 파일 확인
if not os.path.exists(MODEL_PATH_H5):
    print(f"❌ 오류: 기존 모델 파일을 찾을 수 없습니다: {MODEL_PATH_H5}")
    print("\n다음 중 하나를 선택하세요:")
    print("1. gar.py를 실행하여 새로 모델을 학습시키기")
    print("2. 모델 파일 경로를 확인하세요")
    exit(1)

# .keras 파일이 이미 존재하는지 확인
if os.path.exists(MODEL_PATH_KERAS):
    response = input(f"\n⚠️  {MODEL_PATH_KERAS} 파일이 이미 존재합니다.\n덮어쓰시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("변환이 취소되었습니다.")
        exit(0)

print(f"\n📂 기존 모델 로드 중: {MODEL_PATH_H5}")
try:
    # 기존 모델 로드
    model = tf.keras.models.load_model(MODEL_PATH_H5)
    print("✅ 모델 로드 완료!")
    
    # 모델 정보 출력
    print(f"\n📊 모델 정보:")
    print(f"  - 입력 형태: {model.input_shape}")
    print(f"  - 출력 형태: {model.output_shape}")
    print(f"  - 총 파라미터 수: {model.count_params():,}")
    
    # .keras 형식으로 저장
    print(f"\n💾 .keras 형식으로 변환 중: {MODEL_PATH_KERAS}")
    model.save(MODEL_PATH_KERAS)
    print("✅ 변환 완료!")
    
    # 파일 크기 비교
    h5_size = os.path.getsize(MODEL_PATH_H5) / (1024 * 1024)  # MB
    keras_size = os.path.getsize(MODEL_PATH_KERAS) / (1024 * 1024)  # MB
    
    print(f"\n📦 파일 크기:")
    print(f"  - .h5 파일: {h5_size:.2f} MB")
    print(f"  - .keras 파일: {keras_size:.2f} MB")
    
    # 변환된 모델 검증
    print(f"\n🔍 변환된 모델 검증 중...")
    test_model = tf.keras.models.load_model(MODEL_PATH_KERAS)
    print("✅ 변환된 모델이 정상적으로 로드됩니다!")
    
    print("\n" + "=" * 50)
    print("✨ 변환 성공!")
    print("=" * 50)
    print(f"이제 {MODEL_PATH_KERAS} 파일을 사용할 수 있습니다.")
    print("기존 .h5 파일은 삭제해도 되지만, 백업용으로 보관하는 것을 권장합니다.")
    print("=" * 50)
    
except Exception as e:
    print(f"\n❌ 오류 발생: {str(e)}")
    print("\n문제 해결 방법:")
    print("1. TensorFlow 버전 확인 (2.13.0 이상 권장)")
    print("2. 모델 파일이 손상되지 않았는지 확인")
    print("3. gar.py를 실행하여 새로 모델을 학습시키기")
    exit(1)

