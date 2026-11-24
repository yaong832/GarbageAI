# 쓰레기 분류 AI (GarbageAI)

> 📂 [포트폴리오 (Notion)](https://iced-tarsier-455.notion.site/350af40f5e6680d19f93f1653925a3b8) · 🎥 [시연 영상](https://youtu.be/_AzK80T7Q70)

쓰레기 사진을 올리면 8가지 종류 중 하나로 분류하고, 종류별 버리는 방법을 알려 주는 Flask 웹 앱입니다. 모델은 MobileNetV2 전이학습으로 만들었습니다.

## 기능

| 기능 | 설명 |
| --- | --- |
| 분류 모드 | 사진 한 장을 8종 중 하나로 분류하고, 8종 전체의 확률과 처리 방법(`DISPOSAL_GUIDE`)을 보여 줍니다. |
| 탐지 모드 (선택) | TACO(Mask R-CNN)로 사진 속 쓰레기를 여러 개 찾아 상자로 표시하고, TACO 종류를 매핑 표(`class_mapper.py`)로 8종에 맞춰 보여 줍니다. TACO가 없으면 이 모드만 꺼지고 분류 모드는 그대로 동작합니다. |
| 학습 데이터 자동 저장 | 신뢰도가 50% 이상이면 올린 사진(탐지 모드는 잘라낸 물체)을 `data/garbage_dataset/<예측 종류>/`에 저장해 다음 학습에 씁니다. |
| 관리자 화면 (`/admin`) | 탐지 모드에서 8종에 없는 물체가 나오면 `pending`에 모아 두고, 관리자 화면에서 새 클래스로 승인하거나 기존 클래스로 합치거나 거절합니다. 클래스별 이미지 수도 볼 수 있습니다. |

분류하는 8종: 배터리(battery), 음식물·생물(biological), 골판지(cardboard), 유리(glass), 금속(metal), 종이(paper), 플라스틱(plastic), 일반 쓰레기(trash)

## 실행

필요한 것: Windows 또는 Linux/Mac, Python 3.10~3.12 (TensorFlow 2.16 이상을 설치할 수 있는 버전)

### Windows

`run_web.bat`을 더블클릭하면 패키지를 설치하고 서버를 띄운 뒤 브라우저(`http://127.0.0.1:5000`)를 엽니다.

### 직접 실행

```bash
python -m venv venv
venv\Scripts\activate        # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

학습된 모델 `model/garbage_model.keras`가 저장소에 들어 있어서, 데이터셋 없이도 바로 분류할 수 있습니다.

### 환경 변수

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `GARBAGE_FLASK_HOST` | `0.0.0.0` | 같은 PC에서만 쓸 때는 `127.0.0.1` |
| `GARBAGE_FLASK_DEBUG` | 꺼짐 | `1`이면 Flask 디버그 모드 (개발할 때만) |
| `SECRET_KEY` | 실행마다 임의 생성 | 화면 알림 메시지 서명용 키 |

포트 5000을 쓰므로 같은 포트를 쓰는 다른 Flask 앱과 동시에 띄우지 마세요.

> ⚠️ 관리자 화면(`/admin`)에는 로그인이 없습니다. 믿을 수 있는 네트워크에서만 띄우거나 `GARBAGE_FLASK_HOST=127.0.0.1`로 실행하세요.

## 데이터셋

학습 이미지는 저장소에 넣지 않습니다. 이 프로젝트는 Kaggle의 [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) (Suman Kunwar)에서 10개 클래스 중 clothes·shoes를 뺀 8개 클래스를 사용했습니다. cardboard·glass·paper·plastic은 원본보다 적은 약 1,000장씩만 들어 있습니다.

| 클래스 | 사용한 장수 |
| --- | --- |
| battery | 944 |
| biological | 997 |
| cardboard | 1,029 |
| glass | 1,076 |
| metal | 1,020 |
| paper | 1,019 |
| plastic | 1,034 |
| trash | 947 |

라이선스와 인용 방법은 Kaggle 데이터셋 페이지를 따릅니다. 데이터셋 설명 논문: Suman Kunwar, *The Garbage Dataset (GD): A Multi-Class Image Benchmark for Automated Waste Segregation*, [arXiv:2602.10500](https://arxiv.org/abs/2602.10500). Kaggle 데이터셋은 이후 정리(중복·워터마크 제거)되어, 지금 받으면 장수가 위 표와 다를 수 있습니다.

### 준비

Kaggle에서 받은 뒤 아래처럼 8개 폴더만 넣습니다. 다른 폴더(clothes, shoes, 관리자 화면의 `pending` 등)가 있어도 학습 스크립트는 8개 클래스만 읽습니다.

```
data/
└── garbage_dataset/
    ├── battery/
    ├── biological/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

## 학습

| 스크립트 | 용도 |
| --- | --- |
| `python gar.py` (또는 `train_model.bat`) | 처음부터 학습. MobileNetV2(ImageNet 가중치), 224×224, 1단계 15 에포크(분류층만) + 2단계 15 에포크(상위층 미세 조정), 검증 20% |
| `python gar_refine.py` | 기존 모델에 추가 학습(20 에포크, 낮은 학습률). 시작 전에 `model/garbage_model_backup.keras`로 백업 |
| `python convert_model.py` | 예전 `.h5` 모델을 `.keras`로 변환 |

결과는 `model/garbage_model.keras`에 저장되고, 앱을 다시 시작하면 새 모델을 씁니다. 자세한 내용은 [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)를 보세요.

> 자동 저장된 사진은 모델이 예측한 라벨 그대로 들어갑니다. 틀린 예측도 섞일 수 있으니, 다시 학습하기 전에 폴더를 한 번 훑어보는 것이 좋습니다. 관리자 화면에서 승인한 새 클래스는 폴더만 생기고, 지금 모델과 학습 스크립트는 8종만 다룹니다.

## 탐지 모드 준비 (선택)

탐지 모드는 [TACO](https://github.com/pedropro/TACO) 프로젝트의 탐지기를 그대로 불러 씁니다.

1. TACO 저장소를 `C:\Taco`에 받습니다(경로는 `taco_detector.py`의 `TACO_DIR`).
2. 학습된 가중치를 `C:\Taco\detector\models\mask_rcnn_taco.h5`에 둡니다. 클래스 맵은 `C:\Taco\detector\taco_config\map_10.csv`를 씁니다.
3. TACO가 쓰는 `imgaug`는 NumPy 1.x에서만 동작합니다. NumPy 2.x가 깔려 있으면 앱이 탐지 모드를 끄고 시작하므로, `fix_numpy.bat`으로 NumPy를 1.x로 내립니다.

자세한 내용은 [docs/TACO_DETECTOR_GUIDE.md](docs/TACO_DETECTOR_GUIDE.md), [docs/NUMPY_FIX_GUIDE.md](docs/NUMPY_FIX_GUIDE.md)를 보세요.

## 프로젝트 구조

```
GarbageAI/
├── app.py                  # Flask 웹 앱 (분류·탐지·관리자 화면)
├── gar.py                  # 처음부터 학습
├── gar_refine.py           # 기존 모델 추가 학습
├── convert_model.py        # .h5 → .keras 변환
├── taco_detector.py        # TACO Mask R-CNN 래퍼
├── class_mapper.py         # TACO 클래스 → 8종 매핑, 새 클래스 승인 대기 관리
├── run_web.bat             # Windows: 설치 + 실행
├── train_model.bat         # Windows: 설치 + 학습
├── fix_numpy.bat           # Windows: NumPy 1.x로 내리기 (탐지 모드용)
├── requirements.txt
├── model/
│   └── garbage_model.keras # 학습된 분류 모델
├── templates/              # 메인, 분류 결과, 탐지 결과, 관리자 화면 3개
├── docs/                   # 개발 메모 (학습·탐지·의존성 가이드 등)
├── data/garbage_dataset/   # 학습 데이터 (Git에 올리지 않음)
└── static/
    ├── uploads/            # 올린 사진 (Git에 올리지 않음)
    └── detection_results/  # 탐지 결과 이미지 (Git에 올리지 않음)
```

## 기술 스택

- Backend: Flask, TensorFlow/Keras
- 모델: MobileNetV2 전이학습 (분류), TACO Mask R-CNN (탐지, 선택)
- Frontend: HTML, CSS, JavaScript

## 라이선스

개인 학습·포트폴리오용으로 만든 프로젝트입니다. 데이터셋과 TACO는 각 출처의 라이선스를 따릅니다.
