# NumPy 버전 호환성 문제 해결 가이드

## 문제 상황

Python 3.13 환경에서 NumPy 2.3.4가 설치되어 있어 `imgaug`와 호환되지 않습니다.

**오류 메시지:**
```
AttributeError: `np.sctypes` was removed in the NumPy 2.0 release.
```

## 해결 방법

### 방법 1: Python 3.11 이하 사용 (권장)

Python 3.13은 NumPy 1.x의 사전 컴파일된 wheel이 없어서 설치가 어렵습니다.

1. **Python 3.11 설치**
2. **새 가상환경 생성**
   ```bash
   python3.11 -m venv c:/venvs/myproject311
   ```
3. **가상환경 활성화**
   ```bash
   c:/venvs/myproject311/Scripts/activate
   ```
4. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

### 방법 2: 현재 환경에서 분류 모드만 사용

현재 코드는 NumPy 2.x 환경에서도 **분류 모드**는 정상 작동합니다.

- ✅ 분류 모드: 정상 작동
- ❌ 탐지 모드: NumPy 1.x 필요

**탐지 모드는 자동으로 비활성화됩니다.**

### 방법 3: NumPy 다운그레이드 시도 (Python 3.11 이하에서만 가능)

Python 3.11 이하 환경에서:

```bash
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

또는 특정 버전:

```bash
pip install numpy==1.26.3 --force-reinstall
```

## 현재 상태

코드가 수정되어:
- ✅ NumPy 버전을 자동으로 체크
- ✅ NumPy 2.x 환경에서는 탐지 모드 자동 비활성화
- ✅ 분류 모드는 정상 작동
- ✅ 에러 없이 Flask 앱 실행 가능

## 확인 방법

Flask 앱 실행 시 콘솔에 다음 메시지가 표시됩니다:

```
⚠️ NumPy 2.3.4 버전은 imgaug와 호환되지 않습니다.
⚠️ 탐지 기능은 사용할 수 없습니다. 분류 모드만 사용 가능합니다.
⚠️ NumPy 1.x로 다운그레이드하거나 Python 3.11 이하를 사용하세요.
```

또는 (NumPy 1.x가 설치된 경우):

```
✅ TACO 탐지 모듈 로드 완료
```

## 권장 사항

**프로덕션 환경:**
- Python 3.11 사용
- NumPy 1.26.3 설치
- 모든 기능 사용 가능

**개발 환경:**
- 현재 상태로도 분류 모드는 사용 가능
- 탐지 모드는 Python 3.11 환경에서 테스트

