@echo off
echo NumPy 다운그레이드 중...
echo.
echo 현재 NumPy 버전 확인:
python -c "import numpy; print(f'Current: {numpy.__version__}')"
echo.
echo NumPy 1.x로 다운그레이드 중...
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
echo.
echo 완료! 새로운 버전 확인:
python -c "import numpy; print(f'New: {numpy.__version__}')"
echo.
pause

