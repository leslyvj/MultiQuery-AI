@echo off
echo ============================================================
echo   STARTING SERVER WITH GPU OPTIMIZATION
echo ============================================================
echo.

REM Set GPU optimization environment variables
set OLLAMA_NUM_GPU=999
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_FLASH_ATTENTION=1

echo GPU Optimization Settings:
echo   - OLLAMA_NUM_GPU = 999 (All GPU layers)
echo   - OLLAMA_MAX_LOADED_MODELS = 1 (Keep model loaded)
echo   - OLLAMA_FLASH_ATTENTION = 1 (Flash Attention enabled)
echo.
echo Starting server on http://127.0.0.1:8000...
echo.

cd /d E:\multimodal_rag_free
call .venv\Scripts\activate.bat

python run_server.py
pause
