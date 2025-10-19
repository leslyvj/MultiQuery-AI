@echo off
echo ============================================================
echo Starting Multimodal RAG Server
echo ============================================================
echo.
cd /d E:\multimodal_rag_free
call .venv\Scripts\activate.bat
echo.
echo Server will start on: http://127.0.0.1:8000
echo.
echo Fixed Issues:
echo   - FFmpeg automatically added to PATH
echo   - Tesseract errors handled gracefully  
echo   - Audio and Image uploads now work!
echo.
echo ============================================================
echo.
python run_server.py
pause
