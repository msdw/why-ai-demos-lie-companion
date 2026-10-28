@echo off
cd /d "%~dp0"
echo ======================================================================
echo 1. Removing internal build/audit tools from Git tracking...
echo ======================================================================
git rm -f build_epub_pdf_and_kdp.py local_ai_detector_service.py run_claude_and_codex_audit.py 2>nul
git rm --cached push_to_github.bat cleanup_repo.bat 2>nul

echo.
echo ======================================================================
echo 2. Staging MIT LICENSE, README.md, and .gitignore...
echo ======================================================================
git add LICENSE README.md .gitignore

echo.
echo ======================================================================
echo 3. Committing and Pushing to origin main...
echo ======================================================================
git commit -m "feat: publish official MIT license and clean internal tooling from companion vault"
git push origin main

echo.
echo ======================================================================
echo SUCCESS: Companion Vault cleaned and MIT License published!
echo https://github.com/msdw/why-ai-demos-lie-companion
echo ======================================================================
pause
