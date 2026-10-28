@echo off
setlocal enabledelayedexpansion
title Push Companion Repo to GitHub

echo ======================================================================
echo Pushing Companion Verification Vault to GitHub:
echo https://github.com/msdw/why-ai-demos-lie-companion
echo ======================================================================

cd /d "%~dp0"

echo [1/4] Checking Git Status...
git status

echo.
echo [2/4] Staging All Files...
git add -A

echo.
echo [3/4] Committing Changes...
git commit -m "feat: release official companion verification vault for Why AI Demos Lie"

echo.
echo [4/4] Pushing to Remote origin main...
git remote set-url origin https://github.com/msdw/why-ai-demos-lie-companion.git
git branch -M main
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo SUCCESS: Repository pushed successfully!
    echo Remote: https://github.com/msdw/why-ai-demos-lie-companion
    echo ======================================================================
) else (
    echo.
    echo ======================================================================
    echo ERROR: Push failed with exit code %ERRORLEVEL%.
    echo Please verify your GitHub credentials or SSH keys.
    echo ======================================================================
)

pause

