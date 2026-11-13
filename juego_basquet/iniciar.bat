@echo off
cd /d "%~dp0"
py -3.13 juego.py
if errorlevel 1 pause
