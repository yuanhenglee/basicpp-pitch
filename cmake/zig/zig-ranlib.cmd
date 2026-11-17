@echo off
if not defined ZIG_EXE set "ZIG_EXE=%~dp0..\..\.venv\Lib\site-packages\ziglang\zig.exe"
"%ZIG_EXE%" ranlib %*
