@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64
cmake -S . -B out\build\vs-2026-fresh -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Debug
if exist out\build\vs-2026-fresh\Capsaicin.sln (
  start "" "%CD%\out\build\vs-2026-fresh\Capsaicin.sln"
) else (
  echo "Solution not found; listing build folder:"
  dir out\build\vs-2026-fresh
)
pause
