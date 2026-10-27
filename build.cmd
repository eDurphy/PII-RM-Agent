@echo off
rem ==========================================================================
rem  PII-AGENT v003 - spatial-multimodal rebuild. DOPPIO CLIC o:  build.cmd [clean]
rem  NO autobuilder, NO REPL, NO OCR (no Tesseract/OpenCV): image XObjects are
rem  decoded with the vendored single-header stb_image.h (src\pii\stb_image.h,
rem  public domain) via src\pii\pii_img.c (the ONLY STB_IMAGE_IMPLEMENTATION TU).
rem
rem    bin\pii-agent-flash.exe - mask/unmask/purge/selftest. Links math
rem                              (forward) + spatial-visual hybrid forward
rem                              (PII_HYBRID_SPATIAL) + vault + mask; train and
rem                              backprop compiled OUT (-DPII_FLASH_NO_TRAIN).
rem                              Loads I_*.bin into RAM, never opens S_*.
rem    bin\pii-agent.exe       - same four commands + train <path> [X]
rem                              (0/inf/omitted = infinite until Ctrl+C).
rem                              Links BPE + full math incl. backprop + hybrid
rem                              forward + spatial-aware training (zone
rem                              markers, PII_HYBRID_TRAIN); I+S.
rem  Pesi esterni: default data/weights/, override --weights-dir / PII_WEIGHTS_DIR.
rem  Spatial math deps: NONE beyond v002 (integer-only patch projection in
rem  src\eng\nn_spatial.h + qat_spatial_bias_add/qat_patch_token in qatnn_core.c;
rem  header-only stb needs no link lib). Flag: -O3 -mavx2 -mtune=native
rem  -pthread -static. MAI -fopenmp.
rem ==========================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 ( echo Non riesco a entrare in %~dp0 & exit /b 1 )

if /i "%~1"=="clean" (
  echo rimuovo bin\
  if exist bin rmdir /s /q bin
)
if not exist bin mkdir bin
if not exist data\weights mkdir data\weights

where gcc >nul 2>nul
if not errorlevel 1 goto :have_gcc
for %%D in ("C:\msys64\mingw64\bin\gcc.exe" "C:\msys64\ucrt64\bin\gcc.exe") do (
  if exist %%D (
    set "GCCDIR=%%~dpD"
    set "PATH=!GCCDIR!;!PATH!"
    goto :have_gcc
  )
)
echo gcc NON TROVATO (installa MSYS2 mingw64). & exit /b 1

:have_gcc
echo ==================================================================
echo  PII-AGENT v003 - spatial-multimodal build PURE C (+stb_image.h)
gcc -dumpversion
if errorlevel 1 ( echo gcc non eseguibile. & exit /b 1 )
echo ==================================================================

set "CFLAGS=-std=gnu99 -O3 -mavx2 -mtune=native -pthread -Wall -static -Isrc/core -Isrc/pii -Isrc/eng"
set "SRC_PII=src\pii\pii_agent.c src\pii\pii_mask.c src\pii\pii_crypto.c src\pii\pii_pdf.c src\pii\pii_img.c src\pii\pii_train.c src\pii\cli_args.c"
set "SRC_FLASH=src\core\qatnn_core.c src\eng\eng_infer_tu.c %SRC_PII% src\pii\flash_main.c"
set "SRC_FULL=src\core\qatnn_core.c src\eng\eng_train_tu.c %SRC_PII% src\pii\full_main.c"

echo --- [1/2] pii-agent-flash (inference + hybrid spatial forward, NO train) ---
gcc %CFLAGS% -DPII_V002_UNIFIED -DPII_FLASH_NO_TRAIN %SRC_FLASH% -o "bin\pii-agent-flash.tmp.exe" -lz 2> "bin\pii-agent-flash.log"
if errorlevel 1 ( echo FALLITO - vedi bin\pii-agent-flash.log & type "bin\pii-agent-flash.log" & exit /b 1 )
if exist "bin\pii-agent-flash.exe" del "bin\pii-agent-flash.exe"
if exist "bin\pii-agent-flash.exe" ( echo Non posso sovrascrivere: e' in esecuzione. & del "bin\pii-agent-flash.tmp.exe" & exit /b 1 )
ren "bin\pii-agent-flash.tmp.exe" "pii-agent-flash.exe"
echo OK bin\pii-agent-flash.exe

echo --- [2/2] pii-agent (full engine + train + spatial, NO repl) ---
gcc %CFLAGS% -DPII_V002_UNIFIED %SRC_FULL% -o "bin\pii-agent.tmp.exe" -lz -lm -lpsapi 2> "bin\pii-agent.log"
if errorlevel 1 ( echo FALLITO - vedi bin\pii-agent.log & type "bin\pii-agent.log" & exit /b 1 )
if exist "bin\pii-agent.exe" del "bin\pii-agent.exe"
if exist "bin\pii-agent.exe" ( echo Non posso sovrascrivere: e' in esecuzione. & del "bin\pii-agent.tmp.exe" & exit /b 1 )
ren "bin\pii-agent.tmp.exe" "pii-agent.exe"
echo OK bin\pii-agent.exe

for %%F in ("bin\pii-agent-flash.exe" "bin\pii-agent.exe") do echo    %%~nxF  %%~zF byte
echo.
echo Fatto. Prova:  PII.cmd (menu)  ^&  selftest voce 1.
