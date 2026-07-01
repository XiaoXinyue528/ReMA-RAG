@echo off
setlocal

cd /d "%~dp0.."

if not defined START set START=0
if not defined END set END=141

set PLANTUCD_DATA_PATH=data_splits/plantucd_test_142.json
set PYTHONUNBUFFERED=1

echo Data path: %PLANTUCD_DATA_PATH%
echo Index range: %START%..%END%
echo Output dir: plan_rag_extract_gpt4omini_plantucd_test142
echo Existing valid JSON outputs will be skipped.

python -u main.py ^
  --exp plan_rag_extract ^
  --model gpt4omini ^
  --dataset plantucd_test142 ^
  --start_index %START% ^
  --end_index %END% ^
  --gpus 0

python scripts\check_progress.py ^
  --output-dir plan_rag_extract_gpt4omini_plantucd_test142 ^
  --expected 142

endlocal
