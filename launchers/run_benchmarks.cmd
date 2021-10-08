@echo off

SET resulultsFolder=results

IF NOT exist %resulultsFolder% ( mkdir %resulultsFolder% )

SET CurrentDir=%resulultsFolder%\%DATE:/=-%_%TIME::=-%
mkdir %CurrentDir%

SET CurrentBenchmark=clean_control
SET EngineOption=..\data\engineOptions\benchmark_clean_control.json
SET Scene=..\data\scenes\clean_benchmark_scene.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET BenchmarkFolder=%CurrentDir%\%CurrentBenchmark%
mkdir %BenchmarkFolder%
python ../scripts/generate_plots.py benchmarks/benchmark.csv %BenchmarkFolder%
XCOPY /q logs\all.log %BenchmarkFolder%
XCOPY /q %Scene% %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%

SET CurrentBenchmark=clean_unlimited
SET EngineOption=..\data\engineOptions\benchmark_clean_unlimited.json
SET Scene=..\data\scenes\clean_benchmark_scene.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET BenchmarkFolder=%CurrentDir%\%CurrentBenchmark%
mkdir %BenchmarkFolder%
python ../scripts/generate_plots.py benchmarks/benchmark.csv %BenchmarkFolder%
XCOPY /q logs\all.log %BenchmarkFolder%
XCOPY /q %Scene% %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%

SET CurrentBenchmark=clean_limited
SET EngineOption=..\data\engineOptions\benchmark_clean_limited.json
SET Scene=..\data\scenes\clean_benchmark_scene.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET BenchmarkFolder=%CurrentDir%\%CurrentBenchmark%
mkdir %BenchmarkFolder%
python ../scripts/generate_plots.py benchmarks/benchmark.csv %BenchmarkFolder%
XCOPY /q logs\all.log %BenchmarkFolder%
XCOPY /q %Scene% %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%

SET CurrentBenchmark=clean_vsync
SET EngineOption=..\data\engineOptions\benchmark_clean_vsync.json
SET Scene=..\data\scenes\clean_benchmark_scene.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET BenchmarkFolder=%CurrentDir%\%CurrentBenchmark%
mkdir %BenchmarkFolder%
python ../scripts/generate_plots.py benchmarks/benchmark.csv %BenchmarkFolder%
XCOPY /q logs\all.log %BenchmarkFolder%
XCOPY /q %Scene% %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%

goto EOF

:error
echo.
echo An error occured
pause
exit /b 1

:EOF

echo Success
pause
