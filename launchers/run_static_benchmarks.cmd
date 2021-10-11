@echo off

SET resulultsFolder=results

IF NOT exist %resulultsFolder% ( mkdir %resulultsFolder% )

SET CurrentDir=%resulultsFolder%\static_benchmark_%DATE:/=-%_%TIME::=-%
SET CurrentDir=%CurrentDir: =_%
SET TempFolder=%resulultsFolder%\temp
mkdir %CurrentDir%


SET BenchmarkFolder=%CurrentDir%
mkdir %BenchmarkFolder%
SET Scene=..\data\scenes\static_benchmark_scene.json
XCOPY /q %Scene% %BenchmarkFolder%

@REM SET CurrentBenchmark=benchmark_highstdDev_afterInput_control
@REM SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
@REM "../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
@REM   if errorlevel 1 goto error
@REM python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
@REM   if errorlevel 1 goto error
@REM XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
@REM XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
@REM ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
@REM ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
@REM   if errorlevel 1 goto error



SET CurrentBenchmark=benchmark_highstdDev_afterInput_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_afterInput_limited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_afterInput_unlimited_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_afterInput_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_afterInput_vsync_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_afterInput_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_beforeInput_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_beforeInput_limited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_beforeInput_unlimited_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_beforeInput_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_beforeInput_vsync_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_beforeInput_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_device_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_device_limited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_device_unlimited_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_device_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_device_vsync_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_device_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_exec_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_exec_limited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_exec_unlimited_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_exec_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_exec_vsync_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_exec_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_execDevice_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_execDevice_limited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_execDevice_unlimited_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_execDevice_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_execDevice_vsync_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_highstdDev_execDevice_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_afterInput_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_afterInput_limited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_afterInput_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_afterInput_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_beforeInput_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_beforeInput_limited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_beforeInput_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_beforeInput_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_device_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_device_limited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_device_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_device_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_exec_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_exec_limited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_exec_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_exec_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_execDevice_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_execDevice_limited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_execDevice_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error


SET CurrentBenchmark=benchmark_nostdDev_execDevice_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
python ../scripts/generate_plots.py benchmarks/benchmark.csv %TempFolder%
  if errorlevel 1 goto error
XCOPY /q %TempFolder%\downsampled.csv %BenchmarkFolder%
XCOPY /q %TempFolder%\source.csv %BenchmarkFolder%
XCOPY /q %EngineOption% %BenchmarkFolder%
ren %BenchmarkFolder%\downsampled.csv %CurrentBenchmark%_downsampled.csv
ren %BenchmarkFolder%\source.csv %CurrentBenchmark%_source.csv
  if errorlevel 1 goto error

python ../scripts/generate_static_table.py %BenchmarkFolder% %BenchmarkFolder%
  if errorlevel 1 goto error


goto EOF

:error
echo.
echo An error occured
pause
exit /b 1

:EOF
