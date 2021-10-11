
@echo off
SET Scene=..\data\scenes\static_scene.json

@REM SET resulultsFolder=results

@REM IF NOT exist %resulultsFolder% ( mkdir %resulultsFolder% )

@REM SET TempFolder=%resulultsFolder%\temp

SET CurrentBenchmark=benchmark_highstdDev_afterInput_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_afterInput_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_afterInput_unlimited_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_afterInput_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_afterInput_vsync_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_beforeInput_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_beforeInput_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_beforeInput_unlimited_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_beforeInput_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_beforeInput_vsync_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_device_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_device_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_device_unlimited_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_device_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_device_vsync_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_execDevice_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_execDevice_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_execDevice_unlimited_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_execDevice_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_execDevice_vsync_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_exec_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_exec_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_exec_unlimited_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_exec_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_highstdDev_exec_vsync_highPool
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_afterInput_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_afterInput_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_afterInput_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_beforeInput_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_beforeInput_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_beforeInput_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_device_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_device_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_device_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_execDevice_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_execDevice_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_execDevice_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_exec_control
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_exec_unlimited
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error
SET CurrentBenchmark=benchmark_nostdDev_exec_vsync
SET EngineOption=..\data\engineOptions\%CurrentBenchmark%.json
"../build/src/games/playground/Playground.exe" -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene %Scene% --options %EngineOption% --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt
  if errorlevel 1 goto error


goto EOF

:error
echo.
echo An error occured
pause
exit /b 1

:EOF