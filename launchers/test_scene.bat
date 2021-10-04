SET VK_LAYER_SETTINGS_PATH="../data/captureCfg"
SET VK_APIDUMP_OUTPUT_RANGE="0-10-1"
SET VK_APIDUMP_LOG_FILENAME="api_dump_log.txt"
start ../build/src/games/playground/Playground.exe -c../data/configs/playground_default.json -t../data/configs/trackingConfig.json -d../data/ --scene ../data/scenes/test_scene.json --options engineOptions.json --no_persistance --runtime_stats_persistance statistics/runtimeStats.bin --runtime_stats_game_exports statistics/runtimeStatsGameExports.bin --runtime_stats_cache statistics/runtimeStatsCache.bin -s../build/shaders/shaders.bin --report statistics/report.txt