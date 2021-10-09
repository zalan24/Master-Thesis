@echo off

call run_benchmarks.cmd
  if errorlevel 1 goto error

call run_static_benchmarks.cmd
  if errorlevel 1 goto error

goto EOF

:error
echo.
echo An error occured
pause
exit /b 1

:EOF

echo Success
pause
