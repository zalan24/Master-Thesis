#pragma once

#include <string>

#include <loguru.hpp>

struct Logger
{
    Logger(int argc, char* argv[], const std::string& logDir);
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&);
    Logger& operator=(Logger&&);
    ~Logger();
    bool valid = true;
    void close();
};

#define ENGINE 1
#define DRIVER_API 2
#define COMMAND_LOG 9

#define LOG_ENGINE(...) LOG_F(1, __VA_ARGS__)
#define LOG_DRIVER_API(...) LOG_F(2, __VA_ARGS__)
#define LOG_COMMAND(...) LOG_F(9, __VA_ARGS__)
