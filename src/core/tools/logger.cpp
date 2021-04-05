#include "logger.h"

#include <filesystem>

namespace fs = std::filesystem;

Logger::Logger(int argc, char* argv[], const std::string& logDir) {
    fs::path logs{logDir};
    if (!fs::exists(logs))
        fs::create_directories(logs);
    loguru::init(argc, argv);

    static_assert(ENGINE == 1);
    static_assert(DRIVER_API == 2);
    static_assert(COMMAND_LOG == 9);

    loguru::add_file((logs / fs::path{"all.log"}).string().c_str(), loguru::Truncate,
                     loguru::Verbosity_MAX);
    loguru::add_file((logs / fs::path{"warning.log"}).string().c_str(), loguru::Truncate,
                     loguru::Verbosity_WARNING);
    loguru::add_file((logs / fs::path{"error.log"}).string().c_str(), loguru::Truncate,
                     loguru::Verbosity_ERROR);
    loguru::add_file((logs / fs::path{"fatal.log"}).string().c_str(), loguru::Truncate,
                     loguru::Verbosity_FATAL);
    loguru::add_file((logs / fs::path{"engine.log"}).string().c_str(), loguru::Truncate,
                     loguru::Verbosity_1);
    loguru::add_file((logs / fs::path{"driver_api.log"}).string().c_str(), loguru::Truncate,
                     loguru::Verbosity_2);
    loguru::add_file((logs / fs::path{"command_log.log"}).string().c_str(), loguru::Truncate,
                     loguru::Verbosity_9);
    loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
    loguru::g_colorlogtostderr = true;

    loguru::set_verbosity_to_name_callback([](loguru::Verbosity verbosity) -> const char* {
        switch (verbosity) {
            case loguru::Verbosity_INVALID:
                return "Invalid";
            case loguru::Verbosity_OFF:
                return "Off";
            case loguru::Verbosity_FATAL:
                return "Fatal";
            case loguru::Verbosity_ERROR:
                return "Error";
            case loguru::Verbosity_WARNING:
                return "Warning";
            case loguru::Verbosity_INFO:
                // case loguru::Verbosity_0:
                return "Info";
            case loguru::Verbosity_1:
                return "Engine";
            case loguru::Verbosity_2:
                return "DriverApi";
            case loguru::Verbosity_3:
                return "3";
            case loguru::Verbosity_4:
                return "4";
            case loguru::Verbosity_5:
                return "5";
            case loguru::Verbosity_6:
                return "6";
            case loguru::Verbosity_7:
                return "7";
            case loguru::Verbosity_8:
                return "8";
            case loguru::Verbosity_9:
                //   case loguru::Verbosity_MAX:
                return "CommandLog";
        }
        return "Invalid";
    });

    loguru::set_name_to_verbosity_callback([](const char* name) -> loguru::Verbosity {
        if (strcmp(name, "Invalid"))
            return loguru::Verbosity_INVALID;
        if (strcmp(name, "Warning"))
            return loguru::Verbosity_WARNING;
        if (strcmp(name, "Error"))
            return loguru::Verbosity_ERROR;
        if (strcmp(name, "Fatal"))
            return loguru::Verbosity_FATAL;
        if (strcmp(name, "Info"))
            return loguru::Verbosity_INFO;
        if (strcmp(name, "Engine"))
            return loguru::Verbosity_1;
        if (strcmp(name, "DriverApi"))
            return loguru::Verbosity_2;
        if (strcmp(name, "3"))
            return loguru::Verbosity_3;
        if (strcmp(name, "4"))
            return loguru::Verbosity_4;
        if (strcmp(name, "5"))
            return loguru::Verbosity_5;
        if (strcmp(name, "6"))
            return loguru::Verbosity_6;
        if (strcmp(name, "7"))
            return loguru::Verbosity_7;
        if (strcmp(name, "8"))
            return loguru::Verbosity_8;
        if (strcmp(name, "CommandLog"))
            return loguru::Verbosity_9;
        return loguru::Verbosity_INVALID;
    });

    loguru::set_fatal_handler([](const loguru::Message& message) {
        throw std::runtime_error(std::string(message.prefix) + message.message);
    });
}

namespace fs = std::filesystem;

Logger::Logger(Logger&& other) : valid(other.valid) {
    other.valid = false;
}

namespace fs = std::filesystem;

Logger& Logger::operator=(Logger&& other) {
    if (this == &other)
        return *this;
    close();
    valid = other.valid;
    other.valid = false;
    return *this;
}

namespace fs = std::filesystem;

Logger::~Logger() {
    close();
}

namespace fs = std::filesystem;

void Logger::close() {
    if (valid) {
        loguru::shutdown();
        valid = false;
    }
}
