#define __STDC_WANT_LIB_EXT1__ 1

#include "persistance.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <sstream>

void SingleExecutionData::start() {
    // Such clean, such simple, so much print time
    std::stringstream ss;
    auto in_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm timeinfo;
    localtime_s(&timeinfo, &in_time_t);
    ss << std::put_time(&timeinfo, "%Y-%m-%d %X");
    startTime = ss.str();
    endTime = "";
    frameCount = 0;
    sampleInputCount = 0;
    submissionCount = 0;
    submissionCorrections.clear();
    attachmentCorrections.clear();
}

void SingleExecutionData::stop() {
    std::stringstream ss;
    auto in_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm timeinfo;
    localtime_s(&timeinfo, &in_time_t);
    ss << std::put_time(&timeinfo, "%Y-%m-%d %X");
    endTime = ss.str();
}
