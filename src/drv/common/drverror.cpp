#include "drverror.h"

#include <cstdlib>

#include <logger.h>

static drv::CallbackFunction callback;

void drv::set_callback(CallbackFunction func) {
    callback = func;
}

void drv::report_error(CallbackData* data) {
    if (callback)
        callback(data);
}

void drv::drv_assert(bool ok, const char* text) {
    if (ok)
        return;
    LOG_F(FATAL, "Assert error: %s", text ? text : "<0x0>");
    CallbackData data;
    data.text = text;
    data.type = drv::CallbackData::Type::ERROR;
    if (callback)
        callback(&data);
    std::abort();
}
