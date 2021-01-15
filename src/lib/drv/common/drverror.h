#pragma once

#include "drvtypes.h"

namespace drv
{
void set_callback(CallbackFunction func);
void report_error(CallbackData* data);
void drv_assert(bool ok, const char* text = nullptr);
}  // namespace drv
