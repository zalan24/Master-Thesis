#pragma once

namespace drv
{
struct CallbackData
{
    const char* text;
    enum class Type
    {
        VERBOSE,
        NOTE,
        WARNING,
        ERROR,
        FATAL
    } type;
};
using CallbackFunction = void (*)(const CallbackData*);

void set_callback(CallbackFunction func);
void report_error(CallbackData* data);
// TODO currently a lot of string are constructed for this...
void drv_assert(bool ok, const char* text = nullptr);
}  // namespace drv
