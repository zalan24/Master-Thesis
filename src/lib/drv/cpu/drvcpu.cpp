#include "drvcpu.h"

#include <drverror.h>
#include <drvfunctions.h>

#include "cpu.h"
#include "cpu_commands.h"

static bool initialized = false;
using CommandFunction = bool (*)(const drv::CommandData*);
static CommandFunction commands[drv::COMMAND_FUNCTION_COUNT];

static bool init() {
    if (initialized)
        return false;
    commands[drv::CMD_TRANSFER] = drv_cpu::transfer;
    commands[drv::CMD_BIND_COMPUTE_PIPELINE] = drv_cpu::bind_compute_pipeline;
    commands[drv::CMD_DISPATCH] = drv_cpu::dispatch;
    static_assert(drv::CMD_DISPATCH + 1 == drv::COMMAND_FUNCTION_COUNT, "Update this");
    initialized = true;
    return true;
}

static bool close() {
    if (!initialized)
        return false;
    initialized = false;
    return true;
}

bool drv_cpu::command_impl(const drv::CommandData* cmd) {
    return commands[cmd->cmd](cmd);
}

void drv_cpu::register_cpu_drv(drv::DrvFunctions& functions) {
    FILL_OUT_DRV_FUNCTIONS(functions)
}
