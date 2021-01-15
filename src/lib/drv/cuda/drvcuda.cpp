#include "drvcuda.h"

#include <drverror.h>
#include <drvfunctions.h>

#include "cuda_commands.h"
#include "cudactx.h"

static bool initialized = false;
using CommandFunction = bool (*)(const drv::CommandExecutionData* data, const drv::CommandData*);
static CommandFunction commands[drv::COMMAND_FUNCTION_COUNT];

static bool init() {
    if (initialized)
        return false;
    commands[drv::CMD_TRANSFER] = drv_cuda::transfer;
    commands[drv::CMD_BIND_COMPUTE_PIPELINE] = drv_cuda::bind_compute_pipeline;
    commands[drv::CMD_DISPATCH] = drv_cuda::dispatch;
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

bool drv_cuda::command(const drv::CommandData* cmd, const drv::CommandExecutionData* data) {
#ifdef DEBUG
    drv::drv_assert(initialized, "Driver not initialized");
    if (!initialized)
        return false;
#endif
    return commands[cmd->cmd](data, cmd);
}

void drv_cuda::register_cuda_drv(drv::DrvFunctions& functions) {
    FILL_OUT_DRV_FUNCTIONS(functions)
}
