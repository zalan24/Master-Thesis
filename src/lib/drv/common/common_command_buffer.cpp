#include "common_command_buffer.h"

#include <cstring>

#include "drverror.h"

void CommonCommandBuffer::add(drv::CommandData&& command) {
    commands.push_back(std::move(command));
}

void CommonCommandBuffer::add(const drv::CommandData& command) {
    commands.push_back(command);
}
