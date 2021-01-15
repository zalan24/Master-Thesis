#include "drv_command_list_builder.h"

#include <drverror.h>

using namespace drv;

CommandListBuilder::CommandListBuilder(LogicalDevicePtr _device) : device(_device) {
}

CommandList CommandListBuilder::getCommandList() {
    CommandList ret;
    ret.commandCount = static_cast<unsigned int>(commands.size());
    ret.commands = commands.data();
    return ret;
}

void CommandListBuilder::add(CommandData&& commandInfo) {
    CHECK_THREAD;
    commands.push_back(std::move(commandInfo));
}

void CommandListBuilder::add(const CommandData& commandInfo) {
    CHECK_THREAD;
    commands.push_back(commandInfo);
}

void CommandListBuilder::transfer(BufferPtr dst, BufferPtr src, CommandOptions _options) {
    CommandOptions_transfer options;
    *static_cast<CommandOptions*>(&options) = _options;
    BufferMemoryInfo memoryInfo = get_buffer_memory_info(device, dst);
    options.dst = dst;
    options.src = src;
    options.numRegions = 1;
    options.regions[0].dstOffset = 0;
    options.regions[0].srcOffset = 0;
    options.regions[0].size = memoryInfo.size;
    transfer(options);
}

void CommandListBuilder::transfer(BufferPtr dst, BufferPtr src, DeviceSize size,
                                  CommandOptions _options) {
    CommandOptions_transfer options;
    *static_cast<CommandOptions*>(&options) = _options;
    options.dst = dst;
    options.src = src;
    options.numRegions = 1;
    options.regions[0].dstOffset = 0;
    options.regions[0].srcOffset = 0;
    options.regions[0].size = size;
    transfer(options);
}

void CommandListBuilder::transfer(BufferPtr dst, BufferPtr src, DeviceSize offset, DeviceSize size,
                                  CommandOptions _options) {
    CommandOptions_transfer options;
    *static_cast<CommandOptions*>(&options) = _options;
    options.dst = dst;
    options.src = src;
    options.numRegions = 1;
    options.regions[0].dstOffset = offset;
    options.regions[0].srcOffset = offset;
    options.regions[0].size = size;
    transfer(options);
}

void CommandListBuilder::transfer(BufferPtr dst, BufferPtr src, DeviceSize dstOffset,
                                  DeviceSize srcOffset, DeviceSize size, CommandOptions _options) {
    CommandOptions_transfer options;
    *static_cast<CommandOptions*>(&options) = _options;
    options.dst = dst;
    options.src = src;
    options.numRegions = 1;
    options.regions[0].dstOffset = dstOffset;
    options.regions[0].srcOffset = srcOffset;
    options.regions[0].size = size;
    transfer(options);
}

void CommandListBuilder::transfer(const CommandOptions_transfer& options) {
    CHECK_THREAD;
    add(CommandData{options});
}

void CommandListBuilder::dispatch(unsigned int sizeX, CommandOptions _options) {
    dispatch(sizeX, 1, 1, _options);
}

void CommandListBuilder::dispatch(unsigned int sizeX, unsigned int sizeY, CommandOptions _options) {
    dispatch(sizeX, sizeY, 1, _options);
}

void CommandListBuilder::dispatch(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ,
                                  CommandOptions _options) {
    CommandOptions_dispatch options;
    *static_cast<CommandOptions*>(&options) = _options;
    options.sizeX = sizeX;
    options.sizeY = sizeY;
    options.sizeZ = sizeZ;
    dispatch(options);
}

void CommandListBuilder::dispatch(const CommandOptions_dispatch& options) {
    CHECK_THREAD;
    add(CommandData{options});
}
