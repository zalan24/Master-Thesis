#pragma once

#include <memory>
#include <vector>

#include <exclusive.h>

#include "drv.h"

namespace drv
{
class CommandListBuilder : private Exclusive
{
 public:
    explicit CommandListBuilder(LogicalDevicePtr device);

    CommandList getCommandList();

    void add(CommandData&& commandInfo);
    void add(const CommandData& commandInfo);

    // use these with 'OPTIONS_WITH_DEBUG_INFO'

    void transfer(BufferPtr dst, BufferPtr src, CommandOptions options);
    void transfer(BufferPtr dst, BufferPtr src, DeviceSize size, CommandOptions options);
    void transfer(BufferPtr dst, BufferPtr src, DeviceSize offset, DeviceSize size,
                  CommandOptions options);
    void transfer(BufferPtr dst, BufferPtr src, DeviceSize dstOffset, DeviceSize srcOffset,
                  DeviceSize size, CommandOptions options);
    void transfer(const CommandOptions_transfer& options);

    void dispatch(unsigned int sizeX, CommandOptions options);
    void dispatch(unsigned int sizeX, unsigned int sizeY, CommandOptions options);
    void dispatch(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ,
                  CommandOptions options);
    void dispatch(const CommandOptions_dispatch& options);

 private:
    LogicalDevicePtr device;
    std::vector<CommandData> commands;
};

}  // namespace drv
