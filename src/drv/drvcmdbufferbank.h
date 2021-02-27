#pragma once

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "drv_wrappers.h"

namespace drv
{
struct CommandBufferBankGroupInfo
{
    QueueFamilyPtr family;
    bool render_pass_continueos;
    CommandBufferType type;
    std::thread::id thread;
    CommandBufferBankGroupInfo(QueueFamilyPtr _family, bool _render_pass_continueos,
                               CommandBufferType _type)
      : family(_family),
        render_pass_continueos(_render_pass_continueos),
        type(_type),
        thread(std::this_thread::get_id()) {}
    bool operator==(const CommandBufferBankGroupInfo& lhs) const {
        return family == lhs.family && render_pass_continueos == lhs.render_pass_continueos
               && type == lhs.type && thread == lhs.thread;
    }
};
}  // namespace drv

namespace std
{
template <>
struct hash<drv::CommandBufferBankGroupInfo>
{
    std::size_t operator()(const drv::CommandBufferBankGroupInfo& s) const noexcept {
        return std::hash<drv::QueueFamilyPtr>{}(s.family)
               ^ std::hash<bool>{}(s.render_pass_continueos)
               ^ std::hash<std::thread::id>{}(s.thread)
               ^ std::hash<drv::CommandBufferType>{}(s.type);
    }
};
}  // namespace std

namespace drv
{
// Ask for a cmd buffer
// use the cmd buffer
// give the cmd buffer back
class CommandBufferCirculator
{
 public:
    struct CommandBufferHandle
    {
        CommandBufferCirculator* circulator = nullptr;
        QueueFamilyPtr family = NULL_HANDLE;
        size_t bufferIndex = 0;
        CommandBufferPtr commandBufferPtr = NULL_HANDLE;
        operator bool() const { return commandBufferPtr != NULL_HANDLE; }
    };

    CommandBufferCirculator(LogicalDevicePtr device, QueueFamilyPtr family, CommandBufferType type,
                            bool render_pass_continueos);

    ~CommandBufferCirculator();

    CommandBufferCirculator(const CommandBufferCirculator&) = delete;
    CommandBufferCirculator& operator=(const CommandBufferCirculator&) = delete;
    CommandBufferCirculator(CommandBufferCirculator&&) = delete;
    CommandBufferCirculator& operator=(CommandBufferCirculator&&) = delete;

    CommandBufferHandle acquire();
    bool tryAcquire(CommandBufferHandle& handle);

    void startExecution(CommandBufferHandle& handle);
    void finished(CommandBufferHandle&& handle);

    static CommandPoolCreateInfo get_create_info();

 private:
    enum CommandBufferState
    {
        READY = 0,
        RECORDING,
        PENDING
    };
    struct CommandBufferData
    {
        CommandBuffer commandBuffer;
        CommandBufferPtr commandBufferPtr;
        std::atomic<CommandBufferState> state;
        CommandBufferData(CommandBuffer&& _commandBuffer, CommandBufferPtr _commandBufferPtr,
                          CommandBufferState _state)
          : commandBuffer(std::move(_commandBuffer)),
            commandBufferPtr(_commandBufferPtr),
            state(_state) {}
        CommandBufferData(CommandBufferData&&);
        CommandBufferData& operator=(CommandBufferData&&);
    };

    LogicalDevicePtr device;
    QueueFamilyPtr family;
    CommandPool pool;
    drv::CommandBufferType type;
    bool render_pass_continueos;
    std::vector<CommandBufferData> commandBuffers;
    std::atomic<size_t> acquiredStates = 0;
    mutable std::shared_mutex mutex;  // for releasing the command buffer from a different thread
};

class CommandBufferBank
{
 public:
    CommandBufferBank(LogicalDevicePtr device);

    CommandBufferBank(const CommandBufferBank&) = delete;
    CommandBufferBank& operator=(const CommandBufferBank&) = delete;
    CommandBufferBank(CommandBufferBank&&) = delete;
    CommandBufferBank& operator=(CommandBufferBank&&) = delete;

    CommandBufferCirculator::CommandBufferHandle acquire(
      const CommandBufferBankGroupInfo& groupInfo);
    bool tryAcquire(CommandBufferCirculator::CommandBufferHandle& handle,
                    const CommandBufferBankGroupInfo& groupInfo);

 private:
    LogicalDevicePtr device;
    std::unordered_map<CommandBufferBankGroupInfo, std::unique_ptr<CommandBufferCirculator>> pools;
    mutable std::shared_mutex mutex;
};

}  // namespace drv
