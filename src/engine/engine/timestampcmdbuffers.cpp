#include "timestampcmdbuffers.h"

#include <drverror.h>

drv::PipelineStages TimestampCmdBufferPool::get_required_stages(
  drv::PhysicalDevicePtr physicalDevice, drv::QueueFamilyPtr family) {
    drv::CommandTypeMask commandTypes = drv::get_command_type_mask(physicalDevice, family);
    drv::PipelineStages ret(0);
    ret.add(drv::PipelineStages::BOTTOM_OF_PIPE_BIT);
    ret.add(drv::PipelineStages::TOP_OF_PIPE_BIT);
    if (commandTypes & drv::CMD_TYPE_TRANSFER)
        ret.add(drv::PipelineStages::TRANSFER_BIT);
    if (commandTypes & drv::CMD_TYPE_COMPUTE)
        ret.add(drv::PipelineStages::COMPUTE_SHADER_BIT);
    if (commandTypes & drv::CMD_TYPE_GRAPHICS) {
        ret.add(drv::PipelineStages::COMPUTE_SHADER_BIT);
        ret.add(drv::PipelineStages::VERTEX_INPUT_BIT);
        ret.add(drv::PipelineStages::FRAGMENT_SHADER_BIT);
        ret.add(drv::PipelineStages::GEOMETRY_SHADER_BIT);
        ret.add(drv::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT);
        ret.add(drv::PipelineStages::EARLY_FRAGMENT_TESTS_BIT);
        ret.add(drv::PipelineStages::LATE_FRAGMENT_TESTS_BIT);
        ret.add(drv::PipelineStages::TESSELLATION_CONTROL_SHADER_BIT);
        ret.add(drv::PipelineStages::TESSELLATION_EVALUATION_SHADER_BIT);
        ret.add(drv::PipelineStages::DRAW_INDIRECT_BIT);
        ret.add(drv::PipelineStages::VERTEX_SHADER_BIT);
    }
    return ret;
}

drv::CommandPoolCreateInfo TimestampCmdBufferPool::get_cmd_buffer_pool_create_info() {
    drv::CommandPoolCreateInfo ret;
    ret.resetCommandBuffer = false;
    ret.transient = false;
    return ret;
}

drv::CommandBufferCreateInfo
  TimestampCmdBufferPool::CommandBufferData::get_cmd_buffer_create_info() {
    drv::CommandBufferCreateInfo ret;
    ret.type = drv::CommandBufferType::PRIMARY;
    ret.flags = 0;
    return ret;
}

TimestampCmdBufferPool::CommandBufferData::CommandBufferData(drv::LogicalDevicePtr _device,
                                                             drv::CommandPoolPtr _pool)
  : cmdBuffer(_device, _pool, get_cmd_buffer_create_info()) {
}

TimestampCmdBufferPool::TimestampCmdBufferPool(drv::PhysicalDevicePtr physicalDevice,
                                               drv::LogicalDevicePtr _device,
                                               drv::QueueFamilyPtr _family,
                                               uint32_t _maxFramesInFlight,
                                               uint32_t _queriesPerFrame)
  : device(_device),
    family(_family),
    trackedStages(get_required_stages(physicalDevice, family)),
    maxFramesInFlight(_maxFramesInFlight),
    cmdBufferPool(device, family, get_cmd_buffer_pool_create_info()),
    timestampQueryPool(device,
                       maxFramesInFlight * _queriesPerFrame * trackedStages.getStageCount()) {
    cmdBuffers.reserve(maxFramesInFlight * _queriesPerFrame);
    for (uint32_t i = 0; i < maxFramesInFlight * _queriesPerFrame; ++i) {
        cmdBuffers.emplace_back(device, cmdBufferPool);

        StackMemory::MemoryHandle<uint8_t> recorderMem(drv::get_cmd_buffer_recorder_size(),
                                                       TEMPMEM);
        PlacementPtr<drv::DrvCmdBufferRecorder> recorder = drv::create_cmd_buffer_recorder(
          recorderMem, physicalDevice, device, family, cmdBuffers[i].cmdBuffer, false, false);
        recorder->init(0);

        for (uint32_t j = 0; j < trackedStages.getStageCount(); ++j)
            recorder->cmdTimestamp(timestampQueryPool, i * maxFramesInFlight * _queriesPerFrame + j,
                                   trackedStages.getStage(j));
    }
}

TimestampCmdBufferPool::CmdBufferInfo TimestampCmdBufferPool::acquire(FrameId frameId) {
    if (cmdBuffers[nextIndex].useableFrom <= frameId) {
        CmdBufferInfo ret;
        ret.index = nextIndex;
        ret.cmdBuffer = cmdBuffers[nextIndex].cmdBuffer;
        cmdBuffers[nextIndex].useableFrom = frameId + maxFramesInFlight + 1;
        timestampQueryPool.reset(trackedStages.getStageCount() * ret.index,
                                 trackedStages.getStageCount());
        nextIndex = (nextIndex + 1) % cmdBuffers.size();
        return ret;
    }
    else
        return {};
}

DynamicTimestampCmdBufferPool::PerFamilyData::PerFamilyData(drv::PhysicalDevicePtr _physicalDevice,
                                                            drv::LogicalDevicePtr _device,
                                                            drv::QueueFamilyPtr _family,
                                                            uint32_t _maxFramesInFlight,
                                                            uint32_t queriesPerFrame) {
    pools.emplace_back(_physicalDevice, _device, _family, _maxFramesInFlight, queriesPerFrame);
}

DynamicTimestampCmdBufferPool::DynamicTimestampCmdBufferPool(drv::PhysicalDevicePtr _physicalDevice,
                                                             drv::LogicalDevicePtr _device,
                                                             uint32_t _maxFramesInFlight,
                                                             uint32_t _queriesPerFramePerPool)
  : physicalDevice(_physicalDevice),
    device(_device),
    maxFramesInFlight(_maxFramesInFlight),
    queriesPerFramePerPool(_queriesPerFramePerPool) {
}

TimestampCmdBufferPool::CmdBufferInfo DynamicTimestampCmdBufferPool::acquire(
  drv::QueueFamilyPtr family, FrameId frameId) {
    std::unique_lock<std::mutex> lock(mutex);
    auto itr = pools.find(family);
    PerFamilyData& familyPool =
      itr != pools.end()
        ? itr->second
        : pools
            .insert({family, PerFamilyData(physicalDevice, device, family, maxFramesInFlight,
                                           queriesPerFramePerPool)})
            .first->second;
    TimestampCmdBufferPool::CmdBufferInfo ret =
      familyPool.pools[familyPool.poolIndex].acquire(frameId);
    if (ret)
        return ret;
    familyPool.poolIndex = (familyPool.poolIndex + 1) % familyPool.pools.size();
    ret = familyPool.pools[familyPool.poolIndex].acquire(frameId);
    if (ret)
        return ret;
    familyPool.poolIndex = uint32_t(familyPool.pools.size());
    familyPool.pools.emplace_back(physicalDevice, device, family, maxFramesInFlight,
                                  queriesPerFramePerPool);
    ret = familyPool.pools[familyPool.poolIndex].acquire(frameId);
    drv::drv_assert(ret, "Could not acquire timestamp cmd buffer");
    return ret;
}
