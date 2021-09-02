#include "drvvulkan.h"

#include <sstream>
#include <vector>

#include <vulkan/vulkan.h>

#include <features.h>
#include <logger.h>

#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_instance.h"

drv::LogicalDevicePtr DrvVulkan::create_logical_device(const drv::LogicalDeviceCreateInfo* info) {
    std::vector<VkDeviceQueueCreateInfo> queues(info->queueInfoCount);
    std::unordered_map<drv::QueueFamilyPtr, std::mutex> queueFamilyMutexes;
    LOG_DRIVER_API("Creating logical device with queues <%p>: %d",
                   static_cast<const void*>(convertPhysicalDevice(info->physicalDevice)),
                   info->queueInfoCount);
    for (unsigned int i = 0; i < info->queueInfoCount; ++i) {
        std::stringstream priorities;
        for (uint32_t j = 0; j < info->queueInfoPtr[i].count; ++j)
            priorities << info->queueInfoPtr[i].prioritiesPtr[j] << " ";
        LOG_DRIVER_API("#%d/%d: Family:%d, count:%d, priorities: { %s}", i + 1,
                       info->queueInfoCount, info->queueInfoPtr[i].family,
                       info->queueInfoPtr[i].count, priorities.str().c_str());
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = convertFamilyToVk(info->queueInfoPtr[i].family);
        queueCreateInfo.queueCount = info->queueInfoPtr[i].count;
        queueCreateInfo.pQueuePriorities = info->queueInfoPtr[i].prioritiesPtr;

        queueFamilyMutexes[info->queueInfoPtr[i].family];  // init mutex for family

        queues[i] = queueCreateInfo;
    }

    uint32_t deviceExtensionCount;
    VkResult result = vkEnumerateDeviceExtensionProperties(
      convertPhysicalDevice(info->physicalDevice), nullptr, &deviceExtensionCount, nullptr);
    drv::drv_assert(result == VK_SUCCESS, "Could not query device extensions");
    StackMemory::MemoryHandle<VkExtensionProperties> deviceExtensions(deviceExtensionCount,
                                                                      TEMPMEM);
    result = vkEnumerateDeviceExtensionProperties(convertPhysicalDevice(info->physicalDevice),
                                                  nullptr, &deviceExtensionCount, deviceExtensions);
    drv::drv_assert(result == VK_SUCCESS, "Could not query device extensions");
    LOG_DRIVER_API("Available device extensions:");
    for (uint32_t i = 0; i < deviceExtensionCount; ++i)
        LOG_DRIVER_API(" - %s", deviceExtensions[i].extensionName);

    VkPhysicalDeviceFeatures deviceFeatures = {};

    VkPhysicalDeviceVulkan12Features device12Features = {};
    device12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    device12Features.pNext = nullptr;
    device12Features.timelineSemaphore = VK_TRUE;
    device12Features.hostQueryReset = VK_TRUE;

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &device12Features;
    createInfo.pQueueCreateInfos = queues.data();
    createInfo.queueCreateInfoCount = static_cast<unsigned int>(queues.size());

    createInfo.pEnabledFeatures = &deviceFeatures;

    std::vector<const char*> extensions = {};
    if (info->extensions.values.extensions.swapchain)
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (featureconfig::params.shaderPrint)
        extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);

    LOG_DRIVER_API("Used device extensions:");
    for (const char* ext : extensions)
        LOG_DRIVER_API(" - %s", ext);
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    createInfo.enabledLayerCount = 0;

    VkDevice device;
    result =
      vkCreateDevice(convertPhysicalDevice(info->physicalDevice), &createInfo, nullptr, &device);
    drv::drv_assert(result == VK_SUCCESS, "Logical device could not be created");
    LOG_DRIVER_API("Logical device created <%p> for physical device: %p",
                   static_cast<const void*>(device),
                   static_cast<const void*>(convertPhysicalDevice(info->physicalDevice)));

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(convertPhysicalDevice(info->physicalDevice), &deviceProperties);

    drv::LogicalDevicePtr ret = drv::store_ptr<drv::LogicalDevicePtr>(device);
    {
        std::unique_lock<std::mutex> lock(devicesDataMutex);
        auto& data = devicesData[ret] = LogicalDeviceData(this, info->physicalDevice, ret);
        data.queueFamilyMutexes = std::move(queueFamilyMutexes);
        data.timestampPeriod = deviceProperties.limits.timestampPeriod;
    }
    return ret;
}

bool DrvVulkan::delete_logical_device(drv::LogicalDevicePtr device) {
    {
        std::unique_lock<std::mutex> lock(devicesDataMutex);
        devicesData.erase(device);
    }
    LOG_DRIVER_API("Destroy logical device <%p>", static_cast<const void*>(convertDevice(device)));
    vkDestroyDevice(drv::resolve_ptr<VkDevice>(device), nullptr);
    return true;
}

drv::QueuePtr DrvVulkan::get_queue(drv::LogicalDevicePtr device, drv::QueueFamilyPtr family,
                                   unsigned int ind) {
    VkQueue queue;
    vkGetDeviceQueue(drv::resolve_ptr<VkDevice>(device), convertFamilyToVk(family), ind, &queue);
    drv::QueuePtr ret = drv::store_ptr<drv::QueuePtr>(queue);
    {
        std::unique_lock<std::mutex> lock(devicesDataMutex);
        auto itr = devicesData.find(device);
        drv::drv_assert(itr != devicesData.end());
        itr->second.queueToFamily[drv::store_ptr<drv::QueuePtr>(queue)] = family;
        itr->second.queueMutexes[drv::store_ptr<drv::QueuePtr>(queue)];

        uint32_t count;
        vkGetPhysicalDeviceQueueFamilyProperties(convertPhysicalDevice(itr->second.physicalDevice),
                                                 &count, nullptr);
        StackMemory::MemoryHandle<VkQueueFamilyProperties> vkQueueFamilies(count, TEMPMEM);
        vkGetPhysicalDeviceQueueFamilyProperties(convertPhysicalDevice(itr->second.physicalDevice),
                                                 &count, vkQueueFamilies);
        uint32_t familyInd = convertFamilyToVk(family);
        if (vkQueueFamilies[familyInd].timestampValidBits != 0) {
            itr->second.timestampBits[ret] = 0;
            for (uint32_t i = 0; i < vkQueueFamilies[familyInd].timestampValidBits; ++i)
                itr->second.timestampBits[ret] = (itr->second.timestampBits[ret] << 1) | 1;
            if (itr->second.cmdPools.find(family) == itr->second.cmdPools.end()) {
                drv::CommandPoolCreateInfo poolInfo(false, false);
                auto& pool = itr->second.cmdPools[family] =
                  create_command_pool(device, family, &poolInfo);
                auto& cmdBuffer = itr->second.calibrationCmdBuffers[family];

                drv::CommandBufferCreateInfo cmdBufferInfo;
                cmdBufferInfo.flags = drv::CommandBufferCreateInfo::ONE_TIME_SUBMIT_BIT;
                cmdBufferInfo.type = drv::CommandBufferType::PRIMARY;
                cmdBuffer = create_command_buffer(device, pool, &cmdBufferInfo);

                VkCommandBufferBeginInfo info;
                info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                info.pNext = nullptr;
                info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                info.pInheritanceInfo = nullptr;
                VkResult result = vkBeginCommandBuffer(convertCommandBuffer(cmdBuffer), &info);
                drv::drv_assert(result == VK_SUCCESS, "Could not begin recording command buffer");

                vkCmdWriteTimestamp(convertCommandBuffer(cmdBuffer),
                                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                    convertTimestampQueryPool(itr->second.queryPool), 0);

                result = vkEndCommandBuffer(convertCommandBuffer(cmdBuffer));
                drv::drv_assert(result == VK_SUCCESS, "Could not finish recording command buffer");
            }
        }
    }
    return ret;
}

bool DrvVulkan::queue_wait_idle(drv::LogicalDevicePtr, drv::QueuePtr queue) {
    VkResult result = vkQueueWaitIdle(convertQueue(queue));
    return result == VK_SUCCESS;
}

bool DrvVulkan::device_wait_idle(drv::LogicalDevicePtr device) {
    VkResult result = vkDeviceWaitIdle(convertDevice(device));
    return result == VK_SUCCESS;
}

drv::DriverSupport DrvVulkan::get_support(drv::LogicalDevicePtr device) {
    UNUSED(device);
    drv::DriverSupport ret;
    ret.conditionalRendering = false;
    ret.tessellation = false;
    ret.geometry = false;
    ret.taskShaders = false;
    ret.transformFeedback = false;
    ret.shadingRate = false;
    ret.meshShaders = false;
    ret.fragmentDensityMap = false;
    return ret;
}

uint64_t DrvVulkan::sync_gpu_clock(drv::InstancePtr /*_instance*/,
                                   drv::PhysicalDevicePtr /*physicalDevice*/,
                                   drv::LogicalDevicePtr device) {
    // drv_vulkan::Instance* instance = drv::resolve_ptr<drv_vulkan::Instance*>(_instance);
    std::unique_lock<std::mutex> lock(devicesDataMutex);
    auto deviceItr = devicesData.find(device);
    drv::drv_assert(deviceItr != devicesData.end());
    // uint64_t maxDeviation;
    // VkCalibratedTimestampInfoEXT timeStampInfo;
    // timeStampInfo.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
    // timeStampInfo.pNext = nullptr;
    // timeStampInfo.timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
    // Clock::time_point prevHostTime = deviceItr->second.lastSyncTimeHost;
    // uint64_t prevDeviceTicks = deviceItr->second.lastSyncTimeDeviceTicks;
    // VkResult result = instance->vkGetCalibratedTimestampsEXT(
    //   convertDevice(device), 1, &timeStampInfo, &deviceItr->second.lastSyncTimeDeviceTicks,
    //   &maxDeviation);
    // deviceItr->second.lastSyncTimeHost = Clock::now();
    // drv::drv_assert(result == VK_SUCCESS, "Could not calibrate timestamps");

    // if (prevDeviceTicks != 0) {
    //     int64_t deviceDiffNs =
    //       int64_t(double(deviceItr->second.lastSyncTimeDeviceTicks - prevDeviceTicks)
    //               * double(timestampPeriod));
    //     int64_t hostDiffNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
    //                            deviceItr->second.lastSyncTimeHost - prevHostTime)
    //                            .count();
    //     LOG_DRIVER_API(
    //       "Clock calibrated. Max deviation: %llu ns, drift = %lld ns (h: %lld, d: %lld)",
    //       maxDeviation, deviceDiffNs - hostDiffNs, hostDiffNs, deviceDiffNs);
    // }
    // else
    //     LOG_DRIVER_API("Clock calibrated. Max deviation: %llu ns", maxDeviation);
    StackMemory::MemoryHandle<std::unique_lock<std::mutex>> locks(
      deviceItr->second.queueMutexes.size(), TEMPMEM);
    {
        uint32_t ind = 0;
        for (auto& itr : deviceItr->second.queueMutexes)
            locks[ind++] = std::unique_lock<std::mutex>(itr.second);
    }
    uint64_t ret = 0;

    device_wait_idle(device);

    for (auto& [queue, bits] : deviceItr->second.timestampBits) {
        reset_timestamp_queries(device, deviceItr->second.queryPool, 0, 1);
        auto familyItr = deviceItr->second.queueToFamily.find(queue);
        drv::drv_assert(familyItr != deviceItr->second.queueToFamily.end(), "Family not found");
        auto cmdBufferItr = deviceItr->second.calibrationCmdBuffers.find(familyItr->second);
        drv::drv_assert(cmdBufferItr != deviceItr->second.calibrationCmdBuffers.end(),
                        "Command buffer not found");

        VkCommandBuffer vkCmdBuffer = convertCommandBuffer(cmdBufferItr->second);
        VkSubmitInfo submitInfo;
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = nullptr;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &vkCmdBuffer;
        submitInfo.signalSemaphoreCount = 0;
        submitInfo.pSignalSemaphores = nullptr;
        submitInfo.waitSemaphoreCount = 0;
        submitInfo.pWaitSemaphores = nullptr;
        submitInfo.pWaitDstStageMask = nullptr;
        LogicalDeviceData::SyncTimeData syncData;
        vkQueueSubmit(convertQueue(queue), 1, &submitInfo, convertFence(deviceItr->second.fence));
        syncData.lastSyncTimeHost = drv::Clock::now();
        // int64_t submissionNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        //                          submissionTime - deviceItr->second.lastSyncTime)
        //                          .count();
        wait_for_fence(device, 1, &deviceItr->second.fence, true, 0);
        reset_fences(device, 1, &deviceItr->second.fence);
        // uint64_t executionTicks;
        drv::drv_assert(get_timestamp_query_pool_results(device, deviceItr->second.queryPool, 0, 1,
                                                         &syncData.lastSyncTimeDeviceTicks),
                        "Timestamps are not ready yet");
        syncData.lastSyncTimeDeviceTicks &= bits;
        // int64_t executionNs = executionTicks * gpuClockNsPerTick;  // timestampPeriod
        if (auto syncItr = deviceItr->second.queueTimeline.find(queue);
            syncItr != deviceItr->second.queueTimeline.end()) {
            // int64_t deviceDiffNs = int64_t(
            //   double(syncData.lastSyncTimeDeviceTicks - syncItr->second.lastSyncTimeDeviceTicks)
            //   * double(deviceItr->second.timestampPeriod));
            // int64_t hostDiffNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            //                        syncData.lastSyncTimeHost - syncItr->second.lastSyncTimeHost)
            //                        .count();
            // uint64_t diff = uint64_t(deviceDiffNs >= hostDiffNs ? deviceDiffNs - hostDiffNs
            //                                                     : hostDiffNs - deviceDiffNs);
            int64_t hostPredictionDifference =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                deviceItr->second.decode_timestamp(bits, syncItr->second,
                                                   syncData.lastSyncTimeDeviceTicks)
                - syncData.lastSyncTimeHost)
                .count();
            uint64_t diff = uint64_t(hostPredictionDifference >= 0 ? hostPredictionDifference
                                                                   : -hostPredictionDifference);
            if (ret < diff)
                ret = diff;
            // doesn't work unfortunately. It can increase prediction error by a lot
            // if (deviceDiffNs > 0)
            //     syncData.driftHnsPerDns = double(hostDiffNs - deviceDiffNs) / double(deviceDiffNs);
            // LOG_DRIVER_API(
            //   "Clock re-calibrated for <%p>: Drift = %lldns, %lfHns/Dms, prediction error: %lldns",
            //   drv::get_ptr(queue), deviceDiffNs - hostDiffNs, syncData.driftHnsPerDns * 1000000.0,
            //   hostPredictionDifference);
        }
        deviceItr->second.queueTimeline[queue] = syncData;
    }
    return ret;
}
