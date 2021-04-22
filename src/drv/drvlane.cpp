#include "drvlane.h"

#include <algorithm>
#include <functional>

#include <drverror.h>

using namespace drv;

static unsigned int bit_count(size_t s) {
    unsigned int ret = 0;
    while (s) {
        while ((s & 0b1) != 1)
            s >>= 1;
        ret++;
        s ^= 0b1;
    }
    return ret;
}

CommandLaneManager::CommandLaneManager(PhysicalDevicePtr physicalDevice, IWindow* window,
                                       const std::vector<CommandLaneInfo>& laneInfos) {
    unsigned int familyCount = 0;
    drv::get_physical_device_queue_families(physicalDevice, &familyCount, nullptr);
    std::vector<drv::QueueFamily> queueFamilies(familyCount);
    drv::get_physical_device_queue_families(physicalDevice, &familyCount, queueFamilies.data());

    std::unordered_map<QueueFamilyPtr, unsigned int> remaining(familyCount);
    for (size_t i = 0; i < familyCount; ++i)
        remaining[queueFamilies[i].handle] = queueFamilies[i].queueCount;

    for (const CommandLaneInfo& laneInfo : laneInfos) {
        CommandLane lane;
        std::set<QueueFamilyPtr> usedFamilies;

        for (const CommandLaneInfo::CommandQueueInfo& queueInfo : laneInfo.queues) {
            Queue q;
            q.priority = queueInfo.priority;
            q.familyPtr = IGNORE_FAMILY;
            q.commandTypes = queueInfo.commandTypes;
            unsigned int bestMatch = 0;
            for (size_t i = 0; i < familyCount; ++i) {
                if ((queueFamilies[i].commandTypeMask & queueInfo.commandTypes)
                    != queueInfo.commandTypes)
                    continue;
                if (queueInfo.requirePresent
                    && !drv::can_present(physicalDevice, window, queueFamilies[i].handle))
                    continue;
                if (remaining[queueFamilies[i].handle] == 0)
                    continue;
                unsigned int value =
                  bit_count(queueFamilies[i].commandTypeMask ^ queueInfo.preferenceMask);
                if (q.familyPtr == IGNORE_FAMILY || value < bestMatch) {
                    q.familyPtr = queueFamilies[i].handle;
                    bestMatch = value;
                    usedFamilies.insert(q.familyPtr);
                    q.queueIndex = queueFamilies[i].queueCount - remaining[queueFamilies[i].handle];
                }
            }
            if (q.familyPtr == IGNORE_FAMILY)
                throw std::runtime_error("Could not find a queue for lane " + laneInfo.name
                                         + " / queue " + queueInfo.name);
            lane.queues[queueInfo.name] = std::move(q);
        }

        for (const QueueFamilyPtr& ptr : usedFamilies)
            remaining[ptr]--;
        lanes[laneInfo.name] = std::move(lane);
    }

    // Could be optimized
    auto countQueuesOn = [&](drv::QueueFamilyPtr family, unsigned int queueIndex) {
        unsigned int count = 0;
        for (auto& lane : lanes)
            for (auto& queue : lane.second.queues)
                if (queue.second.familyPtr == family && queue.second.queueIndex == queueIndex)
                    count++;
        return count;
    };

    for (const CommandLaneInfo& laneInfo : laneInfos) {
        auto laneItr = lanes.find(laneInfo.name);
        drv::drv_assert(laneItr != lanes.end());
        for (const CommandLaneInfo::CommandQueueInfo& queueInfo : laneInfo.queues) {
            if (!queueInfo.preferDedicated)
                continue;
            auto queueItr = laneItr->second.queues.find(queueInfo.name);
            drv::drv_assert(queueItr != laneItr->second.queues.end());
            // Queue &q = lanes[laneInfo.name].queues[queueInfo.name]
            if (remaining[queueItr->second.familyPtr] == 0)
                continue;
            if (countQueuesOn(queueItr->second.familyPtr, queueItr->second.queueIndex) <= 1)
                continue;
            auto familyItr = std::find_if(queueFamilies.begin(), queueFamilies.end(),
                                          [queueItr](const QueueFamily& family) {
                                              return family.handle == queueItr->second.familyPtr;
                                          });
            drv::drv_assert(familyItr != queueFamilies.end());
            queueItr->second.queueIndex =
              familyItr->queueCount - remaining[queueItr->second.familyPtr];
            remaining[queueItr->second.familyPtr]--;
        }
    }

    std::unordered_map<QueueFamilyPtr, std::unordered_map<size_t, float>> priorities;

    for (auto& lane : lanes)
        for (auto& queue : lane.second.queues)
            priorities[queue.second.familyPtr][queue.second.queueIndex] = std::max(
              priorities[queue.second.familyPtr][queue.second.queueIndex], queue.second.priority);

    for (auto& lane : lanes)
        for (auto& queue : lane.second.queues)
            queue.second.priority = priorities[queue.second.familyPtr][queue.second.queueIndex];
}

const CommandLaneManager::Queue* CommandLaneManager::getQueue(const std::string& laneName,
                                                              const std::string& queueName) const {
    auto itr = lanes.find(laneName);
    if (itr == lanes.end())
        return nullptr;
    auto queueItr = itr->second.queues.find(queueName);
    if (queueItr == itr->second.queues.end())
        return nullptr;
    return &queueItr->second;
}

std::unordered_map<QueueFamilyPtr, std::vector<float>> CommandLaneManager::getQueuePriorityInfo()
  const {
    std::unordered_map<QueueFamilyPtr, std::vector<float>> ret;
    for (const auto& lane : lanes) {
        for (const auto& queue : lane.second.queues) {
            if (ret[queue.second.familyPtr].size() <= queue.second.queueIndex)
                ret[queue.second.familyPtr].resize(queue.second.queueIndex + 1, 0);
            ret[queue.second.familyPtr][queue.second.queueIndex] =
              std::max(queue.second.priority, ret[queue.second.familyPtr][queue.second.queueIndex]);
        }
    }
    return ret;
}
