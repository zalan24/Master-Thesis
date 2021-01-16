#include "drvlane.h"

#include <algorithm>
#include <functional>

#include <drverror.h>

using namespace drv;

static bool assign_lanes(const std::function<void(size_t, size_t)>& assign,
                         std::vector<drv::CommandTypeMask>& requirements,
                         const std::vector<drv::QueueFamily>& families,
                         std::vector<size_t>& remaining) noexcept {
    bool hasRequirements = false;
    // exact match
    for (size_t laneId = 0; laneId < requirements.size(); ++laneId) {
        if (!requirements[laneId])
            continue;
        hasRequirements = true;
        for (size_t familyId = 0; familyId < families.size(); ++familyId) {
            if (remaining[familyId] == 0)
                continue;
            if (families[familyId].commandTypeMask == requirements[laneId]) {
                drv::CommandTypeMask r = requirements[laneId];
                remaining[familyId]--;
                if (assign_lanes(assign, requirements, families, remaining)) {
                    assign(familyId, laneId);
                    return true;
                }
                remaining[familyId]++;
                requirements[laneId] = r;
            }
        }
    }
    if (!hasRequirements)
        return true;
    // no wasting command types
    for (size_t laneId = 0; laneId < requirements.size(); ++laneId) {
        if (!requirements[laneId])
            continue;
        for (size_t familyId = 0; familyId < families.size(); ++familyId) {
            if (remaining[familyId] == 0)
                continue;
            if (families[familyId].commandTypeMask == requirements[laneId])
                continue;  // already processed
            if ((families[familyId].commandTypeMask & requirements[laneId])
                == families[familyId].commandTypeMask) {
                drv::CommandTypeMask r = requirements[laneId];
                remaining[familyId]--;
                if (assign_lanes(assign, requirements, families, remaining)) {
                    assign(familyId, laneId);
                    return true;
                }
                remaining[familyId]++;
                requirements[laneId] = r;
            }
        }
    }
    // anything else
    for (size_t laneId = 0; laneId < requirements.size(); ++laneId) {
        if (!requirements[laneId])
            continue;
        for (size_t familyId = 0; familyId < families.size(); ++familyId) {
            if (remaining[familyId] == 0)
                continue;
            if (families[familyId].commandTypeMask == requirements[laneId])
                continue;  // already processed
            if ((families[familyId].commandTypeMask & requirements[laneId])
                == requirements[laneId]) {
                drv::CommandTypeMask r = requirements[laneId];
                remaining[familyId]--;
                if (assign_lanes(assign, requirements, families, remaining)) {
                    assign(familyId, laneId);
                    return true;
                }
                remaining[familyId]++;
                requirements[laneId] = r;
            }
        }
    }
    return false;
}

CommandLaneManager::CommandLaneManager(PhysicalDevicePtr physicalDevice,
                                       const std::vector<CommandLaneInfo>& laneInfos) {
    unsigned int familyCount = 0;
    drv::get_physical_device_queue_families(physicalDevice, &familyCount, nullptr);
    std::vector<drv::QueueFamily> queueFamilies(familyCount);
    drv::get_physical_device_queue_families(physicalDevice, &familyCount, queueFamilies.data());

    std::vector<bool> chosen(laneInfos.size() * familyCount, false);
    std::vector<size_t> remaining(familyCount, 0);
    std::vector<std::vector<size_t>> familyAssignments(familyCount);
    std::vector<drv::CommandTypeMask> requirements(laneInfos.size(), 0);
    for (size_t i = 0; i < familyCount; ++i)
        remaining[i] = queueFamilies[i].queueCount;
    for (size_t i = 0; i < laneInfos.size(); ++i)
        requirements[i] = laneInfos[i].commandTypes;

    for (const auto& lane : laneInfos)
        lanes[lane.name].priority = lane.priority;

    auto assign = [&](size_t family, size_t laneId) {
        if (chosen[family * laneInfos.size() + laneId])
            return;
        drv::drv_assert(remaining[family] > 0);
        chosen[family * laneInfos.size() + laneId] = true;
        remaining[family]--;
        familyAssignments[family].push_back(laneId);
        requirements[laneId] ^= requirements[laneId] & queueFamilies[family].commandTypeMask;
    };

    // assign queue families to every lane, if there are enough queues in them
    bool requirementsDecreased = true;
    while (requirementsDecreased) {
        for (size_t i = 0; i < familyCount; ++i) {
            requirementsDecreased = false;
            size_t needCount = 0;
            for (size_t j = 0; j < laneInfos.size(); ++j)
                if (requirements[j] & queueFamilies[i].commandTypeMask)
                    needCount++;
            if (needCount <= queueFamilies[i].queueCount) {
                for (size_t j = 0; j < laneInfos.size(); ++j) {
                    if (requirements[j] & queueFamilies[i].commandTypeMask) {
                        assign(i, j);
                        requirementsDecreased = true;
                    }
                }
            }
        }
    }

    // minimun required assignments. The result might not use all queues available
    drv::drv_assert(assign_lanes(assign, requirements, queueFamilies, remaining),
                    "Could not satisfy lane requirements");

    // Generate lanes based on assignments
    for (size_t familyId = 0; familyId < familyCount; ++familyId) {
        if (familyAssignments[familyId].size() == 0)
            continue;
        size_t maxUsedQueueCount = 0;
        for (const size_t laneId : familyAssignments[familyId]) {
            if (laneInfos[laneId].maxQueuesPerFamily == 0) {
                maxUsedQueueCount = queueFamilies[familyId].queueCount;
                break;
            }
            else {
                maxUsedQueueCount += laneInfos[laneId].maxQueuesPerFamily;
                if (maxUsedQueueCount > queueFamilies[familyId].queueCount) {
                    maxUsedQueueCount = queueFamilies[familyId].queueCount;
                    break;
                }
            }
        }
        std::vector<float> weights(laneInfos.size());
        std::transform(laneInfos.begin(), laneInfos.end(), weights.begin(),
                       [](const CommandLaneInfo& lane) { return lane.queueCountWeight; });
        float queueCount = static_cast<float>(maxUsedQueueCount);
        bool hasMaxed = true;
        float totalWeight = 0;
        while (hasMaxed) {
            hasMaxed = false;
            totalWeight = 0;
            for (const size_t laneId : familyAssignments[familyId])
                totalWeight += weights[laneId];
            drv::drv_assert(
              totalWeight > 0,
              "Invalid lane config: Queue cant weight must be positive for all lanes.");
            for (const size_t laneId : familyAssignments[familyId]) {
                if (laneInfos[laneId].maxQueuesPerFamily == 0)
                    continue;
                size_t max = laneInfos[laneId].maxQueuesPerFamily;
                // This can be optimized, if the biggest difference is corrected first in each step
                if (static_cast<size_t>(queueCount * weights[laneId] / totalWeight) > max) {
                    weights[laneId] = max * (totalWeight - weights[laneId]) / (queueCount - max);
                    hasMaxed = true;
                }
            }
        }
        unsigned int queueIndex = 0;
        for (const size_t laneId : familyAssignments[familyId]) {
            size_t numQueues =
              std::max(1u, static_cast<unsigned int>(queueCount * weights[laneId] / totalWeight));
            for (size_t i = 0; i < numQueues; ++i)
                lanes[laneInfos[laneId].name].queues.push_back(
                  CommandLane::LaneQueue{queueFamilies[familyId].handle, queueIndex++});
        }
    }
}

const CommandLane* CommandLaneManager::getLane(const std::string& name) const {
    auto itr = lanes.find(name);
    if (itr == lanes.end())
        return nullptr;
    else
        return &itr->second;
}

std::unordered_map<QueueFamilyPtr, std::vector<float>> CommandLaneManager::getQueuePriorityInfo()
  const {
    std::unordered_map<QueueFamilyPtr, std::vector<float>> ret;
    for (const auto& lane : lanes) {
        for (const CommandLane::LaneQueue& queue : lane.second.queues) {
            if (ret[queue.familyPtr].size() <= queue.queueIndex)
                ret[queue.familyPtr].resize(queue.queueIndex + 1);
            ret[queue.familyPtr][queue.queueIndex] = lane.second.priority;
        }
    }
    return ret;
}
