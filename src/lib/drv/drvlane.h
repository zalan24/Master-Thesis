#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <drv.h>
#include <drv_wrappers.h>
#include <exclusive.h>

namespace drv
{
struct CommandLane
{
    struct LaneQueue
    {
        QueueFamilyPtr familyPtr;
        unsigned int queueIndex;
    };
    float priority;
    std::vector<LaneQueue> queues;
};

class CommandLaneManager
  : public NoCopy
  , private Exclusive
{
 public:
    struct CommandLaneInfo
    {
        std::string name;
        float priority;
        float queueCountWeight;
        size_t maxQueuesPerFamily;  // 0 for no limit
        CommandTypeMask commandTypes;
    };

    CommandLaneManager(PhysicalDevicePtr physicalDevice,
                       const std::vector<CommandLaneInfo>& laneInfos);

    const CommandLane* getLane(const std::string& name) const;

    std::unordered_map<QueueFamilyPtr, std::vector<float>> getQueuePriorityInfo() const;

 private:
    std::unordered_map<std::string, CommandLane> lanes;
};

}  // namespace drv
