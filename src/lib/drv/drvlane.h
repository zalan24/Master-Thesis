#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <drv.h>
#include <drv_wrappers.h>
#include <exclusive.h>

namespace drv
{
class CommandLaneManager
  : public NoCopy
  , private Exclusive
{
 public:
    struct Queue
    {
        QueueFamilyPtr familyPtr;
        unsigned int queueIndex;
        float priority;
    };

    struct CommandLaneInfo
    {
        std::string name;
        struct CommandQueueInfo
        {
            std::string name;
            float priority;
            CommandTypeMask commandTypes;
            CommandTypeMask preferenceMask;  // doesn't need to contain 'commandTypes'
        };
        std::vector<CommandQueueInfo> queues;
    };

    CommandLaneManager(PhysicalDevicePtr physicalDevice,
                       const std::vector<CommandLaneInfo>& laneInfos);

    const Queue* getQueue(const std::string& laneName, const std::string& queueName) const;

    std::unordered_map<QueueFamilyPtr, std::vector<float>> getQueuePriorityInfo() const;

 private:
    struct CommandLane
    {
        // The same queue can be assigned to several names
        std::unordered_map<std::string, Queue> queues;
    };

    std::unordered_map<std::string, CommandLane> lanes;
};

}  // namespace drv
