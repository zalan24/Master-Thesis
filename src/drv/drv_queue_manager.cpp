#include "drv_queue_manager.h"

#include <algorithm>

#include <drverror.h>

#include "drv.h"

using namespace drv;

#define QM_SYNC std::unique_lock<std::mutex> lock(mutex)

QueueManager::QueueManager(LogicalDevicePtr _device, CommandLaneManager* commandLaneMgr)
  : device(_device) {
    for (const auto& itr : commandLaneMgr->getLanes()) {
        LaneInfo laneInfo;
        for (const auto& qItr : itr.second.queues) {
            QueueInfo qInfo;
            qInfo.info = qItr.second;
            qInfo.queue = drv::get_queue(device, qItr.second.familyPtr, qItr.second.queueIndex);
            drv_assert(!is_null_ptr(qInfo.queue), "Could not get queue");
            laneInfo.queues[qItr.first] = std::move(qInfo);
        }
        lanes[itr.first] = std::move(laneInfo);
    }
}

QueueManager::~QueueManager() {
    QM_SYNC;
}

QueueManager::Queue QueueManager::getQueue(const AcquireInfo& info) const {
    QM_SYNC;
    auto laneItr = lanes.find(info.commandLane);
    if (laneItr == lanes.end())
        throw std::runtime_error("Could not find command lane: " + info.commandLane);
    auto queueItr = laneItr->second.queues.find(info.commandQueue);
    if (queueItr == laneItr->second.queues.end())
        throw std::runtime_error("Could not find command queue: " + info.commandQueue);
    Queue ret;
    ret.queue = queueItr->second.queue;
    ret.info = queueItr->second.info;
    return ret;
}
