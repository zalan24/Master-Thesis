#pragma once

#include <iostream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

// This is collected during every execution and used in the next one (in release too)

#include <serializable.h>

#include "stattypes.h"

struct StatsCache final : public IAutoSerializable<StatsCache>
{
    REFLECTABLE
    (
        (std::unordered_map<std::string, std::unique_ptr<StatsCache>>) subnodes,
        (std::map<drv::ImageId, ImageStateStat>) cmdBufferImageStates,
        (std::vector<PipelineStagesStat>) renderpassAttachmentPostUsage,
        (std::vector<SimpleSubresStateStat>) renderpassExternalAttachmentInputs
    )

    mutable std::shared_mutex mutex;

 protected:
    bool needTimeStamp() const override { return true; }
};
