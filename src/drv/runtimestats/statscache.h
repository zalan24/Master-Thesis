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
        (std::vector<ImageUsageStat>) renderpassAttachmentPostUsage,
        (std::vector<SimpleSubresStateStat>) renderpassExternalAttachmentInputs,
        (SemaphoreSyncData) semaphore
    )

    StatsCache() = default;
    // StatsCache(const StatsCache& rhs)
    //   : subnodes(rhs.subnodes),
    //     cmdBufferImageStates(rhs.cmdBufferImageStates),
    //     renderpassAttachmentPostUsage(rhs.renderpassAttachmentPostUsage),
    //     renderpassExternalAttachmentInputs(rhs.renderpassExternalAttachmentInputs) {}
    StatsCache(StatsCache&& rhs)
      : subnodes(std::move(rhs.subnodes)),
        cmdBufferImageStates(std::move(rhs.cmdBufferImageStates)),
        renderpassAttachmentPostUsage(std::move(rhs.renderpassAttachmentPostUsage)),
        renderpassExternalAttachmentInputs(std::move(rhs.renderpassExternalAttachmentInputs)) {}
    // StatsCache& operator=(const StatsCache& rhs) {
    //     subnodes = rhs.subnodes;
    //     cmdBufferImageStates = rhs.cmdBufferImageStates;
    //     renderpassAttachmentPostUsage = rhs.renderpassAttachmentPostUsage;
    //     renderpassExternalAttachmentInputs = rhs.renderpassExternalAttachmentInputs;
    //     return *this;
    // }
    StatsCache& operator=(StatsCache&& rhs) {
        subnodes = std::move(rhs.subnodes);
        cmdBufferImageStates = std::move(rhs.cmdBufferImageStates);
        renderpassAttachmentPostUsage = std::move(rhs.renderpassAttachmentPostUsage);
        renderpassExternalAttachmentInputs = std::move(rhs.renderpassExternalAttachmentInputs);
        return *this;
    }

    mutable std::shared_mutex mutex;

 protected:
    bool needTimeStamp() const override { return true; }
};
