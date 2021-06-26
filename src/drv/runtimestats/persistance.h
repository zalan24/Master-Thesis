#pragma once

// This is heavy stuff, meant to be collected prior to usage, not in release

#include <vector>
#include <shared_mutex>

#include <serializable.h>

struct RenderpassCorrectionData final : public IAutoSerializable<RenderpassCorrectionData>
{
    REFLECTABLE
    (
        (std::string) renderpass,
        (std::string) submission,
        (uint32_t) attachmentId
    )
    bool operator<(const RenderpassCorrectionData& rhs) const {
        if (attachmentId != rhs.attachmentId)
            return attachmentId < rhs.attachmentId;
        if (renderpass != rhs.renderpass)
            return renderpass < rhs.renderpass;
        return submission < rhs.submission;
    }
};

struct SingleExecutionData final : public IAutoSerializable<SingleExecutionData>
{
    REFLECTABLE
    (
        (std::string) startTime,
        (std::string) endTime,
        (uint32_t) frameCount,
        (uint32_t) sampleInputCount,
        (uint32_t) submissionCount,
        (uint32_t) allowedSubmissionCorrections,
        (uint32_t) numCpuAutoSync,
        (std::unordered_map<std::string, uint32_t>) submissionCorrections,
        (std::map<RenderpassCorrectionData, uint32_t>) attachmentCorrections,
        (std::map<std::string, std::vector<std::string>>) invalidSharedResourceUsage
    )

    void start();
    void stop();
};

struct PersistanceNodeData final : public IAutoSerializable<PersistanceNodeData>
{
    REFLECTABLE
    (
        (std::unordered_map<std::string, std::unique_ptr<PersistanceNodeData>>) subnodes,
        (SingleExecutionData) lastExecution
    )

    PersistanceNodeData() = default;
    // PersistanceNodeData(const PersistanceNodeData& rhs)
    //   : subnodes(rhs.subnodes), lastExecution(rhs.lastExecution) {}
    // PersistanceNodeData& operator=(const PersistanceNodeData& rhs) {
    //     subnodes = rhs.subnodes;
    //     lastExecution = rhs.lastExecution;
    //     return *this;
    // }
    PersistanceNodeData(PersistanceNodeData&& rhs)
      : subnodes(std::move(rhs.subnodes)), lastExecution(std::move(rhs.lastExecution)) {}
    PersistanceNodeData& operator=(PersistanceNodeData&& rhs) {
        subnodes = std::move(rhs.subnodes);
        lastExecution = std::move(rhs.lastExecution);
        return *this;
    }

    mutable std::shared_mutex mutex;

 protected:
    bool needTimeStamp() const override { return true; }
};
