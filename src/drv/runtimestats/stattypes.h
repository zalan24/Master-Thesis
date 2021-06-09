#pragma once

#include <serializable.h>
#include <fixedarray.hpp>

#include <drvtypes.h>
#include <drvresourceptrs.hpp>
#include <drvtracking.hpp>

struct PipelineStagesStat final : public IAutoSerializable<PipelineStagesStat>
{
    static constexpr float EXP_AVG = 0.01f;
    static constexpr float THRESHOLD = 0.01f;

    enum ApproximationMode
    {
        TEND_TO_FALSE,
        TEND_TO_TRUE
    };

    REFLECTABLE
    (
        (std::array<float, drv::PipelineStages::get_total_stage_count()>) stages
    )

    void set(const drv::PipelineStages& stages);
    void append(const drv::PipelineStages& stages);
    drv::PipelineStages get(ApproximationMode mode) const;

    PipelineStagesStat();
};

struct MemoryAccessStat final : public IAutoSerializable<MemoryAccessStat>
{
    static constexpr float EXP_AVG = 0.01f;
    static constexpr float THRESHOLD = 0.01f;

    enum ApproximationMode
    {
        TEND_TO_FALSE,
        TEND_TO_TRUE
    };

    REFLECTABLE
    (
        (std::array<float, drv::MemoryBarrier::get_total_access_count()>) mask
    )

    void set(const drv::MemoryBarrier::AccessFlagBitType& mask);
    void append(const drv::MemoryBarrier::AccessFlagBitType& stages);
    drv::MemoryBarrier::AccessFlagBitType get(ApproximationMode mode) const;

    MemoryAccessStat();
};

struct ImageLayoutStat final : public IAutoSerializable<ImageLayoutStat>
{
    static constexpr float EXP_AVG = 0.01f;
    static constexpr float THRESHOLD = 0.01f;

    REFLECTABLE
    (
        (std::array<float, drv::get_image_layout_count()>) layouts
    )

    void set(const drv::ImageLayout layout);
    void append(const drv::ImageLayout layout);
    drv::ImageLayout get() const;

    ImageLayoutStat();
};

struct SimpleSubresStateStat final : public IAutoSerializable<SimpleSubresStateStat>
{
    REFLECTABLE
    (
        (PipelineStagesStat) usableStages,
        (PipelineStagesStat) writes,
        (PipelineStagesStat) reads,
        (MemoryAccessStat) dirtyMask,
        (MemoryAccessStat) visible
    )

    void set(const drv::PerSubresourceRangeTrackData& data);
    void append(const drv::PerSubresourceRangeTrackData& data);
    // Try to assume a clean state by default. It's not possible for usableStages and visible mask
    void get(drv::PerSubresourceRangeTrackData& data) const;
};

struct SubresStateStat final : public IAutoSerializable<SubresStateStat>
{
    static constexpr float EXP_AVG = 0.01f;

    REFLECTABLE
    (
        (float) noFamily,
        (FixedArray<float, 8>) queueFamilies,
        (SimpleSubresStateStat) simpleStats
    )

    void set(const drv::PerSubresourceRangeTrackData& data);
    void append(const drv::PerSubresourceRangeTrackData& data);
    void get(drv::PerSubresourceRangeTrackData& data) const;

    SubresStateStat();

    private:
    void resizeFamilies(size_t size);
};

struct ImageSubresStateStat final : public IAutoSerializable<ImageSubresStateStat>
{
    REFLECTABLE
    (
        (SubresStateStat) subres,
        (ImageLayoutStat) layout
    )

    void set(const drv::ImageSubresourceTrackData& data);
    void append(const drv::ImageSubresourceTrackData& data);
    void get(drv::ImageSubresourceTrackData& data) const;
};

struct ImageStateStat final : public ISerializable
{
    drv::ImagePerSubresourceData<ImageSubresStateStat, 1> subresources;

    bool writeBin(std::ostream& out) const override;
    bool readBin(std::istream& in) override;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;

    bool isCompatible(const drv::TextureInfo& info) const;
    void init(const drv::TextureInfo& info);
};
