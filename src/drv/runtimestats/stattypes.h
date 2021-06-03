#pragma once

#include <serializable.h>
#include <fixedarray.hpp>

#include <drvtypes.h>
#include <drvresourceptrs.hpp>
#include <drvtracking.hpp>

struct SimpleSubresStateStat final : public IAutoSerializable<SimpleSubresStateStat>
{
    REFLECTABLE
    (
        (std::array<float, drv::PipelineStages::get_total_stage_count()>) usableStages,
        (std::array<float, drv::PipelineStages::get_total_stage_count()>) writes,
        (std::array<float, drv::PipelineStages::get_total_stage_count()>) reads,
        (std::array<float, drv::MemoryBarrier::get_total_access_count>()) dirtyMask,
        (std::array<float, drv::MemoryBarrier::get_total_access_count>()) visible
    )

    void set(const drv::PerSubresourceRangeTrackData& data);
    void append(const drv::PerSubresourceRangeTrackData& data);
    void get(drv::PerSubresourceRangeTrackData& data) const;
};

struct SubresStateStat final : public IAutoSerializable<SubresStateStat>
{
    REFLECTABLE
    (
        (float) noFamily,
        (FixedArray<float>) queueFamilies,
        (SimpleSubresStateStat) simpleStats
    )

    void set(const drv::PerSubresourceRangeTrackData& data);
    void append(const drv::PerSubresourceRangeTrackData& data);
    void get(drv::PerSubresourceRangeTrackData& data) const;
};

struct ImageSubresStateStat final : public IAutoSerializable<ImageSubresStateStat>
{
    REFLECTABLE
    (
        (SubresStateStat) subres,
        (std::array<float, drv::get_image_layout_count()>) layout
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

    bool isCompatible(drv::ImagePtr image) const;
};
