#pragma once

#include <serializable.h>
#include <fixedarray.hpp>

#include <drvtypes.h>
#include <drvresourceptrs.hpp>
#include <drvtracking.hpp>
#include <drvimage_types.h>

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

struct ImageUsageStat final : public IAutoSerializable<ImageUsageStat>
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
        (std::array<float, drv::get_image_usage_count()>) usages
    )

    void set(const drv::ImageResourceUsageFlag &usages);
    void append(const drv::ImageResourceUsageFlag &usages);
    drv::ImageResourceUsageFlag get(ApproximationMode mode) const;

    ImageUsageStat();
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
    void get(drv::PerSubresourceRangeTrackData& data, bool tendTo) const;
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
    void get(drv::PerSubresourceRangeTrackData& data, bool tendTo) const;

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
    void get(drv::ImageSubresourceTrackData& data, bool tendTo) const;
};

template <typename T>
struct ImageSubresourcesData final : public ISerializable
{
    drv::ImagePerSubresourceData<T, 1> subresources;

    bool writeBin(std::ostream& out) const override {
        if (!serializeBin(out, subresources.layerCount))
            return false;
        if (!serializeBin(out, subresources.mipCount))
            return false;
        if (!serializeBin(out, subresources.aspects))
            return false;
        for (uint32_t i = 0; i < subresources.size(); ++i)
            if (!serializeBin(out, subresources[i]))
                return false;
        return true;
    }

    bool readBin(std::istream& in) override {
        decltype(subresources.layerCount) layerCount;
        decltype(subresources.mipCount) mipCount;
        decltype(subresources.aspects) aspects;
        if (!serializeBin(in, layerCount))
            return false;
        if (!serializeBin(in, mipCount))
            return false;
        if (!serializeBin(in, aspects))
            return false;
        subresources =
          drv::ImagePerSubresourceData<ImageSubresStateStat, 1>(layerCount, mipCount, aspects);
        for (uint32_t i = 0; i < subresources.size(); ++i)
            if (!serializeBin(in, subresources[i]))
                return false;
        return true;
    }

    void writeJson(json& out) const override {
        out = json::object();
        out["layerCount"] = subresources.layerCount;
        out["mipCount"] = subresources.mipCount;
        out["aspects"] = subresources.aspects;
        json subres = json::array();
        for (uint32_t i = 0; i < subresources.size(); ++i)
            subres.push_back(serialize(subresources[i]));
        out["subres"] = std::move(subres);
    }

    void readJson(const json& in) override {
        if (!in.is_object())
            throw std::runtime_error("Json object expected");
        uint32_t layerCount = in["layerCount"];
        uint32_t mipCount = in["mipCount"];
        drv::ImageAspectBitType aspects = in["aspects"];
        subresources =
          drv::ImagePerSubresourceData<ImageSubresStateStat, 1>(layerCount, mipCount, aspects);
        for (uint32_t i = 0; i < subresources.size(); ++i)
            serialize(in["subres"][i], subresources[i]);
    }

    bool isCompatible(const drv::TextureInfo& info) const {
        return info.arraySize == subresources.layerCount && info.numMips == subresources.mipCount
               && info.aspects == subresources.aspects;
    }

    void init(const drv::TextureInfo& info) {
        subresources = drv::ImagePerSubresourceData<ImageSubresStateStat, 1>(
          info.arraySize, info.numMips, info.aspects);
    }
};

using SemaphoreSyncData = PipelineStagesStat;

using ImageStateStat = ImageSubresourcesData<ImageSubresStateStat>;
