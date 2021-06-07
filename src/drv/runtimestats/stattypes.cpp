#include "stattypes.h"

void PipelineStagesStat::set(const drv::PipelineStages& _stages) {
    for (uint32_t i = 0; i < stages.size(); ++i)
        stages[i] = 0;
    for (uint32_t i = 0; i < _stages.getStageCount(); ++i)
        stages[drv::PipelineStages::get_stage_index(_stages.getStage(i))] = _stages.getStage(i);
}

void PipelineStagesStat::append(const drv::PipelineStages& _stages) {
    for (uint32_t i = 0; i < stages.size(); ++i) {
        stages[i] *= (1.f - EXP_AVG);
        if (_stages.stageFlags & drv::PipelineStages::get_stage(i))
            stages[i] += EXP_AVG * 1.f;
    }
}

drv::PipelineStages PipelineStagesStat::get(ApproximationMode mode) const {
    drv::PipelineStages::FlagType stageFlags = 0;
    for (uint32_t i = 0; i < stages.size(); ++i) {
        switch (mode) {
            case TEND_TO_FALSE:
                if (stages[i] > 1.f - THRESHOLD)
                    stageFlags |= drv::PipelineStages::get_stage(i);
                break;
            case TEND_TO_TRUE:
                if (stages[i] > THRESHOLD)
                    stageFlags |= drv::PipelineStages::get_stage(i);
                break;
        }
    }
    return drv::PipelineStages(stageFlags);
}

PipelineStagesStat::PipelineStagesStat() {
    for (uint32_t i = 0; i < stages.size(); ++i)
        stages[i] = 0.5f;
}

void MemoryAccessStat::set(const drv::MemoryBarrier::AccessFlagBitType& _mask) {
    for (uint32_t i = 0; i < mask.size(); ++i)
        mask[i] = (_mask & drv::MemoryBarrier::get_access(i)) > 0 ? 1.f : 0.f;
}

void MemoryAccessStat::append(const drv::MemoryBarrier::AccessFlagBitType& _mask) {
    for (uint32_t i = 0; i < mask.size(); ++i) {
        mask[i] *= (1.f - EXP_AVG);
        if (_mask & drv::MemoryBarrier::get_access(i))
            mask[i] += EXP_AVG * 1.f;
    }
}

drv::MemoryBarrier::AccessFlagBitType MemoryAccessStat::get(ApproximationMode mode) const {
    drv::MemoryBarrier::AccessFlagBitType ret = 0;
    for (uint32_t i = 0; i < mask.size(); ++i) {
        switch (mode) {
            case TEND_TO_FALSE:
                if (mask[i] > 1.f - THRESHOLD)
                    ret |= drv::MemoryBarrier::get_access(i);
                break;
            case TEND_TO_TRUE:
                if (mask[i] > THRESHOLD)
                    ret |= drv::MemoryBarrier::get_access(i);
                break;
        }
    }
    return ret;
}

MemoryAccessStat::MemoryAccessStat() {
    for (uint32_t i = 0; i < mask.size(); ++i)
        mask[i] = 0.5f;
}

void ImageLayoutStat::set(const drv::ImageLayout layout) {
    for (uint32_t i = 0; i < layouts.size(); ++i)
        layouts[i] = 0.f;
    layouts[drv::get_image_layout_index(layout)] = 1.f;
}

void ImageLayoutStat::append(const drv::ImageLayout layout) {
    for (uint32_t i = 0; i < layouts.size(); ++i)
        layouts[i] *= (1.f - EXP_AVG);
    layouts[drv::get_image_layout_index(layout)] += 1.f * EXP_AVG;
}

drv::ImageLayout ImageLayoutStat::get() const {
    uint32_t max = 0;
    for (uint32_t i = 1; i < layouts.size(); ++i)
        if (layouts[i] > layouts[max])
            max = i;
    return drv::get_image_layout(max);
}

ImageLayoutStat::ImageLayoutStat() {
    for (uint32_t i = 0; i < layouts.size(); ++i)
        layouts[i] = 0.f;
    layouts[drv::get_image_layout_index(drv::ImageLayout::GENERAL)] = 1.f;
}

void SimpleSubresStateStat::set(const drv::PerSubresourceRangeTrackData& data) {
    usableStages.set(drv::PipelineStages(data.usableStages));
    writes.set(drv::PipelineStages(data.ongoingWrites));
    reads.set(drv::PipelineStages(data.ongoingReads));
    dirtyMask.set(data.dirtyMask);
    visible.set(data.visible);
}

void SimpleSubresStateStat::append(const drv::PerSubresourceRangeTrackData& data) {
    usableStages.append(drv::PipelineStages(data.usableStages));
    writes.append(drv::PipelineStages(data.ongoingWrites));
    reads.append(drv::PipelineStages(data.ongoingReads));
    dirtyMask.append(data.dirtyMask);
    visible.append(data.visible);
}

void SimpleSubresStateStat::get(drv::PerSubresourceRangeTrackData& data) const {
    // Assumed usable stages will be used as sync destitanion. It can be a problem for unsupported stages
    data.usableStages = usableStages.get(PipelineStagesStat::TEND_TO_FALSE).stageFlags;
    data.ongoingWrites = writes.get(PipelineStagesStat::TEND_TO_FALSE).stageFlags;
    data.ongoingReads = reads.get(PipelineStagesStat::TEND_TO_FALSE).stageFlags;
    data.dirtyMask = dirtyMask.get(MemoryAccessStat::TEND_TO_FALSE);
    // Assumed accesses will be mode available. This could be a problem if an access is not supported at all
    data.visible = visible.get(MemoryAccessStat::TEND_TO_FALSE);

    if (data.usableStages == 0)
        data.usableStages |= drv::PipelineStages::BOTTOM_OF_PIPE_BIT;
}

void SubresStateStat::resizeFamilies(size_t size) {
    FixedArray<float, 8> cpy(size);
    for (uint32_t i = 0; i < size; ++i)
        cpy[i] = 0;
    for (uint32_t i = 0; i < size && i < queueFamilies.size(); ++i)
        cpy[i] = queueFamilies[i];
    queueFamilies = std::move(cpy);
}

void SubresStateStat::set(const drv::PerSubresourceRangeTrackData& data) {
    noFamily = data.ownership == drv::IGNORE_FAMILY ? 1.f : 0.f;
    if (data.ownership != drv::IGNORE_FAMILY && data.ownership >= queueFamilies.size())
        resizeFamilies(data.ownership + 1);
    for (uint32_t i = 0; i < queueFamilies.size(); ++i)
        queueFamilies[i] = 0;
    if (data.ownership != drv::IGNORE_FAMILY) {
        queueFamilies[data.ownership] = 1.f;
    }
    simpleStats.set(data);
}

void SubresStateStat::append(const drv::PerSubresourceRangeTrackData& data) {
    noFamily *= 1.f - EXP_AVG;
    noFamily += (data.ownership == drv::IGNORE_FAMILY ? 1.f : 0.f) * EXP_AVG;
    if (data.ownership != drv::IGNORE_FAMILY && data.ownership >= queueFamilies.size())
        resizeFamilies(data.ownership + 1);
    for (uint32_t i = 0; i < queueFamilies.size(); ++i)
        queueFamilies[i] *= 1.f - EXP_AVG;
    if (data.ownership != drv::IGNORE_FAMILY)
        queueFamilies[data.ownership] += 1.f * EXP_AVG;
    simpleStats.append(data);
}

void SubresStateStat::get(drv::PerSubresourceRangeTrackData& data) const {
    uint32_t maxFamily = 0;
    for (uint32_t i = 1; i < queueFamilies.size(); ++i)
        if (queueFamilies[i] > queueFamilies[maxFamily])
            maxFamily = i;
    data.ownership = (maxFamily >= queueFamilies.size() || noFamily > queueFamilies[maxFamily])
                       ? drv::IGNORE_FAMILY
                       : maxFamily;
    simpleStats.get(data);
}

SubresStateStat::SubresStateStat() : queueFamilies(0) {
    noFamily = 1.f;
    for (uint32_t i = 0; i < queueFamilies.size(); ++i)
        queueFamilies[i] = 0;
}

void ImageSubresStateStat::set(const drv::ImageSubresourceTrackData& data) {
    subres.set(data);
    layout.set(data.layout);
}

void ImageSubresStateStat::append(const drv::ImageSubresourceTrackData& data) {
    subres.append(data);
    layout.append(data.layout);
}

void ImageSubresStateStat::get(drv::ImageSubresourceTrackData& data) const {
    subres.get(data);
    data.layout = layout.get();
}

bool ImageStateStat::writeBin(std::ostream& out) const {
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

bool ImageStateStat::readBin(std::istream& in) {
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

void ImageStateStat::writeJson(json& out) const {
    out = json::object();
    out["layerCount"] = subresources.layerCount;
    out["mipCount"] = subresources.mipCount;
    out["aspects"] = subresources.aspects;
    json subres = json::array();
    for (uint32_t i = 0; i < subresources.size(); ++i)
        subres.push_back(serialize(subresources[i]));
    out["subres"] = std::move(subres);
}

void ImageStateStat::readJson(const json& in) {
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

bool ImageStateStat::isCompatible(const drv::TextureInfo& info) const {
    return info.arraySize == subresources.layerCount && info.numMips == subresources.mipCount
           && info.aspects == subresources.aspects;
}

void ImageStateStat::init(const drv::TextureInfo& info) {
    subresources = drv::ImagePerSubresourceData<ImageSubresStateStat, 1>(
      info.arraySize, info.numMips, info.aspects);
}
