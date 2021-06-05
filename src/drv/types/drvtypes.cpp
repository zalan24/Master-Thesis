#include "drvtypes.h"

using namespace drv;

#include "drverror.h"

PipelineStages::PipelineStageFlagBits PipelineStages::getEarliestStage() const {
    return getStage(0);
}

PipelineStages::PipelineStages(FlagType flags) : stageFlags(flags) {
}
PipelineStages::PipelineStages(PipelineStageFlagBits stage) : stageFlags(stage) {
}
void PipelineStages::add(FlagType flags) {
    stageFlags |= flags;
}
void PipelineStages::add(PipelineStageFlagBits stage) {
    stageFlags |= stage;
}
void PipelineStages::add(const PipelineStages& stages) {
    stageFlags |= stages.stageFlags;
}
bool PipelineStages::hasAllStages_resolved(FlagType flags) const {
    return (stageFlags & flags) == flags;
}
bool PipelineStages::hasAllStages_resolved(PipelineStageFlagBits stage) const {
    return hasAllStages_resolved(FlagType(stage));
}
bool PipelineStages::hasAnyStage_resolved(FlagType flags) const {
    return (stageFlags & flags) != 0;
}
bool PipelineStages::hasAnyStages_resolved(PipelineStageFlagBits stage) const {
    return hasAnyStage_resolved(stage);
}
PipelineStages::FlagType PipelineStages::get_graphics_bits() {
    return DRAW_INDIRECT_BIT | VERTEX_INPUT_BIT | VERTEX_SHADER_BIT
           | TESSELLATION_CONTROL_SHADER_BIT | TESSELLATION_EVALUATION_SHADER_BIT
           | GEOMETRY_SHADER_BIT | FRAGMENT_SHADER_BIT | EARLY_FRAGMENT_TESTS_BIT
           | LATE_FRAGMENT_TESTS_BIT | COLOR_ATTACHMENT_OUTPUT_BIT;
    //     | MESH_SHADER_BIT_NV
    //    | CONDITIONAL_RENDERING_BIT_EXT | TRANSFORM_FEEDBACK_BIT_EXT
    //    | FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR | TASK_SHADER_BIT_NV
    //    | FRAGMENT_DENSITY_PROCESS_BIT_EXT;
}
PipelineStages::FlagType PipelineStages::get_all_bits(CommandTypeBase queueSupport) {
    FlagType ret = HOST_BIT | TOP_OF_PIPE_BIT | BOTTOM_OF_PIPE_BIT;
    if (queueSupport & CMD_TYPE_TRANSFER)
        ret |= TRANSFER_BIT;
    if (queueSupport & CMD_TYPE_GRAPHICS)
        ret |= get_graphics_bits();
    if (queueSupport & CMD_TYPE_COMPUTE)
        ret |= COMPUTE_SHADER_BIT;
    return ret;
}
uint32_t PipelineStages::getStageCount() const {
    FlagType stages = stageFlags;
    uint32_t ret = 0;
    while (stages) {
        ret += stages & 0b1;
        stages >>= 1;
    }
    return ret;
}
PipelineStages::PipelineStageFlagBits PipelineStages::getStage(uint32_t index) const {
    FlagType stages = stageFlags;
    FlagType ret = 1;
    while (ret <= stages) {
        if (stages & ret)
            if (index-- == 0)
                return static_cast<PipelineStageFlagBits>(ret);
        ret <<= 1;
    }
    throw std::runtime_error("Invalid index for stage");
}

PipelineStages drv::get_image_usage_stages(ImageResourceUsageFlag usages) {
    ImageResourceUsageFlag usage = 1;
    PipelineStages ret;
    while (usages) {
        if (usages & 1) {
            switch (static_cast<ImageResourceUsage>(usage)) {
                case IMAGE_USAGE_TRANSFER_DESTINATION:
                    ret.add(PipelineStages::TRANSFER_BIT);
                    break;
                case IMAGE_USAGE_TRANSFER_SOURCE:
                    ret.add(PipelineStages::TRANSFER_BIT);
                    break;
                case IMAGE_USAGE_PRESENT:
                    ret.add(PipelineStages::BOTTOM_OF_PIPE_BIT);  // based on vulkan spec
                    break;
                case IMAGE_USAGE_ATTACHMENT_INPUT:
                    ret.add(PipelineStages::FRAGMENT_SHADER_BIT);
                    break;
                case IMAGE_USAGE_COLOR_OUTPUT_READ:
                    ret.add(PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT);
                    break;
                case IMAGE_USAGE_COLOR_OUTPUT_WRITE:
                    ret.add(PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT);
                    break;
                case IMAGE_USAGE_DEPTH_STENCIL_READ:
                    ret.add(PipelineStages::EARLY_FRAGMENT_TESTS_BIT);
                    break;
                case IMAGE_USAGE_DEPTH_STENCIL_WRITE:
                    ret.add(PipelineStages::LATE_FRAGMENT_TESTS_BIT);
                    break;
            }
        }
        usages >>= 1;
        usage <<= 1;
    }
    return ret;
}

MemoryBarrier::AccessFlagBitType drv::get_image_usage_accesses(ImageResourceUsageFlag usages) {
    ImageResourceUsageFlag usage = 1;
    MemoryBarrier::AccessFlagBitType ret = 0;
    while (usages) {
        if (usages & 1) {
            switch (static_cast<ImageResourceUsage>(usage)) {
                case IMAGE_USAGE_TRANSFER_DESTINATION:
                    ret |= MemoryBarrier::AccessFlagBits::TRANSFER_WRITE_BIT;
                    break;
                case IMAGE_USAGE_TRANSFER_SOURCE:
                    ret |= MemoryBarrier::AccessFlagBits::TRANSFER_READ_BIT;
                    break;
                case IMAGE_USAGE_PRESENT:
                    ret |= 0;  // made visible automatically by presentation engine
                    break;
                case IMAGE_USAGE_ATTACHMENT_INPUT:
                    ret |= MemoryBarrier::AccessFlagBits::INPUT_ATTACHMENT_READ_BIT;
                    break;
                case IMAGE_USAGE_COLOR_OUTPUT_READ:
                    ret |= MemoryBarrier::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT;
                    break;
                case IMAGE_USAGE_COLOR_OUTPUT_WRITE:
                    ret |= MemoryBarrier::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT;
                    break;
                case IMAGE_USAGE_DEPTH_STENCIL_READ:
                    ret |= MemoryBarrier::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT;
                    break;
                case IMAGE_USAGE_DEPTH_STENCIL_WRITE:
                    ret |= MemoryBarrier::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                    break;
            }
        }
        usages >>= 1;
        usage <<= 1;
    }
    return ret;
}
