#include "drvpipeline_types.h"

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

drv::PipelineStages::FlagType drv::MemoryBarrier::get_supported_stages(AccessFlagBits access) {
    switch (access) {
        case INDIRECT_COMMAND_READ_BIT:
            return PipelineStages::
              DRAW_INDIRECT_BIT;  // | PipelineStages::ACCELERATION_STRUCTURE_BUILD_BIT_NV;
        case INDEX_READ_BIT:
            return PipelineStages::VERTEX_INPUT_BIT;
        case VERTEX_ATTRIBUTE_READ_BIT:
            return PipelineStages::VERTEX_INPUT_BIT;
        case UNIFORM_READ_BIT:
            return 0
                   // | PipelineStages::TASK_SHADER_BIT_NV
                   // | PipelineStages::MESH_SHADER_BIT_NV
                   // | PipelineStages::RAY_TRACING_SHADER_BIT_NV
                   | PipelineStages::VERTEX_SHADER_BIT
                   | PipelineStages::TESSELLATION_CONTROL_SHADER_BIT
                   | PipelineStages::TESSELLATION_EVALUATION_SHADER_BIT
                   | PipelineStages::GEOMETRY_SHADER_BIT | PipelineStages::FRAGMENT_SHADER_BIT
                   | PipelineStages::COMPUTE_SHADER_BIT;
        case SHADER_READ_BIT:
            return 0
                   // | PipelineStages::ACCELERATION_STRUCTURE_BUILD_BIT_NV;
                   // | PipelineStages::TASK_SHADER_BIT_NV
                   // | PipelineStages::MESH_SHADER_BIT_NV
                   // | PipelineStages::RAY_TRACING_SHADER_BIT_NV
                   | PipelineStages::VERTEX_SHADER_BIT
                   | PipelineStages::TESSELLATION_CONTROL_SHADER_BIT
                   | PipelineStages::TESSELLATION_EVALUATION_SHADER_BIT
                   | PipelineStages::GEOMETRY_SHADER_BIT | PipelineStages::FRAGMENT_SHADER_BIT
                   | PipelineStages::COMPUTE_SHADER_BIT;
        case SHADER_WRITE_BIT:
            return 0
                   // | PipelineStages::TASK_SHADER_BIT_NV
                   // | PipelineStages::MESH_SHADER_BIT_NV
                   // | PipelineStages::RAY_TRACING_SHADER_BIT_NV
                   | PipelineStages::VERTEX_SHADER_BIT
                   | PipelineStages::TESSELLATION_CONTROL_SHADER_BIT
                   | PipelineStages::TESSELLATION_EVALUATION_SHADER_BIT
                   | PipelineStages::GEOMETRY_SHADER_BIT | PipelineStages::FRAGMENT_SHADER_BIT
                   | PipelineStages::COMPUTE_SHADER_BIT;
        case INPUT_ATTACHMENT_READ_BIT:
            return PipelineStages::FRAGMENT_SHADER_BIT;
        case COLOR_ATTACHMENT_WRITE_BIT:
            return PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT;
        case COLOR_ATTACHMENT_READ_BIT:
            return PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT;
        case DEPTH_STENCIL_ATTACHMENT_READ_BIT:
            return PipelineStages::EARLY_FRAGMENT_TESTS_BIT
                   | PipelineStages::LATE_FRAGMENT_TESTS_BIT;
        case DEPTH_STENCIL_ATTACHMENT_WRITE_BIT:
            return PipelineStages::EARLY_FRAGMENT_TESTS_BIT
                   | PipelineStages::LATE_FRAGMENT_TESTS_BIT;
        case TRANSFER_READ_BIT:
            return PipelineStages::
              TRANSFER_BIT;  // | PipelineStages::ACCELERATION_STRUCTURE_BUILD_BIT_NV;
        case TRANSFER_WRITE_BIT:
            return PipelineStages::
              TRANSFER_BIT;  // | PipelineStages::ACCELERATION_STRUCTURE_BUILD_BIT_NV;
        case HOST_READ_BIT:
            return PipelineStages::HOST_BIT;
        case HOST_WRITE_BIT:
            return PipelineStages::HOST_BIT;
    }
}
