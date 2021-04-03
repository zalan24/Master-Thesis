#include "drvtypes.h"

using namespace drv;

#include "drverror.h"

PipelineStages drv::get_image_usage_stages(ImageResourceUsageFlag usages) {
    ImageResourceUsageFlag usage = 1;
    PipelineStages ret;
    while (usages) {
        if (usages & 1) {
            switch (static_cast<ImageResourceUsage>(usage)) {
                case IMAGE_USAGE_TRANSFER_DESTINATION:
                    ret.add(PipelineStages::TRANSFER_BIT);
                    break;
                case IMAGE_USAGE_PRESENT:
                    ret.add(PipelineStages::BOTTOM_OF_PIPE_BIT);  // based on vulkan spec
                    break;
                case IMAGE_USAGE_ATTACHMENT_INPUT:
                    ret.add();
                    break;
                case IMAGE_USAGE_COLOR_OUTPUT_READ:
                    ret.add();
                    break;
                case IMAGE_USAGE_COLOR_OUTPUT_WRITE:
                    ret.add();
                    break;
                case IMAGE_USAGE_DEPTH_STENCIL_READ:
                    ret.add();
                    break;
                case IMAGE_USAGE_DEPTH_STENCIL_WRITE:
                    ret.add();
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

ImageLayoutMask drv::get_accepted_image_layouts(ImageResourceUsageFlag usages) {
    ImageResourceUsageFlag usage = 1;
    ImageLayoutMask ret = get_all_layouts_mask();
    while (usages) {
        if (usages & 1) {
            switch (static_cast<ImageResourceUsage>(usage)) {
                case IMAGE_USAGE_TRANSFER_DESTINATION:
                    ret &= static_cast<ImageLayoutMask>(drv::ImageLayout::GENERAL)
                           | static_cast<ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
                           | static_cast<ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);
                    break;
                case IMAGE_USAGE_PRESENT:
                    ret &= static_cast<ImageLayoutMask>(drv::ImageLayout::PRESENT_SRC_KHR)
                           | static_cast<ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);
                    break;
                case IMAGE_USAGE_ATTACHMENT_INPUT:
                    ret &=
                      static_cast<ImageLayoutMask>(drv::ImageLayout::GENERAL)
                      | static_cast<ImageLayoutMask>(drv::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                      | static_cast<ImageLayoutMask>(
                        drv::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
                    break;
                case IMAGE_USAGE_COLOR_OUTPUT_READ:
                    ret &=
                      static_cast<ImageLayoutMask>(drv::ImageLayout::GENERAL)
                      | static_cast<ImageLayoutMask>(drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                    break;
                case IMAGE_USAGE_COLOR_OUTPUT_WRITE:
                    ret &=
                      static_cast<ImageLayoutMask>(drv::ImageLayout::GENERAL)
                      | static_cast<ImageLayoutMask>(drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                    break;
                case IMAGE_USAGE_DEPTH_STENCIL_READ:
                    ret &= static_cast<ImageLayoutMask>(drv::ImageLayout::GENERAL)
                           | static_cast<ImageLayoutMask>(
                             drv::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
                           | static_cast<ImageLayoutMask>(
                             drv::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                    break;
                case IMAGE_USAGE_DEPTH_STENCIL_WRITE:
                    ret &= static_cast<ImageLayoutMask>(drv::ImageLayout::GENERAL)
                           | static_cast<ImageLayoutMask>(
                             drv::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                    break;
            }
        }
        usages >>= 1;
        usage <<= 1;
    }
    return ret;
}

ImageLayoutMask drv::get_all_layouts_mask() {
    return static_cast<ImageLayoutMask>(drv::ImageLayout::UNDEFINED)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::GENERAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::TRANSFER_SRC_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::PREINITIALIZED)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::PRESENT_SRC_KHR)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);
}

ImageAspectBitType drv::get_format_aspects(ImageFormat format) {
    switch (format) {
        case ImageFormat::UNDEFINED:
            return 0;
        case ImageFormat::R4G4_UNORM_PACK8:
        case ImageFormat::R4G4B4A4_UNORM_PACK16:
        case ImageFormat::B4G4R4A4_UNORM_PACK16:
        case ImageFormat::R5G6B5_UNORM_PACK16:
        case ImageFormat::B5G6R5_UNORM_PACK16:
        case ImageFormat::R5G5B5A1_UNORM_PACK16:
        case ImageFormat::B5G5R5A1_UNORM_PACK16:
        case ImageFormat::A1R5G5B5_UNORM_PACK16:
        case ImageFormat::R8_UNORM:
        case ImageFormat::R8_SNORM:
        case ImageFormat::R8_USCALED:
        case ImageFormat::R8_SSCALED:
        case ImageFormat::R8_UINT:
        case ImageFormat::R8_SINT:
        case ImageFormat::R8_SRGB:
        case ImageFormat::R8G8_UNORM:
        case ImageFormat::R8G8_SNORM:
        case ImageFormat::R8G8_USCALED:
        case ImageFormat::R8G8_SSCALED:
        case ImageFormat::R8G8_UINT:
        case ImageFormat::R8G8_SINT:
        case ImageFormat::R8G8_SRGB:
        case ImageFormat::R8G8B8_UNORM:
        case ImageFormat::R8G8B8_SNORM:
        case ImageFormat::R8G8B8_USCALED:
        case ImageFormat::R8G8B8_SSCALED:
        case ImageFormat::R8G8B8_UINT:
        case ImageFormat::R8G8B8_SINT:
        case ImageFormat::R8G8B8_SRGB:
        case ImageFormat::B8G8R8_UNORM:
        case ImageFormat::B8G8R8_SNORM:
        case ImageFormat::B8G8R8_USCALED:
        case ImageFormat::B8G8R8_SSCALED:
        case ImageFormat::B8G8R8_UINT:
        case ImageFormat::B8G8R8_SINT:
        case ImageFormat::B8G8R8_SRGB:
        case ImageFormat::R8G8B8A8_UNORM:
        case ImageFormat::R8G8B8A8_SNORM:
        case ImageFormat::R8G8B8A8_USCALED:
        case ImageFormat::R8G8B8A8_SSCALED:
        case ImageFormat::R8G8B8A8_UINT:
        case ImageFormat::R8G8B8A8_SINT:
        case ImageFormat::R8G8B8A8_SRGB:
        case ImageFormat::B8G8R8A8_UNORM:
        case ImageFormat::B8G8R8A8_SNORM:
        case ImageFormat::B8G8R8A8_USCALED:
        case ImageFormat::B8G8R8A8_SSCALED:
        case ImageFormat::B8G8R8A8_UINT:
        case ImageFormat::B8G8R8A8_SINT:
        case ImageFormat::B8G8R8A8_SRGB:
        case ImageFormat::A8B8G8R8_UNORM_PACK32:
        case ImageFormat::A8B8G8R8_SNORM_PACK32:
        case ImageFormat::A8B8G8R8_USCALED_PACK32:
        case ImageFormat::A8B8G8R8_SSCALED_PACK32:
        case ImageFormat::A8B8G8R8_UINT_PACK32:
        case ImageFormat::A8B8G8R8_SINT_PACK32:
        case ImageFormat::A8B8G8R8_SRGB_PACK32:
        case ImageFormat::A2R10G10B10_UNORM_PACK32:
        case ImageFormat::A2R10G10B10_SNORM_PACK32:
        case ImageFormat::A2R10G10B10_USCALED_PACK32:
        case ImageFormat::A2R10G10B10_SSCALED_PACK32:
        case ImageFormat::A2R10G10B10_UINT_PACK32:
        case ImageFormat::A2R10G10B10_SINT_PACK32:
        case ImageFormat::A2B10G10R10_UNORM_PACK32:
        case ImageFormat::A2B10G10R10_SNORM_PACK32:
        case ImageFormat::A2B10G10R10_USCALED_PACK32:
        case ImageFormat::A2B10G10R10_SSCALED_PACK32:
        case ImageFormat::A2B10G10R10_UINT_PACK32:
        case ImageFormat::A2B10G10R10_SINT_PACK32:
        case ImageFormat::R16_UNORM:
        case ImageFormat::R16_SNORM:
        case ImageFormat::R16_USCALED:
        case ImageFormat::R16_SSCALED:
        case ImageFormat::R16_UINT:
        case ImageFormat::R16_SINT:
        case ImageFormat::R16_SFLOAT:
        case ImageFormat::R16G16_UNORM:
        case ImageFormat::R16G16_SNORM:
        case ImageFormat::R16G16_USCALED:
        case ImageFormat::R16G16_SSCALED:
        case ImageFormat::R16G16_UINT:
        case ImageFormat::R16G16_SINT:
        case ImageFormat::R16G16_SFLOAT:
        case ImageFormat::R16G16B16_UNORM:
        case ImageFormat::R16G16B16_SNORM:
        case ImageFormat::R16G16B16_USCALED:
        case ImageFormat::R16G16B16_SSCALED:
        case ImageFormat::R16G16B16_UINT:
        case ImageFormat::R16G16B16_SINT:
        case ImageFormat::R16G16B16_SFLOAT:
        case ImageFormat::R16G16B16A16_UNORM:
        case ImageFormat::R16G16B16A16_SNORM:
        case ImageFormat::R16G16B16A16_USCALED:
        case ImageFormat::R16G16B16A16_SSCALED:
        case ImageFormat::R16G16B16A16_UINT:
        case ImageFormat::R16G16B16A16_SINT:
        case ImageFormat::R16G16B16A16_SFLOAT:
        case ImageFormat::R32_UINT:
        case ImageFormat::R32_SINT:
        case ImageFormat::R32_SFLOAT:
        case ImageFormat::R32G32_UINT:
        case ImageFormat::R32G32_SINT:
        case ImageFormat::R32G32_SFLOAT:
        case ImageFormat::R32G32B32_UINT:
        case ImageFormat::R32G32B32_SINT:
        case ImageFormat::R32G32B32_SFLOAT:
        case ImageFormat::R32G32B32A32_UINT:
        case ImageFormat::R32G32B32A32_SINT:
        case ImageFormat::R32G32B32A32_SFLOAT:
        case ImageFormat::R64_UINT:
        case ImageFormat::R64_SINT:
        case ImageFormat::R64_SFLOAT:
        case ImageFormat::R64G64_UINT:
        case ImageFormat::R64G64_SINT:
        case ImageFormat::R64G64_SFLOAT:
        case ImageFormat::R64G64B64_UINT:
        case ImageFormat::R64G64B64_SINT:
        case ImageFormat::R64G64B64_SFLOAT:
        case ImageFormat::R64G64B64A64_UINT:
        case ImageFormat::R64G64B64A64_SINT:
        case ImageFormat::R64G64B64A64_SFLOAT:
        case ImageFormat::B10G11R11_UFLOAT_PACK32:
        case ImageFormat::E5B9G9R9_UFLOAT_PACK32:
            return COLOR_BIT;
        case ImageFormat::D16_UNORM:
        case ImageFormat::X8_D24_UNORM_PACK32:
        case ImageFormat::D32_SFLOAT:
            return DEPTH_BIT;
        case ImageFormat::S8_UINT:
            return STENCIL_BIT;
        case ImageFormat::D16_UNORM_S8_UINT:
        case ImageFormat::D24_UNORM_S8_UINT:
        case ImageFormat::D32_SFLOAT_S8_UINT:
            return DEPTH_BIT | STENCIL_BIT;
        case ImageFormat::BC1_RGB_UNORM_BLOCK:
        case ImageFormat::BC1_RGB_SRGB_BLOCK:
        case ImageFormat::BC1_RGBA_UNORM_BLOCK:
        case ImageFormat::BC1_RGBA_SRGB_BLOCK:
        case ImageFormat::BC2_UNORM_BLOCK:
        case ImageFormat::BC2_SRGB_BLOCK:
        case ImageFormat::BC3_UNORM_BLOCK:
        case ImageFormat::BC3_SRGB_BLOCK:
        case ImageFormat::BC4_UNORM_BLOCK:
        case ImageFormat::BC4_SNORM_BLOCK:
        case ImageFormat::BC5_UNORM_BLOCK:
        case ImageFormat::BC5_SNORM_BLOCK:
        case ImageFormat::BC6H_UFLOAT_BLOCK:
        case ImageFormat::BC6H_SFLOAT_BLOCK:
        case ImageFormat::BC7_UNORM_BLOCK:
        case ImageFormat::BC7_SRGB_BLOCK:
        case ImageFormat::ETC2_R8G8B8_UNORM_BLOCK:
        case ImageFormat::ETC2_R8G8B8_SRGB_BLOCK:
        case ImageFormat::ETC2_R8G8B8A1_UNORM_BLOCK:
        case ImageFormat::ETC2_R8G8B8A1_SRGB_BLOCK:
        case ImageFormat::ETC2_R8G8B8A8_UNORM_BLOCK:
        case ImageFormat::ETC2_R8G8B8A8_SRGB_BLOCK:
        case ImageFormat::EAC_R11_UNORM_BLOCK:
        case ImageFormat::EAC_R11_SNORM_BLOCK:
        case ImageFormat::EAC_R11G11_UNORM_BLOCK:
        case ImageFormat::EAC_R11G11_SNORM_BLOCK:
        case ImageFormat::ASTC_4x4_UNORM_BLOCK:
        case ImageFormat::ASTC_4x4_SRGB_BLOCK:
        case ImageFormat::ASTC_5x4_UNORM_BLOCK:
        case ImageFormat::ASTC_5x4_SRGB_BLOCK:
        case ImageFormat::ASTC_5x5_UNORM_BLOCK:
        case ImageFormat::ASTC_5x5_SRGB_BLOCK:
        case ImageFormat::ASTC_6x5_UNORM_BLOCK:
        case ImageFormat::ASTC_6x5_SRGB_BLOCK:
        case ImageFormat::ASTC_6x6_UNORM_BLOCK:
        case ImageFormat::ASTC_6x6_SRGB_BLOCK:
        case ImageFormat::ASTC_8x5_UNORM_BLOCK:
        case ImageFormat::ASTC_8x5_SRGB_BLOCK:
        case ImageFormat::ASTC_8x6_UNORM_BLOCK:
        case ImageFormat::ASTC_8x6_SRGB_BLOCK:
        case ImageFormat::ASTC_8x8_UNORM_BLOCK:
        case ImageFormat::ASTC_8x8_SRGB_BLOCK:
        case ImageFormat::ASTC_10x5_UNORM_BLOCK:
        case ImageFormat::ASTC_10x5_SRGB_BLOCK:
        case ImageFormat::ASTC_10x6_UNORM_BLOCK:
        case ImageFormat::ASTC_10x6_SRGB_BLOCK:
        case ImageFormat::ASTC_10x8_UNORM_BLOCK:
        case ImageFormat::ASTC_10x8_SRGB_BLOCK:
        case ImageFormat::ASTC_10x10_UNORM_BLOCK:
        case ImageFormat::ASTC_10x10_SRGB_BLOCK:
        case ImageFormat::ASTC_12x10_UNORM_BLOCK:
        case ImageFormat::ASTC_12x10_SRGB_BLOCK:
        case ImageFormat::ASTC_12x12_UNORM_BLOCK:
        case ImageFormat::ASTC_12x12_SRGB_BLOCK:
            return COLOR_BIT;
    }
}

PipelineStages::PipelineStageFlagBits PipelineStages::getEarliestStage(
  CommandTypeMask queueSupport) const {
    return getStage(queueSupport, 0);
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
    drv::drv_assert((stageFlags & ALL_GRAPHICS_BIT) == 0 && (stageFlags & ALL_COMMANDS_BIT) == 0);
    return (stageFlags & flags) == flags;
}
bool PipelineStages::hasAllStages_resolved(PipelineStageFlagBits stage) const {
    return hasAllStages_resolved(FlagType(stage));
}
bool PipelineStages::hasAnyStage_resolved(FlagType flags) const {
    drv::drv_assert((stageFlags & ALL_GRAPHICS_BIT) == 0 && (stageFlags & ALL_COMMANDS_BIT) == 0);
    return (stageFlags & flags) != 0;
}
bool PipelineStages::hasAnyStages_resolved(PipelineStageFlagBits stage) const {
    return hasAnyStages_resolved(FlagType(stage));
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
PipelineStages::FlagType PipelineStages::resolve(CommandTypeMask queueSupport) const {
    FlagType ret = stageFlags;
    if (ret & ALL_GRAPHICS_BIT)
        ret = (ret ^ ALL_GRAPHICS_BIT) | get_graphics_bits();
    if (ret & ALL_COMMANDS_BIT)
        ret = (ret ^ ALL_COMMANDS_BIT) | get_all_bits(queueSupport);
    return ret;
}
uint32_t PipelineStages::getStageCount(CommandTypeMask queueSupport) const {
    FlagType stages = resolve(queueSupport);
    uint32_t ret = 0;
    while (stages) {
        ret += stages & 0b1;
        stages >>= 1;
    }
    return ret;
}
PipelineStages::PipelineStageFlagBits PipelineStages::getStage(CommandTypeMask queueSupport,
                                                               uint32_t index) const {
    FlagType stages = resolve(queueSupport);
    FlagType ret = 1;
    while (ret <= stages) {
        if (stages & ret)
            if (index-- == 0)
                return static_cast<PipelineStageFlagBits>(ret);
        ret <<= 1;
    }
    throw std::runtime_error("Invalid index for stage");
}
