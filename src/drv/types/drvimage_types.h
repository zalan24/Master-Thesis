#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include <features.h>

#if ENABLE_RESOURCE_STACKTRACES
#    include <boost/stacktrace.hpp>
#endif

#include <fixedarray.hpp>
#include <serializable.h>

#include "drvres_types.h"
#include "drvresourceptrs.hpp"
#include "drvpipeline_types.h"

namespace drv
{
struct ImageId final : public IAutoSerializable<ImageId>
{
    using SubId = uint32_t;
    static constexpr SubId NO_SUB_ID = std::numeric_limits<SubId>::max();
    // Persistent capable id
    REFLECTABLE
    (
        (std::string) name, // not necessarily unique
        (SubId) subId       // eg. swapchain image index
    )

#if ENABLE_RESOURCE_STACKTRACES
    std::unique_ptr<boost::stacktrace> stackTrace;
#endif
    ImageId(std::string _name, uint32_t _subId)
      : name(std::move(_name)),
        subId(_subId)
#if ENABLE_RESOURCE_STACKTRACES
        ,
        stackTrace(std::make_unique<boost::stacktrace>())
#endif
    {
    }

    ImageId(std::string _name) : ImageId(std::move(_name), NO_SUB_ID) {}
    ImageId() : ImageId("unnamed") {}
    ImageId(const ImageId&) = default;
    ImageId& operator=(const ImageId&) = default;
    ImageId(ImageId&&) = default;
    ImageId& operator=(ImageId&&) = default;

    bool operator==(const ImageId& other) const {
        return name == other.name && subId == other.subId;
    }
    bool operator<(const ImageId& other) const {
        if (subId != other.subId)
            return subId < other.subId;
        return name < other.name;
    }
};

enum class ImageFormat
{
    UNDEFINED = 0,
    R4G4_UNORM_PACK8 = 1,
    R4G4B4A4_UNORM_PACK16 = 2,
    B4G4R4A4_UNORM_PACK16 = 3,
    R5G6B5_UNORM_PACK16 = 4,
    B5G6R5_UNORM_PACK16 = 5,
    R5G5B5A1_UNORM_PACK16 = 6,
    B5G5R5A1_UNORM_PACK16 = 7,
    A1R5G5B5_UNORM_PACK16 = 8,
    R8_UNORM = 9,
    R8_SNORM = 10,
    R8_USCALED = 11,
    R8_SSCALED = 12,
    R8_UINT = 13,
    R8_SINT = 14,
    R8_SRGB = 15,
    R8G8_UNORM = 16,
    R8G8_SNORM = 17,
    R8G8_USCALED = 18,
    R8G8_SSCALED = 19,
    R8G8_UINT = 20,
    R8G8_SINT = 21,
    R8G8_SRGB = 22,
    R8G8B8_UNORM = 23,
    R8G8B8_SNORM = 24,
    R8G8B8_USCALED = 25,
    R8G8B8_SSCALED = 26,
    R8G8B8_UINT = 27,
    R8G8B8_SINT = 28,
    R8G8B8_SRGB = 29,
    B8G8R8_UNORM = 30,
    B8G8R8_SNORM = 31,
    B8G8R8_USCALED = 32,
    B8G8R8_SSCALED = 33,
    B8G8R8_UINT = 34,
    B8G8R8_SINT = 35,
    B8G8R8_SRGB = 36,
    R8G8B8A8_UNORM = 37,
    R8G8B8A8_SNORM = 38,
    R8G8B8A8_USCALED = 39,
    R8G8B8A8_SSCALED = 40,
    R8G8B8A8_UINT = 41,
    R8G8B8A8_SINT = 42,
    R8G8B8A8_SRGB = 43,
    B8G8R8A8_UNORM = 44,
    B8G8R8A8_SNORM = 45,
    B8G8R8A8_USCALED = 46,
    B8G8R8A8_SSCALED = 47,
    B8G8R8A8_UINT = 48,
    B8G8R8A8_SINT = 49,
    B8G8R8A8_SRGB = 50,
    A8B8G8R8_UNORM_PACK32 = 51,
    A8B8G8R8_SNORM_PACK32 = 52,
    A8B8G8R8_USCALED_PACK32 = 53,
    A8B8G8R8_SSCALED_PACK32 = 54,
    A8B8G8R8_UINT_PACK32 = 55,
    A8B8G8R8_SINT_PACK32 = 56,
    A8B8G8R8_SRGB_PACK32 = 57,
    A2R10G10B10_UNORM_PACK32 = 58,
    A2R10G10B10_SNORM_PACK32 = 59,
    A2R10G10B10_USCALED_PACK32 = 60,
    A2R10G10B10_SSCALED_PACK32 = 61,
    A2R10G10B10_UINT_PACK32 = 62,
    A2R10G10B10_SINT_PACK32 = 63,
    A2B10G10R10_UNORM_PACK32 = 64,
    A2B10G10R10_SNORM_PACK32 = 65,
    A2B10G10R10_USCALED_PACK32 = 66,
    A2B10G10R10_SSCALED_PACK32 = 67,
    A2B10G10R10_UINT_PACK32 = 68,
    A2B10G10R10_SINT_PACK32 = 69,
    R16_UNORM = 70,
    R16_SNORM = 71,
    R16_USCALED = 72,
    R16_SSCALED = 73,
    R16_UINT = 74,
    R16_SINT = 75,
    R16_SFLOAT = 76,
    R16G16_UNORM = 77,
    R16G16_SNORM = 78,
    R16G16_USCALED = 79,
    R16G16_SSCALED = 80,
    R16G16_UINT = 81,
    R16G16_SINT = 82,
    R16G16_SFLOAT = 83,
    R16G16B16_UNORM = 84,
    R16G16B16_SNORM = 85,
    R16G16B16_USCALED = 86,
    R16G16B16_SSCALED = 87,
    R16G16B16_UINT = 88,
    R16G16B16_SINT = 89,
    R16G16B16_SFLOAT = 90,
    R16G16B16A16_UNORM = 91,
    R16G16B16A16_SNORM = 92,
    R16G16B16A16_USCALED = 93,
    R16G16B16A16_SSCALED = 94,
    R16G16B16A16_UINT = 95,
    R16G16B16A16_SINT = 96,
    R16G16B16A16_SFLOAT = 97,
    R32_UINT = 98,
    R32_SINT = 99,
    R32_SFLOAT = 100,
    R32G32_UINT = 101,
    R32G32_SINT = 102,
    R32G32_SFLOAT = 103,
    R32G32B32_UINT = 104,
    R32G32B32_SINT = 105,
    R32G32B32_SFLOAT = 106,
    R32G32B32A32_UINT = 107,
    R32G32B32A32_SINT = 108,
    R32G32B32A32_SFLOAT = 109,
    R64_UINT = 110,
    R64_SINT = 111,
    R64_SFLOAT = 112,
    R64G64_UINT = 113,
    R64G64_SINT = 114,
    R64G64_SFLOAT = 115,
    R64G64B64_UINT = 116,
    R64G64B64_SINT = 117,
    R64G64B64_SFLOAT = 118,
    R64G64B64A64_UINT = 119,
    R64G64B64A64_SINT = 120,
    R64G64B64A64_SFLOAT = 121,
    B10G11R11_UFLOAT_PACK32 = 122,
    E5B9G9R9_UFLOAT_PACK32 = 123,
    D16_UNORM = 124,
    X8_D24_UNORM_PACK32 = 125,
    D32_SFLOAT = 126,
    S8_UINT = 127,
    D16_UNORM_S8_UINT = 128,
    D24_UNORM_S8_UINT = 129,
    D32_SFLOAT_S8_UINT = 130,
    BC1_RGB_UNORM_BLOCK = 131,
    BC1_RGB_SRGB_BLOCK = 132,
    BC1_RGBA_UNORM_BLOCK = 133,
    BC1_RGBA_SRGB_BLOCK = 134,
    BC2_UNORM_BLOCK = 135,
    BC2_SRGB_BLOCK = 136,
    BC3_UNORM_BLOCK = 137,
    BC3_SRGB_BLOCK = 138,
    BC4_UNORM_BLOCK = 139,
    BC4_SNORM_BLOCK = 140,
    BC5_UNORM_BLOCK = 141,
    BC5_SNORM_BLOCK = 142,
    BC6H_UFLOAT_BLOCK = 143,
    BC6H_SFLOAT_BLOCK = 144,
    BC7_UNORM_BLOCK = 145,
    BC7_SRGB_BLOCK = 146,
    ETC2_R8G8B8_UNORM_BLOCK = 147,
    ETC2_R8G8B8_SRGB_BLOCK = 148,
    ETC2_R8G8B8A1_UNORM_BLOCK = 149,
    ETC2_R8G8B8A1_SRGB_BLOCK = 150,
    ETC2_R8G8B8A8_UNORM_BLOCK = 151,
    ETC2_R8G8B8A8_SRGB_BLOCK = 152,
    EAC_R11_UNORM_BLOCK = 153,
    EAC_R11_SNORM_BLOCK = 154,
    EAC_R11G11_UNORM_BLOCK = 155,
    EAC_R11G11_SNORM_BLOCK = 156,
    ASTC_4x4_UNORM_BLOCK = 157,
    ASTC_4x4_SRGB_BLOCK = 158,
    ASTC_5x4_UNORM_BLOCK = 159,
    ASTC_5x4_SRGB_BLOCK = 160,
    ASTC_5x5_UNORM_BLOCK = 161,
    ASTC_5x5_SRGB_BLOCK = 162,
    ASTC_6x5_UNORM_BLOCK = 163,
    ASTC_6x5_SRGB_BLOCK = 164,
    ASTC_6x6_UNORM_BLOCK = 165,
    ASTC_6x6_SRGB_BLOCK = 166,
    ASTC_8x5_UNORM_BLOCK = 167,
    ASTC_8x5_SRGB_BLOCK = 168,
    ASTC_8x6_UNORM_BLOCK = 169,
    ASTC_8x6_SRGB_BLOCK = 170,
    ASTC_8x8_UNORM_BLOCK = 171,
    ASTC_8x8_SRGB_BLOCK = 172,
    ASTC_10x5_UNORM_BLOCK = 173,
    ASTC_10x5_SRGB_BLOCK = 174,
    ASTC_10x6_UNORM_BLOCK = 175,
    ASTC_10x6_SRGB_BLOCK = 176,
    ASTC_10x8_UNORM_BLOCK = 177,
    ASTC_10x8_SRGB_BLOCK = 178,
    ASTC_10x10_UNORM_BLOCK = 179,
    ASTC_10x10_SRGB_BLOCK = 180,
    ASTC_12x10_UNORM_BLOCK = 181,
    ASTC_12x10_SRGB_BLOCK = 182,
    ASTC_12x12_UNORM_BLOCK = 183,
    ASTC_12x12_SRGB_BLOCK = 184,
    // G8B8G8R8_422_UNORM = 1000156000,
    // B8G8R8G8_422_UNORM = 1000156001,
    // G8_B8_R8_3PLANE_420_UNORM = 1000156002,
    // G8_B8R8_2PLANE_420_UNORM = 1000156003,
    // G8_B8_R8_3PLANE_422_UNORM = 1000156004,
    // G8_B8R8_2PLANE_422_UNORM = 1000156005,
    // G8_B8_R8_3PLANE_444_UNORM = 1000156006,
    // R10X6_UNORM_PACK16 = 1000156007,
    // R10X6G10X6_UNORM_2PACK16 = 1000156008,
    // R10X6G10X6B10X6A10X6_UNORM_4PACK16 = 1000156009,
    // G10X6B10X6G10X6R10X6_422_UNORM_4PACK16 = 1000156010,
    // B10X6G10X6R10X6G10X6_422_UNORM_4PACK16 = 1000156011,
    // G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 = 1000156012,
    // G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 = 1000156013,
    // G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 = 1000156014,
    // G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 = 1000156015,
    // G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 = 1000156016,
    // R12X4_UNORM_PACK16 = 1000156017,
    // R12X4G12X4_UNORM_2PACK16 = 1000156018,
    // R12X4G12X4B12X4A12X4_UNORM_4PACK16 = 1000156019,
    // G12X4B12X4G12X4R12X4_422_UNORM_4PACK16 = 1000156020,
    // B12X4G12X4R12X4G12X4_422_UNORM_4PACK16 = 1000156021,
    // G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 = 1000156022,
    // G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 = 1000156023,
    // G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 = 1000156024,
    // G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 = 1000156025,
    // G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 = 1000156026,
    // G16B16G16R16_422_UNORM = 1000156027,
    // B16G16R16G16_422_UNORM = 1000156028,
    // G16_B16_R16_3PLANE_420_UNORM = 1000156029,
    // G16_B16R16_2PLANE_420_UNORM = 1000156030,
    // G16_B16_R16_3PLANE_422_UNORM = 1000156031,
    // G16_B16R16_2PLANE_422_UNORM = 1000156032,
    // G16_B16_R16_3PLANE_444_UNORM = 1000156033,
    // PVRTC1_2BPP_UNORM_BLOCK_IMG = 1000054000,
    // PVRTC1_4BPP_UNORM_BLOCK_IMG = 1000054001,
    // PVRTC2_2BPP_UNORM_BLOCK_IMG = 1000054002,
    // PVRTC2_4BPP_UNORM_BLOCK_IMG = 1000054003,
    // PVRTC1_2BPP_SRGB_BLOCK_IMG = 1000054004,
    // PVRTC1_4BPP_SRGB_BLOCK_IMG = 1000054005,
    // PVRTC2_2BPP_SRGB_BLOCK_IMG = 1000054006,
    // PVRTC2_4BPP_SRGB_BLOCK_IMG = 1000054007,
    // ASTC_4x4_SFLOAT_BLOCK_EXT = 1000066000,
    // ASTC_5x4_SFLOAT_BLOCK_EXT = 1000066001,
    // ASTC_5x5_SFLOAT_BLOCK_EXT = 1000066002,
    // ASTC_6x5_SFLOAT_BLOCK_EXT = 1000066003,
    // ASTC_6x6_SFLOAT_BLOCK_EXT = 1000066004,
    // ASTC_8x5_SFLOAT_BLOCK_EXT = 1000066005,
    // ASTC_8x6_SFLOAT_BLOCK_EXT = 1000066006,
    // ASTC_8x8_SFLOAT_BLOCK_EXT = 1000066007,
    // ASTC_10x5_SFLOAT_BLOCK_EXT = 1000066008,
    // ASTC_10x6_SFLOAT_BLOCK_EXT = 1000066009,
    // ASTC_10x8_SFLOAT_BLOCK_EXT = 1000066010,
    // ASTC_10x10_SFLOAT_BLOCK_EXT = 1000066011,
    // ASTC_12x10_SFLOAT_BLOCK_EXT = 1000066012,
    // ASTC_12x12_SFLOAT_BLOCK_EXT = 1000066013,
    // A4R4G4B4_UNORM_PACK16_EXT = 1000340000,
    // A4B4G4R4_UNORM_PACK16_EXT = 1000340001
};

using ImageLayoutMask = uint32_t;
enum class ImageLayout : ImageLayoutMask
{
    UNDEFINED = 1 << 0,
    GENERAL = 1 << 1,
    COLOR_ATTACHMENT_OPTIMAL = 1 << 2,
    DEPTH_STENCIL_ATTACHMENT_OPTIMAL = 1 << 3,
    DEPTH_STENCIL_READ_ONLY_OPTIMAL = 1 << 4,
    SHADER_READ_ONLY_OPTIMAL = 1 << 5,
    TRANSFER_SRC_OPTIMAL = 1 << 6,
    TRANSFER_DST_OPTIMAL = 1 << 7,
    PREINITIALIZED = 1 << 8,
    PRESENT_SRC_KHR = 1 << 9,
    SHARED_PRESENT_KHR = 1 << 10
    // Provided by VK_VERSION_1_1
    // DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL = 1000117000,
    // // Provided by VK_VERSION_1_1
    // DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL = 1000117001,
    // // Provided by VK_VERSION_1_2
    // DEPTH_ATTACHMENT_OPTIMAL = 1000241000,
    // // Provided by VK_VERSION_1_2
    // DEPTH_READ_ONLY_OPTIMAL = 1000241001,
    // // Provided by VK_VERSION_1_2
    // STENCIL_ATTACHMENT_OPTIMAL = 1000241002,
    // // Provided by VK_VERSION_1_2
    // STENCIL_READ_ONLY_OPTIMAL = 1000241003,
    // // Provided by VK_KHR_swapchain
    // PRESENT_SRC_KHR = 1000001002,
    // // Provided by VK_KHR_shared_presentable_image
    // SHARED_PRESENT_KHR = 1000111000,
    // // Provided by VK_NV_shading_rate_image
    // SHADING_RATE_OPTIMAL_NV = 1000164003,
    // // Provided by VK_EXT_fragment_density_map
    // FRAGMENT_DENSITY_MAP_OPTIMAL_EXT = 1000218000,
    // // Provided by VK_KHR_maintenance2
    // DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR = DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
    // // Provided by VK_KHR_maintenance2
    // DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR = DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
    // // Provided by VK_KHR_fragment_shading_rate
    // FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR = SHADING_RATE_OPTIMAL_NV,
    // // Provided by VK_KHR_separate_depth_stencil_layouts
    // DEPTH_ATTACHMENT_OPTIMAL_KHR = DEPTH_ATTACHMENT_OPTIMAL,
    // // Provided by VK_KHR_separate_depth_stencil_layouts
    // DEPTH_READ_ONLY_OPTIMAL_KHR = DEPTH_READ_ONLY_OPTIMAL,
    // // Provided by VK_KHR_separate_depth_stencil_layouts
    // STENCIL_ATTACHMENT_OPTIMAL_KHR = STENCIL_ATTACHMENT_OPTIMAL,
    // // Provided by VK_KHR_separate_depth_stencil_layouts
    // STENCIL_READ_ONLY_OPTIMAL_KHR = STENCIL_READ_ONLY_OPTIMAL,
};
constexpr ImageLayoutMask get_all_layouts_mask() {
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

constexpr uint32_t get_image_layout_count() {
    uint32_t ret = 0;
    constexpr ImageLayoutMask allMask = get_all_layouts_mask();
    for (uint32_t i = 0; i < sizeof(drv::ImageLayoutMask) * 8; ++i)
        if ((1 << i) & allMask)
            ret++;
    return ret;
}
constexpr ImageLayout get_image_layout(uint32_t index) {
    return static_cast<ImageLayout>(1 << index);
}

constexpr uint32_t get_image_layout_index(ImageLayout layout) {
    for (uint32_t i = 0; i < get_image_layout_count(); ++i)
        if (get_image_layout(i) == layout)
            return i;
    return get_image_layout_count();
}

struct Offset2D
{
    int32_t x;
    int32_t y;
    bool operator==(const Offset2D& rhs) const { return x == rhs.x && y == rhs.y; }
};

struct Offset3D
{
    int32_t x;
    int32_t y;
    int32_t d;
    bool operator==(const Offset3D& rhs) const { return x == rhs.x && y == rhs.y && d == rhs.d; }
};

struct Extent2D
{
    uint32_t width;
    uint32_t height;
    bool operator==(const Extent2D& rhs) const {
        return width == rhs.width && height == rhs.height;
    }
    bool operator!=(const Extent2D& rhs) const { return !(*this == rhs); }
};

struct Rect2D
{
    Offset2D offset;
    Extent2D extent;
    bool operator==(const Rect2D& rhs) const {
        return offset == rhs.offset && extent == rhs.extent;
    }
};

struct Extent3D
{
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

enum class SampleCount : uint32_t
{
    SAMPLE_COUNT_1 = 0x00000001,
    SAMPLE_COUNT_2 = 0x00000002,
    SAMPLE_COUNT_4 = 0x00000004,
    SAMPLE_COUNT_8 = 0x00000008,
    SAMPLE_COUNT_16 = 0x00000010,
    SAMPLE_COUNT_32 = 0x00000020,
    SAMPLE_COUNT_64 = 0x00000040,
};

struct ClearRect
{
    Rect2D rect;
    uint32_t baseLayer;
    uint32_t layerCount;
};

using ImageAspectBitType = uint32_t;
enum AspectFlagBits : ImageAspectBitType
{
    COLOR_BIT = 0x00000001,
    DEPTH_BIT = 0x00000002,
    STENCIL_BIT = 0x00000004,
    METADATA_BIT = 0x00000008,
    // TODO if more are enabled, check the ranges
    // larger numbers cannot be used with UsedAspectMask (this type is for aspect id, not aspect value)
    // Provided by VK_VERSION_1_1
    // PLANE_0_BIT = 0x00000010,
    // // Provided by VK_VERSION_1_1
    // PLANE_1_BIT = 0x00000020,
    // // Provided by VK_VERSION_1_1
    // PLANE_2_BIT = 0x00000040,
    // // Provided by VK_EXT_image_drm_format_modifier
    // MEMORY_PLANE_0_BIT_EXT = 0x00000080,
    // // Provided by VK_EXT_image_drm_format_modifier
    // MEMORY_PLANE_1_BIT_EXT = 0x00000100,
    // // Provided by VK_EXT_image_drm_format_modifier
    // MEMORY_PLANE_2_BIT_EXT = 0x00000200,
    // // Provided by VK_EXT_image_drm_format_modifier
    // MEMORY_PLANE_3_BIT_EXT = 0x00000400,
    // // Provided by VK_KHR_sampler_ycbcr_conversion
    // PLANE_0_BIT_KHR = PLANE_0_BIT,
    // // Provided by VK_KHR_sampler_ycbcr_conversion
    // PLANE_1_BIT_KHR = PLANE_1_BIT,
    // // Provided by VK_KHR_sampler_ycbcr_conversion
    // PLANE_2_BIT_KHR = PLANE_2_BIT,
};
static constexpr ImageAspectBitType ALL_ASPECTS =
  COLOR_BIT | DEPTH_BIT | STENCIL_BIT | METADATA_BIT;
static constexpr ImageAspectBitType ASPECTS_COUNT = 4;
static_assert(ALL_ASPECTS + 1 == 1 << ASPECTS_COUNT);

static constexpr uint32_t get_aspect_id(AspectFlagBits aspect) {
    switch (aspect) {
        case COLOR_BIT:
            return 0;
        case DEPTH_BIT:
            return 1;
        case STENCIL_BIT:
            return 2;
        case METADATA_BIT:
            return 3;
    }
}
static constexpr AspectFlagBits get_aspect_by_id(uint32_t id) {
    return static_cast<AspectFlagBits>(1 << id);
}
static constexpr uint32_t aspect_count(ImageAspectBitType bits) {
    uint32_t ret = 0;
    for (uint32_t i = 1; i <= bits; i <<= 1)
        if (bits & i)
            ret++;
    return ret;
}
static_assert(get_aspect_by_id(get_aspect_id(COLOR_BIT)) == COLOR_BIT);
static_assert(get_aspect_by_id(get_aspect_id(DEPTH_BIT)) == DEPTH_BIT);
static_assert(get_aspect_by_id(get_aspect_id(STENCIL_BIT)) == STENCIL_BIT);
static_assert(get_aspect_by_id(get_aspect_id(METADATA_BIT)) == METADATA_BIT);

struct BufferSubresourceRange
{
    DeviceSize offset;
    DeviceSize size;
};

struct ImageSubresourceRange
{
    static constexpr uint32_t REMAINING_MIP_LEVELS = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t REMAINING_ARRAY_LAYERS = std::numeric_limits<uint32_t>::max();
    ImageAspectBitType aspectMask;
    uint32_t baseMipLevel;
    uint32_t levelCount;
    uint32_t baseArrayLayer;
    uint32_t layerCount;

    template <typename F>
    void traverse(uint32_t imageLayers, uint32_t imageLevels, F&& f) const {
        uint32_t layers =
          layerCount == REMAINING_ARRAY_LAYERS ? imageLayers - baseArrayLayer : layerCount;
        uint32_t levels =
          levelCount == REMAINING_MIP_LEVELS ? imageLevels - baseMipLevel : levelCount;
        for (uint32_t layer = baseArrayLayer; layer < layers; ++layer)
            for (uint32_t mip = baseMipLevel; mip < levels; ++mip)
                for (uint32_t aspect = 0; aspect < ASPECTS_COUNT; ++aspect)
                    if (aspectMask & get_aspect_by_id(aspect))
                        f(layer, mip, get_aspect_by_id(aspect));
    }
    bool has(uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) const;
};

struct ImageSubresourceSet
{
    static constexpr uint32_t MAX_MIP_LEVELS = 16;
    // static constexpr uint32_t MAX_ARRAY_SIZE = 16;
    using MipBit = uint16_t;
    // using UsedLayerMap = uint16_t;
    using UsedAspectMap = uint8_t;
    // UsedLayerMap usedLayers = 0;
    UsedAspectMap usedAspects = 0;
    struct PerLayerData
    {
        MipBit aspects[ASPECTS_COUNT] = {0};
        MipBit& operator[](size_t i) { return aspects[i]; }
        const MipBit& operator[](size_t i) const { return aspects[i]; }
        operator bool() const {
            for (uint32_t i = 0; i < ASPECTS_COUNT; ++i)
                if (aspects[i])
                    return true;
            return false;
        }
    };
    using MipBits = FixedArray<PerLayerData, 2>;
    MipBits mipBits;
    // MipBit mipBits[MAX_ARRAY_SIZE][ASPECTS_COUNT] = {{0}};
    static_assert(MAX_MIP_LEVELS <= sizeof(MipBit) * 8);
    // static_assert(MAX_ARRAY_SIZE <= sizeof(UsedLayerMap) * 8);
    static_assert(ASPECTS_COUNT <= sizeof(UsedAspectMap) * 8);
    explicit ImageSubresourceSet(size_t layerCount);
    void set0();
    void set(uint32_t baseLayer, uint32_t numLayers, uint32_t baseMip, uint32_t numMips,
             ImageAspectBitType aspect);
    void set(const ImageSubresourceRange& range, uint32_t imageLayers, uint32_t imageMips);

    void add(uint32_t layer, uint32_t mip, AspectFlagBits aspect);
    bool has(uint32_t layer, uint32_t mip, AspectFlagBits aspect) const;

    bool overlap(const ImageSubresourceSet& b) const;
    bool operator==(const ImageSubresourceSet& b) const;
    void merge(const ImageSubresourceSet& b);
    uint32_t getLayerCount() const;
    MipBit getMaxMipMask() const;
    UsedAspectMap getUsedAspects() const;
    bool isAspectMaskConstant() const;
    bool isLayerUsed(uint32_t layer) const;
    MipBit getMips(uint32_t layer, AspectFlagBits aspect) const;
    template <typename F>
    void traverse(F&& f) const {
        // if (!usedLayers)
        //     return;
        for (uint32_t i = 0; i < mipBits.size() /*&& (usedLayers >> i)*/; ++i) {
            // if (!(usedLayers & (1 << i)))
            //     continue;
            for (uint32_t j = 0; j < ASPECTS_COUNT && (usedAspects >> j); ++j) {
                if (!(usedAspects & (1 << j)))
                    continue;
                const MipBit& currentMipBits = mipBits[i][j];
                for (uint32_t mip = 0; mip < MAX_MIP_LEVELS && (currentMipBits >> mip); ++mip)
                    if ((1 << mip) & currentMipBits)
                        f(i, mip, get_aspect_by_id(j));
            }
        }
    }
};

struct ImageSubresourceLayers
{
    ImageAspectBitType aspectMask;
    uint32_t mipLevel;
    uint32_t baseArrayLayer;
    uint32_t layerCount;
};

struct ImageBlit
{
    ImageSubresourceLayers srcSubresource;
    Offset3D srcOffsets[2];
    ImageSubresourceLayers dstSubresource;
    Offset3D dstOffsets[2];
};

struct ImageCopyRegion
{
    ImageSubresourceLayers srcSubresource;
    Offset3D srcOffset;
    ImageSubresourceLayers dstSubresource;
    Offset3D dstOffset;
    Extent3D extent;
};

struct ImageCreateInfo
{
    ImageId imageId;
    // flags?
    enum Type
    {
        TYPE_1D = 0,
        TYPE_2D = 1,
        TYPE_3D = 2,
    } type;
    ImageFormat format;
    Extent3D extent;
    uint32_t mipLevels;
    uint32_t arrayLayers;
    SampleCount sampleCount;
    enum Tiling
    {
        TILING_OPTIMAL = 0,
        TILING_LINEAR = 1
    } tiling = TILING_OPTIMAL;
    using UsageType = unsigned int;
    enum UsageFlagBits : UsageType
    {
        TRANSFER_SRC_BIT = 0x00000001,
        TRANSFER_DST_BIT = 0x00000002,
        SAMPLED_BIT = 0x00000004,
        STORAGE_BIT = 0x00000008,
        COLOR_ATTACHMENT_BIT = 0x00000010,
        DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000020,
        TRANSIENT_ATTACHMENT_BIT = 0x00000040,
        INPUT_ATTACHMENT_BIT = 0x00000080,
        //   // Provided by VK_NV_shading_rate_image
        //     VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV = 0x00000100,
        //   // Provided by VK_EXT_fragment_density_map
        //     VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT = 0x00000200,
        //   // Provided by VK_KHR_fragment_shading_rate
        //     VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR = VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV,
    };
    UsageType usage;
    SharingType sharingType = SharingType::EXCLUSIVE;
    unsigned int familyCount = 0;
    QueueFamilyPtr* families = nullptr;
    ImageLayout initialLayout = ImageLayout::UNDEFINED;
};

struct TextureInfo
{
    const drv::ImageId* imageId;
    Extent3D extent;
    uint32_t numMips;
    uint32_t arraySize;
    ImageFormat format;
    SampleCount samples;
    ImageAspectBitType aspects;
    ImageCreateInfo::Type type;
    ImageSubresourceRange getSubresourceRange() const;
};

struct BufferInfo
{
    DeviceSize size;
    BufferSubresourceRange getSubresourceRange() const;
};

struct ImageViewCreateInfo
{
    ImagePtr image;
    enum Type
    {
        TYPE_1D = 0,
        TYPE_2D = 1,
        TYPE_3D = 2,
        TYPE_CUBE = 3,
        TYPE_1D_ARRAY = 4,
        TYPE_2D_ARRAY = 5,
        TYPE_CUBE_ARRAY = 6,
    } type;
    ImageFormat format;
    enum class ComponentSwizzle
    {
        IDENTITY = 0,
        ZERO = 1,
        ONE = 2,
        R = 3,
        G = 4,
        B = 5,
        A = 6,
    };
    struct ComponentMapping
    {
        ComponentSwizzle r;
        ComponentSwizzle g;
        ComponentSwizzle b;
        ComponentSwizzle a;
    } components;
    ImageSubresourceRange subresourceRange;
};

struct ClearColorValue
{
    union Value
    {
        float float32[4];
        int32_t int32[4];
        uint32_t uint32[4];
    } value;
    ClearColorValue() = default;
    ClearColorValue(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
        value.uint32[0] = r;
        value.uint32[1] = g;
        value.uint32[2] = b;
        value.uint32[3] = a;
    }
    ClearColorValue(int32_t r, int32_t g, int32_t b, int32_t a) {
        value.int32[0] = r;
        value.int32[1] = g;
        value.int32[2] = b;
        value.int32[3] = a;
    }
    ClearColorValue(float r, float g, float b, float a) {
        value.float32[0] = r;
        value.float32[1] = g;
        value.float32[2] = b;
        value.float32[3] = a;
    }
};

struct ClearDepthStencilValue
{
    float depth;
    uint32_t stencil;
};

struct ClearValue
{
    enum Type
    {
        COLOR,
        DEPTH
    } type;
    union Value
    {
        ClearColorValue color;
        ClearDepthStencilValue depthStencil;
    } value;
};

using ImageResourceUsageFlag = uint64_t;
enum ImageResourceUsage : ImageResourceUsageFlag
{
    IMAGE_USAGE_TRANSFER_DESTINATION = 1ull << 0,
    IMAGE_USAGE_TRANSFER_SOURCE = 1ull << 1,
    IMAGE_USAGE_PRESENT = 1ull << 2,
    IMAGE_USAGE_ATTACHMENT_INPUT = 1ull << 3,
    IMAGE_USAGE_COLOR_OUTPUT_READ = 1ull << 4,
    IMAGE_USAGE_COLOR_OUTPUT_WRITE = 1ull << 5,
    IMAGE_USAGE_DEPTH_STENCIL_READ = 1ull << 6,
    IMAGE_USAGE_DEPTH_STENCIL_WRITE = 1ull << 7,
    // Increment get_image_usage_count if new usage is added
};
constexpr uint32_t get_image_usage_count() {
    return 8;
}

constexpr ImageResourceUsage get_image_usage(uint32_t index) {
    return static_cast<ImageResourceUsage>(1 << index);
}

constexpr uint32_t get_index_of_image_usage(ImageResourceUsage usage) {
    for (uint32_t i = 0; i < get_image_usage_count(); ++i)
        if (get_image_usage(i) == usage)
            return i;
    return get_image_usage_count();
}

ImageLayoutMask get_accepted_image_layouts(ImageResourceUsageFlag usages);

ImageAspectBitType get_format_aspects(ImageFormat format);

enum class ImageFilter
{
    NEAREST = 0,
    LINEAR = 1,
    // Provided by VK_IMG_filter_cubic
    // VK_FILTER_CUBIC_IMG = 1000015000,
    // Provided by VK_EXT_filter_cubic
    // VK_FILTER_CUBIC_EXT = VK_FILTER_CUBIC_IMG,
};


PipelineStages get_image_usage_stages(ImageResourceUsageFlag usages);
MemoryBarrier::AccessFlagBitType get_image_usage_accesses(ImageResourceUsageFlag usages);

}  // namespace drv

namespace std
{
template <>
struct hash<drv::Offset2D>
{
    std::size_t operator()(const drv::Offset2D& s) const noexcept {
        return std::hash<int32_t>{}(s.x) ^ std::hash<int32_t>{}(s.y);
    }
};
template <>
struct hash<drv::Offset3D>
{
    std::size_t operator()(const drv::Offset3D& s) const noexcept {
        return std::hash<int32_t>{}(s.x) ^ std::hash<int32_t>{}(s.y) ^ std::hash<int32_t>{}(s.d);
    }
};
template <>
struct hash<drv::Extent2D>
{
    std::size_t operator()(const drv::Extent2D& s) const noexcept {
        return std::hash<uint32_t>{}(s.width) ^ std::hash<uint32_t>{}(s.height);
    }
};
template <>
struct hash<drv::Rect2D>
{
    std::size_t operator()(const drv::Rect2D& s) const noexcept {
        return std::hash<drv::Offset2D>{}(s.offset) ^ std::hash<drv::Extent2D>{}(s.extent);
    }
};
}  // namespace std
