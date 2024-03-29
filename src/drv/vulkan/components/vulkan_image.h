#pragma once

#include "drvvulkan.h"

#include <logger.h>

#include <vulkan/vulkan.h>
#include <drvtracking.hpp>

namespace drv_vulkan
{
struct Image
{
    drv::ImageId imageId;
    VkImage image = VK_NULL_HANDLE;
    drv::Extent3D extent;
    uint32_t arraySize = 1;
    uint32_t numMipLevels = 1;
    drv::ImageAspectBitType aspects = 0;
    bool sharedResource = true;
    drv::SampleCount sampleCount;
    drv::ImageFormat format;
    drv::ImageCreateInfo::Type type;
    bool swapchainImage = false;
    drv::DeviceMemoryPtr memoryPtr = drv::get_null_ptr<drv::DeviceMemoryPtr>();
    drv::DeviceSize offset = 0;
    drv::MemoryType memoryType;

    // This state is only valid during linear submission, not parallel recording
    drv::GlobalImageTrackingState linearTrackingState;

    Image(drv::ImageId _imageId, VkImage _image, drv::Extent3D _extent, uint32_t _arraySize,
          uint32_t _numMipLevels, drv::ImageAspectBitType _aspects, bool _sharedResource,
          drv::SampleCount _sampleCount, drv::ImageFormat _format, drv::ImageCreateInfo::Type _type,
          bool _swapchainImage)
      : imageId(std::move(_imageId)),
        image(_image),
        extent(_extent),
        arraySize(_arraySize),
        numMipLevels(_numMipLevels),
        aspects(_aspects),
        sharedResource(_sharedResource),
        sampleCount(_sampleCount),
        format(_format),
        type(_type),
        swapchainImage(_swapchainImage),
        linearTrackingState(arraySize, numMipLevels, aspects) {
        if constexpr (featureconfig::get_params().logResourcesCreations)
            LOG_DRIVER_API("[RES] Vulkan image created: %s/%d <%p>: %dx%dx%d layers: %d, mip: %d",
                           imageId.name.c_str(), imageId.subId,
                           reinterpret_cast<const void*>(image), extent.width, extent.height,
                           extent.depth, arraySize, numMipLevels);
    }
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    ~Image() {
        if constexpr (featureconfig::get_params().logResourcesCreations)
            LOG_DRIVER_API("[RES] Vulkan image destroyed: %s/%d <%p>", imageId.name.c_str(),
                           imageId.subId, reinterpret_cast<const void*>(image));
    }
};

struct ImageView
{
    drv::ImagePtr image;
    VkImageView view;
    drv::ImageFormat format;
    drv::ImageSubresourceRange subresource;
};
}  // namespace drv_vulkan
