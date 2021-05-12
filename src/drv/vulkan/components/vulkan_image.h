#pragma once

#include "drvvulkan.h"

#include <vulkan/vulkan.h>
#include <drvtypes/drvtracking.hpp>

namespace drv_vulkan
{
struct Image
{
    VkImage image = VK_NULL_HANDLE;
    drv::Extent3D extent;
    uint32_t arraySize = 1;
    uint32_t numMipLevels = 1;
    drv::ImageAspectBitType aspects = 0;
    bool sharedResource = true;
    drv::SampleCount sampleCount;
    drv::ImageFormat format;
    bool swapchainImage = false;
    drv::DeviceMemoryPtr memoryPtr = nullptr;
    drv::DeviceSize offset = 0;

    // This state is only valid during linear submission, not parallel recording
    drv::ImageTrackingState linearTrackingState;
    // TODO remove
    std::vector<drv::ImageTrackingState> trackingStates;

    Image(VkImage _image, drv::Extent3D _extent, uint32_t _arraySize, uint32_t _numMipLevels,
          drv::ImageAspectBitType _aspects, bool _sharedResource, drv::SampleCount _sampleCount,
          drv::ImageFormat _format, bool _swapchainImage)
      : image(_image),
        extent(_extent),
        arraySize(_arraySize),
        numMipLevels(_numMipLevels),
        aspects(_aspects),
        sharedResource(_sharedResource),
        sampleCount(_sampleCount),
        format(_format),
        swapchainImage(_swapchainImage),
        linearTrackingState(arraySize, numMipLevels, aspects) {
        trackingStates.reserve(MAX_NUM_TRACKING_SLOTS);
        for (uint32_t i = 0; i < MAX_NUM_TRACKING_SLOTS; ++i)
            trackingStates.emplace_back(arraySize, numMipLevels, aspects);
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
