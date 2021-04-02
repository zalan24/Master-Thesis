#pragma once

#include <drv_wrappers.h>

#include <boundresource.hpp>

struct ImageViewData
{
    LogicalDevicePtr device;
    ImageViewCreateInfo info;
};

TODO;

class BoundImageView final : public BoundResource<drv::ImageView, ImageViewData>
{
 public:
 private:
};
