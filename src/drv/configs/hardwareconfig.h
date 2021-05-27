#pragma once

#include <cstdint>

#include <serializable.h>

namespace drv
{
struct DeviceLimits final : public IAutoSerializable<DeviceLimits>
{
    REFLECTABLE
    (
        (uint32_t) maxPushConstantsSize
    )

    DeviceLimits() : maxPushConstantsSize(0) {}

    // uint32_t maxPushConstantsSize = 0;

    // REFLECT()
};
}  // namespace drv
