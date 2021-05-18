#pragma once

#include <cstdint>

#include <serializable.h>

namespace drv
{
struct DeviceLimits final : public ISerializable
{
    uint32_t maxPushConstantsSize = 0;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;
};
}  // namespace drv
