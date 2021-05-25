#pragma once

#include <vector>

#include <drvtypes.h>

namespace drv
{
struct DestriptorSetLayoutInfoHolder
{
    std::vector<DescriptorSetLayoutCreateInfo::Binding> bindings;
};

}  // namespace drv
