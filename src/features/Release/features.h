#pragma once

#include "../featuresStructures.h"

#define ENABLE_RESOURCE_PTR_VALIDATION 0

namespace featureconfig
{
inline constexpr Params get_params() {
    Params ret = {};
    ret.debugLevel = DEBUGGING_NONE;
    ret.shaderPrint = false;
    return ret;
}
static constexpr Params params = get_params();
}  // namespace featureconfig
