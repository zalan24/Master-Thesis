#pragma once

#include "../featuresStructures.h"

#define ENABLE_RESOURCE_PTR_VALIDATION 1
#define ENABLE_RUNTIME_STATS_GENERATION 1

namespace featureconfig
{
inline constexpr Params get_params() {
    Params ret = {};
    ret.debugLevel = DEBUGGING_EXTRA_VALIDATION;
    ret.shaderPrint = false;
    return ret;
}
static constexpr Params params = get_params();
}  // namespace featureconfig

#include "featuresValidate.h"
