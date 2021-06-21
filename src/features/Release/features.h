#pragma once

#include "../featuresStructures.h"

#define ENABLE_RESOURCE_PTR_VALIDATION 0
#define ENABLE_RUNTIME_STATS_GENERATION 0
#define ENABLE_RESOURCE_STACKTRACES 0
#define ENABLE_NODE_RESOURCE_VALIDATION 0

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

#include "featuresValidate.h"
