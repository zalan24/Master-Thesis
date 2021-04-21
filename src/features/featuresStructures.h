#pragma once

namespace featureconfig
{
enum Debugging
{
    DEBUGGING_NONE = 0,
    DEBUGGING_BASIC = 1,
    DEBUGGING_VALIDATION_LAYERS = 2,
    DEBUGGING_EXTRA_VALIDATION = 3
};

struct Params
{
    Debugging debugLevel = DEBUGGING_NONE;
    bool shaderPrint = false;  // disables some validation features
};
}  // namespace featureconfig
