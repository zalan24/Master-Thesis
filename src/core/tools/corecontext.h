#pragma once

#include "memory.hpp"
#include "singleton.hpp"

// TODO this should be used in drv vulkan instead of current local memory pools

class CoreContext final : public Singleton<CoreContext>
{
 public:
    struct Config
    {
        size_t tempmemSize = 1 << 20;  // 1 Mib
    };

    StackMemory tempmem;

    explicit CoreContext(const Config& config);
};

template <>
CoreContext* Singleton<CoreContext>::instance;

#define TEMPMEM (&CoreContext::getSingleton()->tempmem)
