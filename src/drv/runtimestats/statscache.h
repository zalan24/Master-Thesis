#pragma once

#include <iostream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

// This is collected during every execution and used in the next one (in release too)

#include <serializable.h>

struct StatsCache final : public IAutoSerializable<StatsCache>
{
    REFLECTABLE
    (
        (std::unordered_map<std::string, std::unique_ptr<StatsCache>>) subnodes
    )

    mutable std::shared_mutex mutex;

 protected:
    bool needTimeStamp() const override { return true; }
};
