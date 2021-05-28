#pragma once

// This is heavy stuff, meant to be collected prior to usage, not in release

#include <vector>
#include <shared_mutex>

#include <serializable.h>

struct PersistanceNodeData final : public IAutoSerializable<PersistanceNodeData>
{
    REFLECTABLE
    (
        (std::unordered_map<std::string, std::unique_ptr<PersistanceNodeData>>) subnodes
    )

    mutable std::shared_mutex mutex;

 protected:
    bool needTimeStamp() const override { return true; }
};
