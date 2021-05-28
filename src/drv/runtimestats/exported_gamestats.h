#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

// This is constant data, meant to be shipped with the exe (must be platform independent / different platforms need different files)

#include <serializable.h>
struct GameExportsNodeData final : public IAutoSerializable<GameExportsNodeData>
{
    REFLECTABLE
    (
        (std::unordered_map<std::string, std::unique_ptr<GameExportsNodeData>>) subnodes
    )

 protected:
    bool needTimeStamp() const override { return true; }
};
