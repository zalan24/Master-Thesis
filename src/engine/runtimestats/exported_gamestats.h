#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

struct GameExportsNodeData
{
    static constexpr uint32_t FILE_HEADER = 0x12545678;
    static constexpr uint32_t FILE_END = 0xEDCBA787;

    void save(std::ostream& out) const;
    void load(std::istream& in);
    std::unordered_map<std::string, std::unique_ptr<GameExportsNodeData>> subnodes;
};
