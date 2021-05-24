#pragma once

#include <iostream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

// This is collected during every execution and used in the next one (in release too)

struct StatsCache
{
    static constexpr uint32_t FILE_HEADER = 0x11545678;
    static constexpr uint32_t FILE_END = 0xEFCBA787;

    void save(std::ostream& out) const;
    void load(std::istream& in);
    std::unordered_map<std::string, std::unique_ptr<StatsCache>> subnodes;

    mutable std::shared_mutex mutex;
};
