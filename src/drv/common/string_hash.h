#pragma once
// https://gist.github.com/Lee-R/3839813

#include <cstdint>

namespace detail
{
// FNV-1a 32bit hashing algorithm.
constexpr std::uint32_t fnv1a_32(char const* s, size_t count) {
    return ((count ? fnv1a_32(s, count - 1) : 2166136261u) ^ static_cast<uint32_t>(s[count]))
           * 16777619u;
}
}  // namespace detail

constexpr std::uint32_t operator"" _hash(char const* s, size_t count) {
    return detail::fnv1a_32(s, count);
}

namespace drv
{
using StrHash = std::uint32_t;
}  // namespace drv
