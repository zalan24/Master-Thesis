#pragma once

#include <cstdint>
#include <limits>

using FrameId = uint64_t;
static constexpr FrameId INVALID_FRAME = std::numeric_limits<FrameId>::max();

using NodeId = uint32_t;
static constexpr NodeId INVALID_NODE = std::numeric_limits<NodeId>::max();
