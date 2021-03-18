#pragma once

#include <cstdint>
#include <limits>

using FrameId = uint64_t;
static constexpr FrameId INVALID_FRAME = std::numeric_limits<FrameId>::max();
