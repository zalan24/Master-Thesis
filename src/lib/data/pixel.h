#pragma once

#include <cstdint>

struct alignas(16) RGBA
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
    inline void set(float fr, float fg, float fb, float fa) {
        r = static_cast<uint8_t>(fr * 255);
        g = static_cast<uint8_t>(fg * 255);
        b = static_cast<uint8_t>(fb * 255);
        a = static_cast<uint8_t>(fa * 255);
    }
};
