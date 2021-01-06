#pragma once

#include <cstdint>

struct alignas(16) RGBA
{
    uint8_t a = 0;
    uint8_t b = 0;
    uint8_t g = 0;
    uint8_t r = 0;
    RGBA() {}
    RGBA(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a) {
        r = _r;
        g = _g;
        b = _b;
        a = _a;
    }
    inline void set(float fr, float fg, float fb, float fa) {
        r = static_cast<uint8_t>(fr * 255);
        g = static_cast<uint8_t>(fg * 255);
        b = static_cast<uint8_t>(fb * 255);
        a = static_cast<uint8_t>(fa * 255);
    }
};
