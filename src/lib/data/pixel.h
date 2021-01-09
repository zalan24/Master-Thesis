#pragma once

#include <cstdint>

struct RGBA
{
    uint8_t a = 255;
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
    uint8_t& operator[](uint8_t index) {
        switch (index) {
            case 0:
                return r;
            case 1:
                return g;
            case 2:
                return b;
            case 3:
                return a;
        }
        assert(false);
        return r;
    }
    const uint8_t& operator[](uint8_t index) const {
        switch (index) {
            case 0:
                return r;
            case 1:
                return g;
            case 2:
                return b;
            case 3:
                return a;
        }
        assert(false);
        return r;
    }
    inline void set(size_t index, float value) {
        (*this)[static_cast<uint8_t>(index)] = static_cast<uint8_t>(value * 255);
    }
    static constexpr size_t numChannels() { return 4; }
};
