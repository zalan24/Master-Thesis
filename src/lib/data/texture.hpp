#pragma once

#include <algorithm>
#include <vector>

template <typename P>
class Texture
{
 public:
    Texture(size_t w, size_t h, size_t d = 1)
      : width(w), height(h), depth(d), data(width * height * depth) {}

    const P& get(size_t x, size_t y, size_t z = 0) const { return data[translate(x, y, z)]; }
    void set(const P& pixel, size_t x, size_t y, size_t z = 0) const {
        data[translate(x, y, z)] = pixel;
    }
    void set(P&& pixel, size_t x, size_t y, size_t z = 0) const {
        data[translate(x, y, z)] = std::move(pixel);
    }

    void clear(const P& clear_value) { std::fill(data.begin(), data.end(), clear_value); }

    size_t getWidth() const { return width; }
    size_t getHeight() const { return height; }
    size_t getDepth() const { return depth; }
    const P* getData() const { return data.data(); }

 private:
    size_t width;
    size_t height;
    size_t depth;
    std::vector<P> data;

    size_t translate(size_t x, size_t y, size_t z) const {
        return z * width * height + y * width + x;
    }
};
