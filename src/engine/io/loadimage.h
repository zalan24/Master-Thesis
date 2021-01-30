#pragma once

#include <regex>
#include <string>

#include <texture.hpp>

unsigned char* load_image(const std::string& filename, int& width, int& height, int& channels);
void free_image(unsigned char* img);

unsigned char* load_image(const void* data, size_t size, int& width, int& height, int& channels);

template <typename P>
inline Texture<P> load_image(const std::string& filename) {
    int width, height, channels;
    unsigned char* img = load_image(filename, width, height, channels);
    if (!img)
        throw std::runtime_error("Could not load image: " + filename);
    try {
        Texture<P> ret(static_cast<unsigned int>(width), static_cast<unsigned int>(height));
        if (static_cast<unsigned int>(channels) > P::numChannels())
            throw std::runtime_error("Image has more channels than the requested pixel type");
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                P pixel;
                for (uint8_t i = 0; i < channels; ++i)
                    pixel[i] = img[(y * width + x) * channels + i];
                ret.set(std::move(pixel), size_t(x), size_t(height - y - 1));
            }
        }
        free_image(img);
        return ret;
    }
    catch (...) {
        free_image(img);
        throw;
    }
}

template <typename P>
inline Texture<P> load_image(const void* data, size_t size) {
    int width, height, channels;
    unsigned char* img = load_image(data, size, width, height, channels);
    if (!img)
        throw std::runtime_error("Could not load image from memory");
    try {
        Texture<P> ret(static_cast<unsigned int>(width), static_cast<unsigned int>(height));
        if (static_cast<unsigned int>(channels) > P::numChannels())
            throw std::runtime_error("Image has more channels than the requested pixel type");
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                P pixel;
                for (uint8_t i = 0; i < channels; ++i)
                    pixel[i] = img[(y * width + x) * channels + i];
                ret.set(std::move(pixel), size_t(x), size_t(height - y - 1));
            }
        }
        free_image(img);
        return ret;
    }
    catch (...) {
        free_image(img);
        throw;
    }
}
