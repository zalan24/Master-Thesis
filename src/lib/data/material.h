#pragma once

#include <cassert>
#include <variant>

#include "pixel.h"
#include "texture.hpp"

class Material
{
 public:
    template <typename P>
    using Channel = std::variant<Texture<P>, P>;

    Material(Channel<RGBA>&& albedo_alpha);

 private:
    Texture<RGBA> albedo_alpha;

    template <typename P>
    static Texture<P> channel_to_tex(Channel<P>&& channel) {
        if (std::holds_alternative<Texture<P>>(channel))
            return std::move(std::get<Texture<P>>(channel));
        assert(std::holds_alternative<P>(channel));
        Texture<P> ret(1, 1, 1);
        ret.clear(std::get<P>(channel));
        return ret;
    }
};
