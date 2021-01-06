#pragma once

#include <cassert>
#include <memory>
#include <variant>

#include "pixel.h"
#include "texture.hpp"

class Material
{
 public:
    template <typename P>
    using TexturePtr = std::shared_ptr<Texture<P>>;
    template <typename P>
    using Channel = std::variant<TexturePtr<P>, P>;

    Material(Channel<RGBA>&& albedo_alpha);

    const Texture<RGBA>* getAlbedoAlpha() const;

 private:
    std::shared_ptr<Texture<RGBA>> albedo_alpha;

    template <typename P>
    static TexturePtr<P> channel_to_tex(Channel<P>&& channel) {
        if (std::holds_alternative<TexturePtr<P>>(channel))
            return std::get<TexturePtr<P>>(channel);
        assert(std::holds_alternative<P>(channel));
        TexturePtr<P> ret = std::make_unique<Texture<P>>(1, 1, 1);
        ret->clear(std::get<P>(channel));
        return ret;
    }
};
