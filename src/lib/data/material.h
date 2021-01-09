#pragma once

#include <cassert>
#include <memory>
#include <variant>

#include "describedresource.hpp"
#include "textureprovider.h"

class GlTexture;

class Material
{
 public:
    using DiffuseRes = DescribedResource<TextureProvider>;
    using DiffuseTexRef = GenericResourcePool::ResourceRef;

    template <typename D, typename R>
    using Channel = std::variant<D, R>;

    using DiffuseChannel = Channel<DiffuseRes, DiffuseTexRef>;

    Material(DiffuseChannel&& albedo_alpha);

    const DiffuseTexRef& getAlbedoAlpha() const;

 private:
    DiffuseChannel albedo_alpha;
};
