#pragma once

#include <cassert>
#include <memory>
#include <variant>

#include <serializable.h>

#include "describedresource.hpp"
#include "textureprovider.h"

class GlTexture;

class Material final : public ISerializable
{
 public:
    using DiffuseRes = DescribedResource<TextureProvider>;
    using DiffuseTexRef = GenericResourcePool::ResourceRef;

    template <typename D, typename R>
    using Channel = std::variant<D, R>;

    using DiffuseChannel = Channel<DiffuseRes, DiffuseTexRef>;

    Material();
    Material(DiffuseChannel&& albedo_alpha);

    const DiffuseTexRef& getAlbedoAlpha() const;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;

    operator bool() const;

 private:
    DiffuseChannel albedo_alpha;
};
