#pragma once

#include <gltexture.h>
#include <resourcepool.h>
#include <textureprovider.h>

class GlTextureProvider final : public TextureProvider
{
 public:
    GlTextureProvider(ResourcePool<GlTexture, TextureProvider::ResourceDescriptor>* texPool);
    ~GlTextureProvider() override {}

    GenericResourcePool::ResourceRef getResource(const ResourceDescriptor& desc) const override;

    GenericResourcePool::ResourceRef createResource(const Texture<RGBA>& tex) const override;
    GenericResourcePool::ResourceRef createResource(const std::string& filename) const override;

 private:
    ResourcePool<GlTexture, TextureProvider::ResourceDescriptor>* texPool;
};
