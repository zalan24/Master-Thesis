#pragma once

#include <memory>

#include <meshprovider.h>
#include <resourcepool.h>
#include <textureprovider.h>

// class GlTexture;
// class GlMesh;

class ResourceManager
{
 public:
    // using GlTextureRef = GenericResourcePool::ResourceRef;
    // using GlMeshRef = GenericResourcePool::ResourceRef;

    struct ResourceInfos
    {
        std::string resourceFolder;
        std::string modelResourcesJson;
    };

    static ResourceManager* getSingleton() { return instance; }

    ResourceManager(ResourceInfos resource_infos);
    ~ResourceManager();

    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;

    // auto* getGlMeshPool() { return &glMeshPool; }
    // auto* getGlTexturePool() { return &glTexturePool; }

    const TextureProvider* getTexProvider() const { return textureProvider.get(); }
    const MeshProvider* getMeshProvider() const { return meshProvider.get(); }

 private:
    static ResourceManager* instance;

    ResourceInfos resourceInfos;

    // ResourcePool<GlTexture, TextureProvider::ResourceDescriptor> glTexturePool;
    // ResourcePool<GlMesh, MeshProvider::ResourceDescriptor> glMeshPool;

    std::unique_ptr<TextureProvider> textureProvider;
    std::unique_ptr<MeshProvider> meshProvider;
};
