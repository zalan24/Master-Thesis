#pragma once

#include <glmesh.h>
#include <meshprovider.h>
#include <textureprovider.h>

class GlMeshProvider final : public MeshProvider
{
 public:
    GlMeshProvider(std::string data_path, ModelResourceMap model_resources,
                   TextureProvider* texProvider,
                   ResourcePool<GlMesh, MeshProvider::ResourceDescriptor>* meshPool);
    ~GlMeshProvider() override = default;

    GenericResourcePool::ResourceRef getResource(
      const ResourceDescriptor& desc) const override final;
    GenericResourcePool::ResourceRef createResource(
      const std::string& resName) const override final;
    GenericResourcePool::ResourceRef createResource(const Mesh& m) const override final;
    GenericResourcePool::ResourceRef createResource(const ModelResource& res) const override final;

 private:
    std::string dataPath;
    ModelResourceMap modelResources;
    TextureProvider* texProvider;
    ResourcePool<GlMesh, MeshProvider::ResourceDescriptor>* meshPool;
};
