#pragma once

#include <glmesh.h>
#include <meshprovider.h>
#include <textureprovider.h>

class GlMeshProvider final : public MeshProvider
{
 public:
    GlMeshProvider(TextureProvider* texProvider,
                   ResourcePool<GlMesh, MeshProvider::ResourceDescriptor>* meshPool);
    ~GlMeshProvider() override = default;

    GenericResourcePool::ResourceRef getResource(const ResourceDescriptor& desc) const override;
    GenericResourcePool::ResourceRef createResource(const std::string& filename) const override;
    GenericResourcePool::ResourceRef createResource(
      const ResourceDescriptor::Type type) const override;
    GenericResourcePool::ResourceRef createResource(const Mesh& m) const override;

 private:
    TextureProvider* texProvider;
    ResourcePool<GlMesh, MeshProvider::ResourceDescriptor>* meshPool;
};
