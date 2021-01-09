#include "glmeshprovider.h"

#include <loadmesh.h>

GlMeshProvider::GlMeshProvider(TextureProvider* _texProvider,
                               ResourcePool<GlMesh, MeshProvider::ResourceDescriptor>* _meshPool)
  : texProvider(_texProvider), meshPool(_meshPool) {
}

GenericResourcePool::ResourceRef GlMeshProvider::getResource(const ResourceDescriptor& desc) const {
    GenericResourcePool::ResourceId resId = meshPool->getDescId(desc);
    if (resId != GenericResourcePool::INVALID_RESOURCE)
        return meshPool->get(resId);
    GenericResourcePool::ResourceRef res;
    if (desc.isFile())
        res = createResource(desc.getFilename());
    else
        res = createResource(desc.getType());
    meshPool->registerDesc(desc, res.getId());
    return res;
}

GenericResourcePool::ResourceRef GlMeshProvider::createResource(const std::string& filename) const {
    Mesh m = load_mesh(filename, texProvider);
    return createResource(m);
}

GenericResourcePool::ResourceRef GlMeshProvider::createResource(
  const ResourceDescriptor::Type type) const {
    switch (type) {
        case ResourceDescriptor::FILE:
            assert(false);
            return {};
        case ResourceDescriptor::CUBE:
            return createResource(create_cube(1));
        case ResourceDescriptor::SPHERE:
            return createResource(create_sphere(32, 16, 1));
        case ResourceDescriptor::PLANE:
            return createResource(create_plane(glm::vec3(0, 0, 0), glm::vec3(0, 1, 0), 1));
    }
    assert(false);
    return {};
}

GenericResourcePool::ResourceRef GlMeshProvider::createResource(const Mesh& m) const {
    std::unique_ptr<GlMesh> glMesh = std::make_unique<GlMesh>();
    glMesh->upload(std::move(m));
    return meshPool->add(std::move(glMesh));
}
