#include "glmeshprovider.h"

#include <loadmesh.h>

GlMeshProvider::GlMeshProvider(TextureProvider* _texProvider,
                               ResourcePool<GlMesh, MeshProvider::ResourceDescriptor>* _meshPool)
  : texProvider(_texProvider), meshPool(_meshPool) {
}

static void fix_mesh(Mesh& m, const MeshProvider::ResourceDescriptor& desc) {
    if (desc.getFlipYZ()) {
        glm::mat4 tm = m.getNodeTm();
        std::swap(tm[1], tm[2]);
        m.setNodeTm(tm);
    }
}

static Mesh generate_mesh(MeshProvider::ResourceDescriptor::Type type) {
    switch (type) {
        case MeshProvider::ResourceDescriptor::FILE:
            assert(false);
            return {};
        case MeshProvider::ResourceDescriptor::CUBE:
            return create_cube(1);
        case MeshProvider::ResourceDescriptor::SPHERE:
            return create_sphere(32, 16, 1);
        case MeshProvider::ResourceDescriptor::PLANE:
            return create_plane(glm::vec3(0, 0, 0), glm::vec3(0, 1, 0), 1);
    }
    assert(false);
    return {};
}

GenericResourcePool::ResourceRef GlMeshProvider::getResource(const ResourceDescriptor& desc) const {
    GenericResourcePool::ResourceId resId = meshPool->getDescId(desc);
    if (resId != GenericResourcePool::INVALID_RESOURCE)
        return meshPool->get(resId);
    GenericResourcePool::ResourceRef res;
    Mesh m;
    if (desc.isFile())
        m = load_mesh(desc.getFilename(), texProvider);
    else
        m = generate_mesh(desc.getType());
    fix_mesh(m, desc);
    res = createResource(std::move(m));
    meshPool->registerDesc(desc, res.getId());
    return res;
}

GenericResourcePool::ResourceRef GlMeshProvider::createResource(const std::string& filename) const {
    Mesh m = load_mesh(filename, texProvider);
    return createResource(m);
}

GenericResourcePool::ResourceRef GlMeshProvider::createResource(
  const ResourceDescriptor::Type type) const {
    return createResource(generate_mesh(type));
}

GenericResourcePool::ResourceRef GlMeshProvider::createResource(const Mesh& m) const {
    std::unique_ptr<GlMesh> glMesh = std::make_unique<GlMesh>();
    glMesh->upload(std::move(m));
    return meshPool->add(std::move(glMesh));
}
