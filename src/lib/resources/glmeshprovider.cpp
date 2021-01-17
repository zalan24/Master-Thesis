#include "glmeshprovider.h"

#include <fstream>

#include <loadmesh.h>

GlMeshProvider::GlMeshProvider(std::string data_path, const std::string& model_resources_file,
                               TextureProvider* _texProvider,
                               ResourcePool<GlMesh, MeshProvider::ResourceDescriptor>* _meshPool)
  : dataPath(std::move(data_path)), texProvider(_texProvider), meshPool(_meshPool) {
    json modelResourcesJson;
    {
        std::ifstream modeResourcesIn(model_resources_file);
        if (!modeResourcesIn.is_open())
            throw std::runtime_error("Could not find model resources file: "
                                     + model_resources_file);
        modeResourcesIn >> modelResourcesJson;
    }
    ISerializable::serialize(modelResourcesJson, modelResources);
}

static void fix_mesh(Mesh& m, const std::string& axisOrder) {
    if (axisOrder.size() != 3)
        throw std::runtime_error("Invalid axis order: " + axisOrder);
    if (axisOrder == "xyz")
        return;
    Mesh::Skeleton* skeleton = m.getSkeleton();
    Mesh::Bone bone = skeleton->getBone(skeleton->getRoot());
    if (axisOrder == "xzy") {
        std::swap(bone.localTm[1], bone.localTm[2]);
    }
    else if (axisOrder == "yxz") {
        std::swap(bone.localTm[0], bone.localTm[1]);
    }
    else if (axisOrder == "zyx") {
        std::swap(bone.localTm[0], bone.localTm[2]);
    }
    else if (axisOrder == "yzx") {
        std::swap(bone.localTm[0], bone.localTm[2]);
        std::swap(bone.localTm[0], bone.localTm[1]);
    }
    else if (axisOrder == "zxy") {
        std::swap(bone.localTm[0], bone.localTm[1]);
        std::swap(bone.localTm[0], bone.localTm[2]);
    }
    else
        throw std::runtime_error("Invalid axis order: " + axisOrder);
    skeleton->setBone(skeleton->getRoot(), std::move(bone));
}

static Mesh generate_mesh(const MeshProvider::ModelResource& res) {
    if (res.file == "@cube")
        return create_cube(res.size);
    if (res.file == "@sphere")
        return create_sphere(32, 16, res.size);
    if (res.file == "@plane")
        return create_plane(glm::vec3(0, 0, 0), glm::vec3(0, 1, 0), res.size);
    throw std::runtime_error("Unkown generation type: " + res.file);
}

GenericResourcePool::ResourceRef GlMeshProvider::getResource(const ResourceDescriptor& desc) const {
    GenericResourcePool::ResourceId resId = meshPool->getDescId(desc);
    if (resId != GenericResourcePool::INVALID_RESOURCE)
        return meshPool->get(resId);
    GenericResourcePool::ResourceRef res = createResource(desc.getResourceName());
    meshPool->registerDesc(desc, res.getId());
    return res;
}

GenericResourcePool::ResourceRef GlMeshProvider::createResource(const std::string& resName) const {
    auto itr = modelResources.find(resName);
    if (itr == modelResources.end())
        throw std::runtime_error("Could not find model: " + resName);
    return createResource(itr->second);
}

GenericResourcePool::ResourceRef GlMeshProvider::createResource(const Mesh& m) const {
    std::unique_ptr<GlMesh> glMesh = std::make_unique<GlMesh>();
    glMesh->upload(std::move(m));
    return meshPool->add(std::move(glMesh));
}

GenericResourcePool::ResourceRef GlMeshProvider::createResource(const ModelResource& res) const {
    Mesh m;
    if (res.file.size() > 0 && res.file[0] == '@')
        m = generate_mesh(res);
    else
        m = load_mesh(dataPath + "/" + res.file, res, texProvider);
    fix_mesh(m, res.axisOrder);
    return createResource(m);
}
