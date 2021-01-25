#include "meshprovider.h"

MeshProvider* MeshProvider::instance = nullptr;

MeshProvider::MeshProvider() {
    assert(instance == nullptr);
    instance = this;
}

MeshProvider::~MeshProvider() {
    assert(instance == this);
    instance = nullptr;
}

MeshProvider::ResourceDescriptor::ResourceDescriptor(const std::string& res_name)
  : resName(res_name) {
}

void MeshProvider::ResourceDescriptor::writeJson(json& out) const {
    WRITE_OBJECT(resName, out);
}

void MeshProvider::ResourceDescriptor::readJson(const json& in) {
    READ_OBJECT(resName, in);
}

void MeshProvider::ModelResource::writeJson(json& out) const {
    WRITE_OBJECT(file, out);
    WRITE_OBJECT(size, out);
    WRITE_OBJECT(axisOrder, out);
    WRITE_OBJECT(excludeMeshes, out);
    WRITE_OBJECT(materialOverrides, out);
    WRITE_OBJECT(globalMaterialOverride, out);
    WRITE_OBJECT(meshSlots, out);
    WRITE_OBJECT(cameraConfig, out);
}

void MeshProvider::ModelResource::readJson(const json& in) {
    READ_OBJECT(file, in);
    READ_OBJECT_OPT(size, in, 1.f);
    READ_OBJECT_OPT(axisOrder, in, "xyz");
    READ_OBJECT_OPT(excludeMeshes, in, {});
    READ_OBJECT_OPT(materialOverrides, in, {});
    READ_OBJECT_OPT(globalMaterialOverride, in, {});
    READ_OBJECT_OPT(meshSlots, in, {});
    READ_OBJECT_OPT(cameraConfig, in, {});
}

void MeshProvider::CameraConfig::writeJson(json& out) const {
    WRITE_OBJECT(bones, out);
    WRITE_OBJECT(tm, out);
}

void MeshProvider::CameraConfig::readJson(const json& in) {
    READ_OBJECT_OPT(bones, in, {});
    READ_OBJECT_OPT(tm, in, glm::mat4(1.f));
}
