#include "meshprovider.h"

MeshProvider::~MeshProvider() {
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
    // WRITE_OBJECT(materialOverrides, out);
    WRITE_OBJECT(meshSlots, out);
}

void MeshProvider::ModelResource::readJson(const json& in) {
    READ_OBJECT(file, in);
    READ_OBJECT(size, in);
    READ_OBJECT(axisOrder, in);
    READ_OBJECT(excludeMeshes, in);
    // READ_OBJECT(materialOverrides, in);
    READ_OBJECT(meshSlots, in);
}
