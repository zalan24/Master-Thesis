#include "mesh.h"

#include <algorithm>

#include <util.hpp>

#include "material.h"

static std::unique_ptr<Material> getDefaultMaterial() {
    // glm::vec4 albedo
    TextureProvider::ResourceDescriptor diffuseDesc(glm::vec4(0, 0, 0, 1));
    Material::DiffuseRes diffuseRes(texProveder, std::move(diffuseDesc));
    return std::make_unique<Material>(std::move(diffuseRes));
}

Mesh::Mesh() {
    setMaterial(getDefaultMaterial());
}

Mesh::Mesh(const std::string& nodeName) : name(nodeName) {
    setMaterial(getDefaultMaterial());
}

Mesh::VertexIndex Mesh::addVertex(const VertexData& vert) {
    VertexIndex ret = safeCast<VertexIndex>(vertices.size());
    vertices.push_back(vert);
    return ret;
}

void Mesh::addFace() {
    VertexIndex ind =
      safeCast<VertexIndex>(assertReturn(vertices.size(), [](auto val) { return val >= 3; }) - 3);
    addFace(ind, ind + 1, ind + 2);
}

void Mesh::addFaceRev() {
    VertexIndex ind =
      safeCast<VertexIndex>(assertReturn(vertices.size(), [](auto val) { return val >= 3; }) - 3);
    addFace(ind + 2, ind, ind + 1);
}

void Mesh::addFace(VertexIndex p1, VertexIndex p2, VertexIndex p3) {
    indices.push_back(p1);
    indices.push_back(p2);
    indices.push_back(p3);
}

void Mesh::traverse(const std::function<bool(const Mesh&, const TraverseData&)>& functor,
                    const TraverseData& data) const {
    TraverseData d;
    d.tm = data.tm * nodeTransform;
    d.parent = this;
    if (functor(*this, data))
        for (const Mesh& m : children)
            m.traverse(functor, d);
}

void Mesh::traverse(const std::function<bool(Mesh&, const TraverseData&)>& functor,
                    const TraverseData& data) {
    TraverseData d;
    d.tm = data.tm * nodeTransform;
    d.parent = this;
    if (functor(*this, data))
        for (Mesh& m : children)
            m.traverse(functor, d);
}

void Mesh::setNodeTm(const glm::mat4& tm) {
    nodeTransform = tm;
}

void Mesh::addChild(const Mesh& m) {
    children.push_back(m);
}

void Mesh::addChild(Mesh&& m) {
    children.push_back(std::move(m));
}

void Mesh::setMaterial(const std::shared_ptr<Material>& mat) {
    material = mat;
}

void Mesh::setMaterial(std::shared_ptr<Material>&& mat) {
    material = std::move(mat);
}

const Material* Mesh::getMaterial() const {
    return material.get();
}
