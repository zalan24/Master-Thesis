#include "mesh.h"

#include <algorithm>

#include <util.hpp>

#include "material.h"

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

void Mesh::traverse(const std::function<bool(const Mesh&, const glm::mat4&)>& functor,
                    const glm::mat4 rootTm) const {
    glm::mat4 tm = rootTm * nodeTransform;
    if (functor(*this, tm))
        for (const Mesh& m : children)
            m.traverse(functor, tm);
}

void Mesh::traverse(const std::function<bool(Mesh&, const glm::mat4&)>& functor,
                    const glm::mat4 rootTm) {
    glm::mat4 tm = rootTm * nodeTransform;
    if (functor(*this, tm))
        for (Mesh& m : children)
            m.traverse(functor, tm);
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
