#include "mesh.h"

#include <algorithm>

#include <util.hpp>

#include "material.h"

Mesh::Skeleton::Skeleton() {
    Bone bone;
    bone.localTm = glm::mat4(1.f);
    bone.offset = glm::mat4(1.f);
    bone.parent = INVALID_BONE;
    rootBone = addBone(std::move(bone));
}

Mesh::Mesh() {
}

Mesh::VertexIndex Mesh::Segment::addVertex(const VertexData& vert) {
    VertexIndex ret = safeCast<VertexIndex>(vertices.size());
    vertices.push_back(vert);
    return ret;
}

void Mesh::Segment::addFace() {
    VertexIndex ind =
      safeCast<VertexIndex>(assertReturn(vertices.size(), [](auto val) { return val >= 3; }) - 3);
    addFace(ind, ind + 1, ind + 2);
}

void Mesh::Segment::addFaceRev() {
    VertexIndex ind =
      safeCast<VertexIndex>(assertReturn(vertices.size(), [](auto val) { return val >= 3; }) - 3);
    addFace(ind + 2, ind, ind + 1);
}

void Mesh::Segment::addFace(VertexIndex p1, VertexIndex p2, VertexIndex p3) {
    indices.push_back(p1);
    indices.push_back(p2);
    indices.push_back(p3);
}

Mesh::MaterialIndex Mesh::addMaterial(Material&& mat) {
    MaterialIndex ret = materials.size();
    materials.emplace_back(std::move(mat));
    return ret;
}

Mesh::MaterialIndex Mesh::addMaterial(const Material& mat) {
    MaterialIndex ret = materials.size();
    materials.emplace_back(mat);
    return ret;
}

Mesh::SegmentIndex Mesh::addSegment(Mesh::Segment&& segment) {
    segments.emplace_back(std::move(segment));
}

void Mesh::sortSegments() {
    std::sort(segments.begin(), segments.end(),
              [](const Segment& lhs, const Segment& rhs) { return lhs.mat < rhs.mat; });
}

const Material* Mesh::getMaterial(MaterialIndex mat) const {
    return &materials[mat];
}
