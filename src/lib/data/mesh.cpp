#include "mesh.h"

#include <algorithm>

#include <util.hpp>

#include "material.h"

Mesh::Skeleton::Skeleton() {
    Bone bone;
    bone.localTm = glm::mat4(1.f);
    bone.parent = INVALID_BONE;
    rootBone = addBone(std::move(bone));
}

Mesh::BoneIndex Mesh::Skeleton::addBone(Bone&& bone) {
    BoneIndex boneId = static_cast<BoneIndex>(bones.size());
    bones.emplace_back(std::move(bone));
    return boneId;
}

void Mesh::Skeleton::registerBone(BoneIndex boneId, const std::string& name) {
    boneMap[name] = boneId;
}

Mesh::BoneIndex Mesh::Skeleton::getBoneId(const std::string& name) const {
    auto itr = boneMap.find(name);
    if (itr == boneMap.end())
        return INVALID_BONE;
    return itr->second;
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
    MaterialIndex ret = static_cast<MaterialIndex>(materials.size());
    materials.emplace_back(std::move(mat));
    return ret;
}

Mesh::MaterialIndex Mesh::addMaterial(const Material& mat) {
    MaterialIndex ret = static_cast<MaterialIndex>(materials.size());
    materials.emplace_back(mat);
    return ret;
}

Mesh::SegmentIndex Mesh::addSegment(Mesh::Segment&& segment) {
    SegmentIndex ret = static_cast<SegmentIndex>(segments.size());
    segments.emplace_back(std::move(segment));
    return ret;
}

void Mesh::sortSegments() {
    std::sort(segments.begin(), segments.end(),
              [](const Segment& lhs, const Segment& rhs) { return lhs.mat < rhs.mat; });
}

const Material* Mesh::getMaterial(MaterialIndex mat) const {
    return &materials[mat];
}
