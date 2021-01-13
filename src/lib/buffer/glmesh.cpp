#include "glmesh.h"

#include <material.h>
#include <mesh.h>

#include "gltexture.h"

GlMesh::GlMesh() : glBuffer(GL_ARRAY_BUFFER), glIndices(GL_ELEMENT_ARRAY_BUFFER) {
}

void GlMesh::clear() {
    materials.clear();
    bones.clear();
    segments.clear();
}

void GlMesh::upload(const Mesh& mesh) {
    clear();
    std::vector<Mesh::VertexData> vertices;
    std::vector<Mesh::VertexIndex> indices;
    size_t vertexOffset = 0;
    size_t indexOffset = 0;
    segments.reserve(mesh.getSegmentCount());
    for (size_t i = 0; i < mesh.getSegmentCount(); ++i) {
        const Mesh::Segment& s = mesh.getSegment(i);
        vertices.insert(vertices.end(), s.vertices.begin(), s.vertices.end());
        indices.insert(indices.end(), s.indices.begin(), s.indices.end());
        Segment segment;
        segment.indexOffset = indexOffset;
        segment.vertexOffset = vertexOffset;
        segment.indexCount = s.indices.size();
        segment.matId = s.mat;
        segments.push_back(std::move(segment));
        vertexOffset += s.vertices.size();
        indexOffset += sizeof(Mesh::VertexIndex) * s.indices.size();
    }
    materials.reserve(mesh.getMaterialCount());
    for (size_t i = 0; i < mesh.getMaterialCount(); ++i)
        materials.push_back({mesh.getMaterials()[i].getAlbedoAlpha()});

    const Mesh::Skeleton* skeleton = mesh.getSkeleton();
    bones.reserve(skeleton->getBoneCount());
    for (size_t i = 0; i < skeleton->getBoneCount(); ++i) {
        const Mesh::Bone& bone = skeleton->getBones()[i];
        BoneInfo boneInfo;
        boneInfo.parent = bone.parent;
        boneInfo.defaultTm = bone.localTm;
        // required for updater
        assert(boneInfo.parent <= i || boneInfo.parent == Mesh::INVALID_BONE);
        bones.push_back(std::move(boneInfo));
    }

    glBuffer.upload(vertices, GL_STATIC_DRAW);
    glIndices.upload(indices, GL_STATIC_DRAW);
}

void GlMesh::bind() const {
    glBuffer.bind();
    glIndices.bind();
}

void GlMesh::unbind() const {
    glBuffer.unbind();
    glIndices.unbind();
}

size_t GlMesh::getSegmentCount() const {
    return segments.size();
}
const GlMesh::Segment* GlMesh::getSegments() const {
    return segments.data();
}

void GlMesh::bindState(const State& state) const {
    state.bonesBuffer.bind(state.boneBinding);
}

void GlMesh::unbindState(const State& state) const {
    state.bonesBuffer.unbind(state.boneBinding);
}

GlMesh::State GlMesh::createState(GLuint boneBinding) const {
    State ret;
    ret.boneBinding = boneBinding;
    ret.bonesBuffer.reset(GL_SHADER_STORAGE_BUFFER);
    ret.boneTms.resize(bones.size(), glm::mat4(1.f));
    ret.bones.resize(bones.size());
    std::transform(bones.begin(), bones.end(), ret.bones.begin(), [](const BoneInfo& boneInfo) {
        BoneState state;
        state.localTm = boneInfo.defaultTm;
        return state;
    });
    ret.invalidBones = true;
    return ret;
}

void GlMesh::updateState(State& state) const {
    if (state.invalidBones) {
        for (size_t i = 0; i < bones.size(); ++i) {
            if (bones[i].parent < i)
                state.boneTms[i] = state.boneTms[bones[i].parent] * state.bones[i].localTm;
            else
                state.boneTms[i] = state.bones[i].localTm;
        }
        state.bonesBuffer.upload(state.boneTms, GL_DYNAMIC_DRAW);
        state.invalidBones = false;
    }
}

const GlMesh::Material& GlMesh::getMat(Mesh::MaterialIndex id) const {
    return materials[id];
}
