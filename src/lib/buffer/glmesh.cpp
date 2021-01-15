#include "glmesh.h"

#include <material.h>
#include <mesh.h>

#include "gltexture.h"

GlMesh::GlMesh()
  : glVertices(GL_ARRAY_BUFFER),
    glIndices(GL_ELEMENT_ARRAY_BUFFER),
    glSkeletonVertices(GL_ARRAY_BUFFER),
    glSkeletonIndices(GL_ELEMENT_ARRAY_BUFFER) {
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
    std::vector<Mesh::BoneVertex> boneVertices;
    std::vector<Mesh::VertexIndex> boneIndices;
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
        segment.boneOffsets = s.boneOffsets;
        segments.push_back(std::move(segment));
        vertexOffset += s.vertices.size();
        indexOffset += sizeof(Mesh::VertexIndex) * s.indices.size();
    }
    materials.reserve(mesh.getMaterialCount());
    for (size_t i = 0; i < mesh.getMaterialCount(); ++i)
        materials.push_back({mesh.getMaterials()[i].getAlbedoAlpha()});

    const Mesh::Skeleton* skeleton = mesh.getSkeleton();
    bones.reserve(skeleton->getBoneCount());
    std::vector<unsigned int> boneDepth(skeleton->getBoneCount(), 0);
    std::vector<bool> leafBones(skeleton->getBoneCount(), true);
    for (size_t i = 0; i < skeleton->getBoneCount(); ++i) {
        const Mesh::Bone& bone = skeleton->getBones()[i];
        BoneInfo boneInfo;
        boneInfo.parent = bone.parent;
        boneInfo.defaultTm = bone.localTm;
        // required for updater
        assert(boneInfo.parent <= i || boneInfo.parent == Mesh::INVALID_BONE);
        bones.push_back(std::move(boneInfo));
        if (bone.parent != Mesh::INVALID_BONE && bone.parent != i) {
            leafBones[bone.parent] = false;
            boneDepth[i] = boneDepth[bone.parent] + 1;
            boneIndices.push_back(static_cast<Mesh::VertexIndex>(bones[i].parent));
            boneIndices.push_back(static_cast<Mesh::VertexIndex>(i));
        }
        Mesh::BoneVertex boneVertex;
        boneVertex.vId_Depth = glm::vec2(i, boneDepth[i]);
        boneVertex.vPos = glm::vec3(0, 0, 0);
        boneVertices.push_back(std::move(boneVertex));
    }
    for (size_t i = 0; i < skeleton->getBoneCount(); ++i) {
        if (!leafBones[i])
            continue;
        boneIndices.push_back(static_cast<Mesh::VertexIndex>(i));
        boneIndices.push_back(static_cast<Mesh::VertexIndex>(boneVertices.size()));
        Mesh::BoneVertex vert = boneVertices[i];
        vert.vPos = glm::vec3(0, 0, 1);
        vert.vId_Depth.y = vert.vId_Depth.y + 1;
        boneVertices.push_back(std::move(vert));
    }

    glVertices.upload(vertices, GL_STATIC_DRAW);
    glIndices.upload(indices, GL_STATIC_DRAW);

    glSkeletonVertices.upload(boneVertices, GL_STATIC_DRAW);
    glSkeletonIndices.upload(boneIndices, GL_STATIC_DRAW);
    boneIndexCount = boneIndices.size();
}

void GlMesh::bind() const {
    glVertices.bind();
    glIndices.bind();
}

void GlMesh::unbind() const {
    glVertices.unbind();
    glIndices.unbind();
}

void GlMesh::bindSkeleton() const {
    glSkeletonVertices.bind();
    glSkeletonIndices.bind();
}

void GlMesh::unbindSkeleton() const {
    glSkeletonVertices.bind();
    glSkeletonIndices.bind();
}

size_t GlMesh::getSegmentCount() const {
    return segments.size();
}
const GlMesh::Segment* GlMesh::getSegments() const {
    return segments.data();
}

void GlMesh::bindBones(const State& state) const {
    state.bonesBuffer.bind(state.boneBinding);
}

void GlMesh::unbindBones(const State& state) const {
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
    return ret;
}

void GlMesh::updateState(State& state) const {
    if (state.boneTms.size() != bones.size())
        state.boneTms.resize(bones.size());
    // TODO update bone states here
}

void GlMesh::uploadBones(const State& state, const glm::mat4* offsets) const {
    for (size_t i = 0; i < bones.size(); ++i) {
        if (bones[i].parent < i)
            state.boneTms[i] = state.boneTms[bones[i].parent] * state.bones[i].localTm;
        else
            state.boneTms[i] = state.bones[i].localTm;
    }
    if (offsets != nullptr)
        for (size_t i = 0; i < bones.size(); ++i)
            state.boneTms[i] = state.boneTms[i] * offsets[i];
    state.bonesBuffer.upload(state.boneTms, GL_DYNAMIC_DRAW);
}

const GlMesh::Material& GlMesh::getMat(Mesh::MaterialIndex id) const {
    return materials[id];
}
