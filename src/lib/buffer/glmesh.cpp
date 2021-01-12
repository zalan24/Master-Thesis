#include "glmesh.h"

#include <material.h>
#include <mesh.h>

#include "gltexture.h"

GlMesh::GlMesh() : glBuffer(GL_ARRAY_BUFFER), glIndices(GL_ELEMENT_ARRAY_BUFFER) {
}

void GlMesh::upload(const Mesh& mesh) {
    std::vector<Mesh::VertexData> vertices;
    std::vector<Mesh::VertexIndex> indices;
    size_t vertexOffset = 0;
    size_t indexOffset = 0;
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

    bones = ;
    materials = ;

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
    state.bonesBuffer.bind();
}

void GlMesh::unbindState(const State& state) const {
    state.bonesBuffer.unbind();
}

GlMesh::State GlMesh::createState() const {
    State ret;
    ret.bonesBuffer.reset(GL_SHADER_STORAGE_BUFFER);
    ret.boneTms.resize(bones.size());
    ret.invalidBones = true;
    // TODO create bone state vector
    return ret;
}

// void GlMesh::updateNodeState(NodeState* states, size_t idx) const {
//     if (!states[idx].invalidTm)
//         return;
//     states[idx].invalidTm = false;
//     if (nodes[idx].parent == idx) {
//         states[idx].globTm = states[idx].localTm;
//     }
//     else {
//         assert(nodes[idx].parent < idx);
//         updateNodeState(states, nodes[idx].parent);
//         states[idx].globTm = states[nodes[idx].parent].globTm * states[idx].localTm;
//     }
// }

void GlMesh::updateState(State& state) const {
    // TODO update bone states

    state.invalidBones = false;
    state.bonesBuffer.upload(state.boneTms, GL_DYNAMIC_DRAW);
}

const GlMesh::Material& GlMesh::getMat(Mesh::MaterialIndex id) const {
    return materials[id];
}
