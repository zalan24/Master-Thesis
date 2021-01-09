#include "glmesh.h"

#include <material.h>
#include <mesh.h>

#include "gltexture.h"

GlMesh::GlMesh() : glBuffer(GL_ARRAY_BUFFER), glIndices(GL_ELEMENT_ARRAY_BUFFER) {
}

void GlMesh::updateState(NodeState* states, size_t idx) const {
    if (!states[idx].invalidTm)
        return;
    states[idx].invalidTm = false;
    if (nodes[idx].parent == idx) {
        states[idx].globTm = states[idx].localTm;
    }
    else {
        assert(nodes[idx].parent < idx);
        updateState(states, nodes[idx].parent);
        states[idx].globTm = states[nodes[idx].parent].globTm * states[idx].localTm;
    }
}

void GlMesh::updateStates(NodeState* states) const {
    for (size_t i = 0; i < getNodeCount(); ++i)
        if (states[i].invalidTm)
            updateState(states, i);
}

void GlMesh::createStates(NodeState* states) const {
    for (size_t i = 0; i < getNodeCount(); ++i) {
        states[i].localTm = nodes[i].defaultTm;
        states[i].invalidTm = true;
    }
}

void GlMesh::upload(const Mesh& mesh) {
    std::vector<Mesh::VertexData> vertices;
    std::vector<Mesh::VertexIndex> indices;
    std::unordered_map<const Mesh*, size_t> meshIndices;
    size_t currentIndex = 0;
    size_t vertexOffset = 0;
    size_t indexOffset = 0;
    mesh.traverse([&](const Mesh& m, const Mesh::TraverseData& data) {
        vertices.insert(vertices.end(), m.getVertices().begin(), m.getVertices().end());
        indices.insert(indices.end(), m.getIndices().begin(), m.getIndices().end());
        Node node;
        node.defaultTm = m.getNodeTm();
        node.indexOffset = indexOffset;
        node.vertexOffset = vertexOffset;
        node.indexCount = m.getIndices().size();
        auto itr = meshIndices.find(data.parent);
        assert(itr != meshIndices.end());
        node.parent = itr->second;
        node.diffuseRef =
          m.getMaterial() ? m.getMaterial()->getAlbedoAlpha() : GenericResourcePool::ResourceRef();
        nodes.push_back(std::move(node));
        vertexOffset += m.getVertices().size();
        indexOffset += sizeof(Mesh::VertexIndex) * m.getIndices().size();
        meshIndices[&m] = currentIndex++;
        return true;
    });
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
