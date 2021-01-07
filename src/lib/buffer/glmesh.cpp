#include "glmesh.h"

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
    size_t vertexOffset = 0;
    size_t indexOffset = 0;
    mesh.traverse([&](Mesh& m, const glm::mat4&) {
        vertices.insert(vertices.end(), m.getVertices().begin(), m.getVertices().end());
        indices.insert(indices.end(), m.getIndices().begin(), m.getIndices().end());
        Node node;
        node.defaultTm = m.getNodeTm();
        node.indexOffset = indexOffset;
        node.vertexOffset = vertexOffset;
        node.indexCount = m.getIndices().size();
        node.parent = ;
        node.diffuseTex = ;
        // diffuseTex(GL_TEXTURE_2D, GL_RGBA)
        // const Material* mat = mesh.getMaterial();
        // diffuseTex.upload(mat->getAlbedoAlpha());
        nodes.push_back(node);
        vertexOffset += m.getVertices().size();
        indexOffset += sizeof(Mesh::VertexIndex) * m.getIndices().size();
        return true;
    });
    glBuffer.upload(vertices, GL_STATIC_DRAW);
    glIndices.upload(indices, GL_STATIC_DRAW);
    checkError();
}

void GlMesh::bind() const {
    glBuffer.bind();
    glIndices.bind();
}

void GlMesh::unbind() const {
    glBuffer.unbind();
    glIndices.unbind();
}
