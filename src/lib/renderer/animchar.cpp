#include "animchar.h"

Animchar::Animchar(const Mesh& m, Entity* parent, const Entity::AffineTransform& localTm)
  : DrawableEntity(parent, localTm),
    mesh(m),
    glBuffer(GL_ARRAY_BUFFER),
    glIndices(GL_ELEMENT_ARRAY_BUFFER) {
    checkError();
    bindVertexAttributes();
    uploadData();
}

Animchar::Animchar(Mesh&& m, Entity* parent, const Entity::AffineTransform& localTm)
  : DrawableEntity(parent, localTm),
    mesh(std::move(m)),
    glBuffer(GL_ARRAY_BUFFER),
    glIndices(GL_ELEMENT_ARRAY_BUFFER) {
    checkError();
    bindVertexAttributes();
    uploadData();
}

void Animchar::bindVertexAttributes() {
    attributeBinder.addAttribute(&Mesh::VertexData::position, "vPos");
    attributeBinder.addAttribute(&Mesh::VertexData::normal, "vNorm", true);
    attributeBinder.addAttribute(&Mesh::VertexData::texcoord, "vTex");
    checkError();
}

void Animchar::uploadData() {
    std::vector<Mesh::VertexData> vertices;
    std::vector<Mesh::VertexIndex> indices;
    mesh.traverse([&](Mesh& m, const glm::mat4&) {
        vertices.insert(vertices.end(), m.getVertices().begin(), m.getVertices().end());
        indices.insert(indices.end(), m.getIndices().begin(), m.getIndices().end());
        return true;
    });
    glBuffer.upload(vertices, GL_STATIC_DRAW);
    glIndices.upload(indices, GL_STATIC_DRAW);
    checkError();
}

void Animchar::draw(const RenderContext& ctx) const {
    if (isHidden())
        return;
    ctx.shaderManager->useProgram("animchar");
    glBuffer.bind();
    glIndices.bind();
    attributeBinder.bind(*ctx.shaderManager);
    ctx.shaderManager->setUniform("lightColor", ctx.lightColor);
    ctx.shaderManager->setUniform("lightDir", ctx.lightDir);
    ctx.shaderManager->setUniform("ambientColor", ctx.ambientColor);

    const AffineTransform modelTm = getWorldTransform();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    size_t vertexOffset = 0;
    size_t indexOffset = 0;
    mesh.traverse([&](const Mesh& m, const glm::mat4& nodeTm) {
        if (m.getIndices().size() > 0) {
            AffineTransform tm = modelTm * nodeTm;
            ctx.shaderManager->setUniform("PVM", ctx.pv * tm);
            ctx.shaderManager->setUniform("model", tm);
            glDrawElementsBaseVertex(GL_TRIANGLES, m.getIndices().size(), GL_UNSIGNED_INT,
                                     reinterpret_cast<void*>(indexOffset), vertexOffset);
        }
        vertexOffset += m.getVertices().size();
        indexOffset += sizeof(Mesh::VertexIndex) * m.getIndices().size();
        return true;
    });
}
