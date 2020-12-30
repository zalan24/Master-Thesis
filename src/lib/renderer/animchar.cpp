#include "animchar.h"

Animchar::Animchar(const Mesh& m, Entity* parent, const Entity::AffineTransform& localTm)
  : DrawableEntity(parent, localTm),
    mesh(m),
    glBuffer(GL_ARRAY_BUFFER),
    glIndices(GL_ELEMENT_ARRAY_BUFFER) {
    bindVertexAttributes();
}

Animchar::Animchar(Mesh&& m, Entity* parent, const Entity::AffineTransform& localTm)
  : DrawableEntity(parent, localTm),
    mesh(std::move(m)),
    glBuffer(GL_ARRAY_BUFFER),
    glIndices(GL_ELEMENT_ARRAY_BUFFER) {
    bindVertexAttributes();
}

void Animchar::bindVertexAttributes() {
    attributeBinder.addAttribute(&Mesh::VertexData::position, "vPos");
    attributeBinder.addAttribute(&Mesh::VertexData::normal, "vNorm", true);
    attributeBinder.addAttribute(&Mesh::VertexData::color, "vCol");
    // attributeBinder.addAttribute(&VertexData::texcoord, "vTex");
}

void Animchar::uploadData() {
    glBuffer.upload(mesh.getVertices(), GL_STATIC_DRAW);
    glIndices.upload(mesh.getIndices(), GL_STATIC_DRAW);
}

void Animchar::draw(const RenderContext& ctx) const {
    ctx.shaderManager->useProgram("animchar");
    glBuffer.bind();
    glIndices.bind();
    attributeBinder.bind(*ctx.shaderManager);
    ctx.shaderManager->setUniform("PVM", ctx.pv * getLocalTransform());
    ctx.shaderManager->setUniform("model", getLocalTransform());
    ctx.shaderManager->setUniform("lightColor", ctx.lightColor);
    ctx.shaderManager->setUniform("lightDir", ctx.lightDir);
    ctx.shaderManager->setUniform("ambientColor", ctx.ambientColor);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLES, mesh.getIndices().size(), GL_UNSIGNED_INT, 0);
}
