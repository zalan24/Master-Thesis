#include "animchar.h"

#include <gltexture.h>
#include <material.h>
#include <mesh.h>

Animchar::Animchar(const Mesh& m, Entity* parent, const Entity::AffineTransform& localTm)
  : Animchar(Mesh(m), parent, localTm) {
}

Animchar::Animchar(Mesh&& m, Entity* parent, const Entity::AffineTransform& localTm)
  : DrawableEntity(parent, localTm) {
    bindVertexAttributes();
    // TODO
    // std::unique_ptr<GlMesh> glMesh = std::make_unique<GlMesh>();
    // glMesh->upload(m);
    // meshRef = ResourceManager::getSingleton()->getGlMeshPool()->add(std::move(glMesh));
    assert(false);
    checkError();
}

void Animchar::bindVertexAttributes() {
    attributeBinder.addAttribute(&Mesh::VertexData::position, "vPos");
    attributeBinder.addAttribute(&Mesh::VertexData::normal, "vNorm", true);
    attributeBinder.addAttribute(&Mesh::VertexData::color, "vColor");
    attributeBinder.addAttribute(&Mesh::VertexData::texcoord, "vTex");
    checkError();
}

void Animchar::beforedraw(const RenderContext&) {
    if (nodeStates.size() != meshRef.getRes<GlMesh>()->getNodeCount()) {
        nodeStates.resize(meshRef.getRes<GlMesh>()->getNodeCount());
        meshRef.getRes<GlMesh>()->createStates(nodeStates.data());
    }
    meshRef.getRes<GlMesh>()->updateStates(nodeStates.data());
}

void Animchar::draw(const RenderContext& ctx) const {
    if (isHidden())
        return;
    ctx.shaderManager->useProgram("animchar");
    meshRef.getRes<GlMesh>()->bind();
    attributeBinder.bind(*ctx.shaderManager);
    ctx.shaderManager->setUniform("lightColor", ctx.lightColor);
    ctx.shaderManager->setUniform("lightDir", ctx.lightDir);
    ctx.shaderManager->setUniform("ambientColor", ctx.ambientColor);
    ctx.shaderManager->setUniform("alphaClipping", alphaClipping);

    const AffineTransform modelTm = getWorldTransform();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    for (size_t i = 0; i < meshRef.getRes<GlMesh>()->getNodeCount(); ++i) {
        const GlMesh::Node& node = meshRef.getRes<GlMesh>()->getNodes()[i];
        if (node.indexCount == 0)
            continue;
        AffineTransform tm = modelTm * nodeStates[i].globTm;
        ctx.shaderManager->setUniform("PVM", ctx.pv * tm);
        ctx.shaderManager->setUniform("model", tm);
        ctx.shaderManager->bindTexture("diffuse_tex", node.diffuseRef.getRes<GlTexture>());
        glDrawElementsBaseVertex(GL_TRIANGLES, node.indexCount, GL_UNSIGNED_INT,
                                 reinterpret_cast<void*>(node.indexOffset), node.vertexOffset);
        ctx.shaderManager->unbindTexture("diffuse_tex", node.diffuseRef.getRes<GlTexture>());
    }

    meshRef.getRes<GlMesh>()->unbind();
}
