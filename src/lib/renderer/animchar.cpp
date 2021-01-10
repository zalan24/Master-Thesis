#include "animchar.h"

#include <gltexture.h>
#include <mesh.h>
#include <resourcemanager.h>

std::unique_ptr<Material> Animchar::getDefaultMaterial() {
    const TextureProvider* texProvider = ResourceManager::getSingleton()->getTexProvider();
    TextureProvider::ResourceDescriptor diffuseDesc(glm::vec4(1, 1, 1, 1));
    Material::DiffuseRes diffuseRes(texProvider, std::move(diffuseDesc));
    return std::make_unique<Material>(std::move(diffuseRes));
}

Animchar::Animchar(MeshRes&& m, Entity* parent, const Entity::AffineTransform& localTm)
  : DrawableEntity(parent, localTm), mesh(std::move(m)) {
    bindVertexAttributes();
    fixMat();
    checkError();
}

void Animchar::fixMat() {
    if (overrideMat && !material)
        material = getDefaultMaterial();
    for (size_t i = 0; i < getGlMesh()->getNodeCount() && !material; ++i)
        if (!getGlMesh()->getNodes()[i].diffuseRef)
            material = getDefaultMaterial();
    checkError();
}

const GlMesh* Animchar::getGlMesh() const {
    return mesh.getRes().getRes<GlMesh>();
}

void Animchar::bindVertexAttributes() {
    attributeBinder.addAttribute(&Mesh::VertexData::position, "vPos");
    attributeBinder.addAttribute(&Mesh::VertexData::normal, "vNorm", true);
    attributeBinder.addAttribute(&Mesh::VertexData::color, "vColor");
    attributeBinder.addAttribute(&Mesh::VertexData::texcoord, "vTex");
    checkError();
}

void Animchar::beforedraw(const RenderContext&) {
    if (nodeStates.size() != getGlMesh()->getNodeCount()) {
        nodeStates.resize(getGlMesh()->getNodeCount());
        getGlMesh()->createStates(nodeStates.data());
    }
    getGlMesh()->updateStates(nodeStates.data());
}

void Animchar::draw(const RenderContext& ctx) const {
    if (isHidden())
        return;
    ctx.shaderManager->useProgram("animchar");
    getGlMesh()->bind();
    attributeBinder.bind(*ctx.shaderManager);
    ctx.shaderManager->setUniform("lightColor", ctx.lightColor);
    ctx.shaderManager->setUniform("lightDir", ctx.lightDir);
    ctx.shaderManager->setUniform("ambientColor", ctx.ambientColor);
    ctx.shaderManager->setUniform("alphaClipping", alphaClipping);

    const AffineTransform modelTm = getWorldTransform();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    for (size_t i = 0; i < getGlMesh()->getNodeCount(); ++i) {
        const GlMesh::Node& node = getGlMesh()->getNodes()[i];
        if (node.indexCount == 0)
            continue;
        AffineTransform tm = modelTm * nodeStates[i].globTm;
        ctx.shaderManager->setUniform("PVM", ctx.pv * tm);
        ctx.shaderManager->setUniform("model", tm);

        const GenericResourcePool::ResourceRef& diffuseRef =
          !node.diffuseRef || overrideMat ? material->getAlbedoAlpha() : node.diffuseRef;

        ctx.shaderManager->bindTexture("diffuse_tex", diffuseRef.getRes<GlTexture>());
        glDrawElementsBaseVertex(GL_TRIANGLES, node.indexCount, GL_UNSIGNED_INT,
                                 reinterpret_cast<void*>(node.indexOffset), node.vertexOffset);
        ctx.shaderManager->unbindTexture("diffuse_tex", diffuseRef.getRes<GlTexture>());
    }

    getGlMesh()->unbind();
}

void Animchar::setMaterial(std::unique_ptr<Material>&& mat, bool _overrideMat) {
    material = std::move(mat);
    overrideMat = _overrideMat;
    fixMat();
}

void Animchar::setMaterial(const std::shared_ptr<Material>& mat, bool _overrideMat) {
    material = std::move(mat);
    overrideMat = _overrideMat;
    fixMat();
}
