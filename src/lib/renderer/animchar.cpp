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
  : DrawableEntity(parent, localTm), mesh(std::move(m)), glMeshState(getGlMesh()->createState(0)) {
    bindVertexAttributes();
    fixMat();
    checkError();
}

void Animchar::fixMat() {
    if (overrideMat && !material)
        material = getDefaultMaterial();
    for (size_t i = 0; i < getGlMesh()->getSegmentCount() && !material; ++i)
        if (getGlMesh()->getSegments()[i].matId == Mesh::INVALID_MATERIAL)
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
    attributeBinder.addAttribute(&Mesh::VertexData::boneIds, "vBoneIds");
    attributeBinder.addAttribute(&Mesh::VertexData::boneWeights, "vBoneWeights");
    checkError();
}

void Animchar::beforedraw(const RenderContext&) {
    getGlMesh()->updateState(glMeshState);
}

static void update_texture_state(const RenderContext& ctx, const std::string& name,
                                 const GenericResourcePool::ResourceRef*& current,
                                 const GenericResourcePool::ResourceRef* tex) {
    if (current != tex) {
        if (current)
            ctx.shaderManager->unbindTexture(name, current->getRes<GlTexture>());
        current = tex;
        if (current)
            ctx.shaderManager->bindTexture(name, current->getRes<GlTexture>());
    }
}

// TODO this function is not expection safe
void Animchar::draw(const RenderContext& ctx) const {
    if (isHidden())
        return;
    ctx.shaderManager->useProgram("animchar");
    getGlMesh()->bind();
    getGlMesh()->bindState(glMeshState);
    attributeBinder.bind(*ctx.shaderManager);
    ctx.shaderManager->setUniform("lightColor", ctx.lightColor);
    ctx.shaderManager->setUniform("lightDir", ctx.lightDir);
    ctx.shaderManager->setUniform("ambientColor", ctx.ambientColor);
    ctx.shaderManager->setUniform("alphaClipping", alphaClipping);
    const AffineTransform modelTm = getWorldTransform();
    ctx.shaderManager->setUniform("PVM", ctx.pv * modelTm);
    ctx.shaderManager->setUniform("model", modelTm);

    const GenericResourcePool::ResourceRef* currentDiffuse = nullptr;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    for (size_t i = 0; i < getGlMesh()->getSegmentCount(); ++i) {
        const GlMesh::Segment& segment = getGlMesh()->getSegments()[i];
        if (segment.indexCount == 0)
            continue;

        const GenericResourcePool::ResourceRef* diffuse =
          material ? &material->getAlbedoAlpha() : nullptr;
        if (segment.matId != Mesh::INVALID_MATERIAL && !overrideMat) {
            const GlMesh::Material& mat = getGlMesh()->getMat(segment.matId);
            diffuse = &mat.diffuseRef;
        }
        assert(diffuse != nullptr);
        update_texture_state(ctx, "diffuse_tex", currentDiffuse, diffuse);

        glDrawElementsBaseVertex(GL_TRIANGLES, segment.indexCount, GL_UNSIGNED_INT,
                                 reinterpret_cast<void*>(segment.indexOffset),
                                 segment.vertexOffset);
    }

    update_texture_state(ctx, "diffuse_tex", currentDiffuse, nullptr);

    getGlMesh()->unbindState(glMeshState);
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
