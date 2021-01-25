#include "animchar.h"

#include <gltexture.h>
#include <mesh.h>
#include <resourcemanager.h>

std::unique_ptr<Material> Animchar::getDefaultMaterial() {
    TextureProvider::ResourceDescriptor diffuseDesc(glm::vec4(1, 1, 1, 1));
    Material::DiffuseRes diffuseRes(std::move(diffuseDesc));
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

    skeletonAttributeBinder.addAttribute(&Mesh::BoneVertex::vId_Depth, "vId_Depth");
    skeletonAttributeBinder.addAttribute(&Mesh::BoneVertex::vPos, "vPos");
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

void Animchar::renderBones(const RenderContext& ctx) const {
    ctx.shaderManager->useProgram("skeleton");
    getGlMesh()->uploadBones(glMeshState, nullptr);
    getGlMesh()->bindSkeleton();
    getGlMesh()->bindBones(glMeshState);
    glDisable(GL_DEPTH_TEST);
    skeletonAttributeBinder.bind(*ctx.shaderManager);

    const AffineTransform modelTm = getWorldTransform();
    ctx.shaderManager->setUniform("PVM", ctx.pv * modelTm);

    static const glm::vec3 colors[2] = {glm::vec3(0.9, 0.85, 0.75), glm::vec3(0.75, 0.85, 0.9)};
    ctx.shaderManager->setUniforms("colors", colors, size_t(2));

    glDrawElements(GL_LINES, getGlMesh()->getSkeletonIndexCount(), GL_UNSIGNED_INT, nullptr);

    getGlMesh()->unbindBones(glMeshState);
    getGlMesh()->unbindSkeleton();
    checkError();
    glEnable(GL_DEPTH_TEST);
}

// TODO this function is not expection safe
void Animchar::draw(const RenderContext& ctx) const {
    if (isHidden())
        return;
    std::unique_lock<std::mutex> lock(ctx.mutex);
    ctx.shaderManager->useProgram("animchar");
    getGlMesh()->bind();
    attributeBinder.bind(*ctx.shaderManager);
    ctx.shaderManager->setUniform("lightColor", ctx.lightColor);
    ctx.shaderManager->setUniform("lightDir", ctx.lightDir);
    ctx.shaderManager->setUniform("ambientColor", ctx.ambientColor);
    ctx.shaderManager->setUniform("alphaClipping", alphaClipping);
    const AffineTransform modelTm = getWorldTransform();
    ctx.shaderManager->setUniform("PVM", ctx.pv * modelTm);
    ctx.shaderManager->setUniform("model", modelTm);

    const GenericResourcePool::ResourceRef* currentDiffuse = nullptr;

    bool drawSkeleton = drawBones && glMeshState.bones.size() > 1;

    unsigned int currentStencil = ((*ctx.currentStencil)++ % ctx.maxStencil) + 1;
    if (currentStencil == ctx.maxStencil)
        glClearStencil(0);
    glStencilFunc(GL_ALWAYS, currentStencil, 0xFF);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    for (size_t i = 0; i < getGlMesh()->getSegmentCount(); ++i) {
        const GlMesh::Segment& segment = getGlMesh()->getSegments()[i];
        if (segment.indexCount == 0)
            continue;
        getGlMesh()->uploadBones(glMeshState, segment.boneOffsets.data());
        getGlMesh()->bindBones(glMeshState);

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
        getGlMesh()->unbindBones(glMeshState);
    }

    update_texture_state(ctx, "diffuse_tex", currentDiffuse, nullptr);

    getGlMesh()->unbind();

    checkError();

    if (drawSkeleton) {
        glStencilFunc(GL_EQUAL, currentStencil, 0xFF);
        renderBones(ctx);
        glStencilFunc(GL_ALWAYS, 0, 0xFF);
    }
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

glm::mat4 Animchar::getFocusOffset() const {
    const Mesh::CameraData& cameraData = getGlMesh()->getCameraData();
    if (cameraData.bones.size() == 0)
        return glm::mat4(1.f);
    glm::mat4 ret(0.f);
    for (const Mesh::CameraData::BoneInfo& bone : cameraData.bones)
        ret += getGlMesh()->getBoneWtm(glMeshState, bone.index) * bone.offset * bone.weight;
    ret /= ret[3][3];
    return ret;
}
