#pragma once

#include <memory>

#include <glmesh.h>
#include <material.h>
#include <meshprovider.h>
#include <resourcepool.h>
#include <shadermanager.h>
#include <describedresource.hpp>

#include "controllercamera.h"
#include "drawableentity.h"

class Material;
class Animchar
  : public DrawableEntity
  , public IFollowable
{
 public:
    using MeshRes = DescribedResource<MeshProvider>;

    Animchar(MeshRes&& mesh, Entity* parent = nullptr,
             const Entity::AffineTransform& localTm = Entity::AffineTransform(1.f));

    void draw(const RenderContext& ctx) const override final;
    void beforedraw(const RenderContext& ctx) override final;

    void setMaterial(const std::shared_ptr<Material>& mat, bool overrideMat);
    void setMaterial(std::unique_ptr<Material>&& mat, bool overrideMat);

    glm::mat4 getFocusOffset() const override;

 private:
    AttributeBinder attributeBinder;
    AttributeBinder skeletonAttributeBinder;
    MeshRes mesh;
    std::shared_ptr<Material> material;
    GlMesh::State glMeshState;
    float alphaClipping = 0.5;
    bool overrideMat = false;
    bool drawBones = true;

    void bindVertexAttributes();
    const GlMesh* getGlMesh() const;
    void fixMat();

    static std::unique_ptr<Material> getDefaultMaterial();
    void renderBones(const RenderContext& ctx) const;
};
