#pragma once

#include <memory>

#include <glmesh.h>
#include <material.h>
#include <meshprovider.h>
#include <resourcepool.h>
#include <shadermanager.h>
#include <describedresource.hpp>

#include "drawableentity.h"

class Material;
class Animchar : public DrawableEntity
{
 public:
    using MeshRes = DescribedResource<MeshProvider>;

    Animchar(MeshRes&& mesh, Entity* parent = nullptr,
             const Entity::AffineTransform& localTm = Entity::AffineTransform(1.f));

    void draw(const RenderContext& ctx) const override final;
    void beforedraw(const RenderContext& ctx) override final;

 private:
    AttributeBinder attributeBinder;
    MeshRes mesh;
    std::unique_ptr<Material> material;
    std::vector<GlMesh::NodeState> nodeStates;
    float alphaClipping = 0.5;

    void bindVertexAttributes();
    const GlMesh* getGlMesh() const;
    static std::unique_ptr<Material> getDefaultMaterial();
};
