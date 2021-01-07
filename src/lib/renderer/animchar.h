#pragma once

#include <glmesh.h>
#include <shadermanager.h>
#include <resourcepool.hpp>

#include "drawableentity.h"

class GlMesh;
class Animchar : public DrawableEntity
{
 public:
    Animchar(const Mesh& mesh, Entity* parent = nullptr,
             const Entity::AffineTransform& localTm = Entity::AffineTransform(1.f));
    Animchar(Mesh&& mesh, Entity* parent = nullptr,
             const Entity::AffineTransform& localTm = Entity::AffineTransform(1.f));

    void draw(const RenderContext& ctx) const override final;
    void beforedraw(const RenderContext& ctx) override final;

 private:
    AttributeBinder attributeBinder;
    std::string shaderProgram;
    ResourcePool<GlMesh>::ResourceRef meshRef;
    std::vector<GlMesh::NodeState> nodeStates;
    float alphaClipping = 0.5;

    void bindVertexAttributes();
};
