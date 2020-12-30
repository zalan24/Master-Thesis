#pragma once

#include <mesh.h>
#include <shadermanager.h>
#include <buffer.hpp>

#include "drawableentity.h"

class Animchar : public DrawableEntity
{
 public:
    Animchar(const Mesh& mesh, Entity* parent = nullptr,
             const Entity::AffineTransform& localTm = {});
    Animchar(Mesh&& mesh, Entity* parent = nullptr, const Entity::AffineTransform& localTm = {});

    void draw(const RenderContext& ctx) const override final;

 private:
    AttributeBinder attributeBinder;
    Mesh mesh;
    std::string shaderProgram;
    Buffer<Mesh::VertexData> glBuffer;
    Buffer<Mesh::VertexIndex> glIndices;

    void uploadData();
    void bindVertexAttributes();
};
