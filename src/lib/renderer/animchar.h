#pragma once

#include "drawableentity.h"

class Animchar : public DrawableEntity
{
 public:
    Animchar(Entity* parent = nullptr, const Entity::AffineTransform& localTm = {});

    void draw(const RenderContext& ctx) const override final;

 private:
};
