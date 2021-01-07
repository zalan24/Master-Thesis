#pragma once

#include <entity.h>

#include "rendercontext.h"

class DrawableEntity : public Entity
{
 public:
    DrawableEntity(Entity* parent = nullptr,
                   const Entity::AffineTransform& localTm = Entity::AffineTransform(1.f));

    void setHidden(bool hidden);
    bool isHidden() const;

    virtual void draw(const RenderContext& ctx) const = 0;
    virtual void beforedraw(const RenderContext& ctx) {}

 protected:
    void gatherEntityEntries(std::vector<ISerializable::Entry>& entries) const override final;
    virtual void gatherDrawableentityEntries(std::vector<ISerializable::Entry>&) const {}

 private:
    bool hidden = false;
};
