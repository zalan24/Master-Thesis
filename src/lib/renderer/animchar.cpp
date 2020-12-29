#include "animchar.h"

Animchar::Animchar(Entity* parent, const Entity::AffineTransform& localTm)
  : DrawableEntity(parent, localTm) {
}

void Animchar::draw(const RenderContext& ctx) const {
    // TODO
}
