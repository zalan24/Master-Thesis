#include "drawableentity.h"

DrawableEntity::DrawableEntity(Entity* parent, const Entity::AffineTransform& localTm)
  : Entity(parent, localTm) {
}

void DrawableEntity::setHidden(bool h) {
    hidden = h;
}

bool DrawableEntity::isHidden() const {
    return hidden;
}

void DrawableEntity::gatherEntityEntries(std::vector<ISerializable::Entry>& entries) const {
    REGISTER_ENTRY(hidden, entries);
    gatherDrawableentityEntries(entries);
}
