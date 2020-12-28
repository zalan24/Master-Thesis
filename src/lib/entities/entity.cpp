#include "entity.h"

Entity::Entity(Entity* _parent, const AffineTransform& localTm)
  : parent(_parent), localTransform(localTm) {
    if (parent)
        parent->addChild(this);
}

Entity::~Entity() {
    if (parent)
        parent->removeChild(this);
}

void Entity::destroyChildren() {
    std::vector<Entity*> copy = children;  // cheap and easy
    for (Entity*& c : copy)
        delete c;
    assert(children.size() == 0);
    children.clear();
}

Entity::AffineTransform Entity::getWorldTransform() const {
    return parent ? parent->getWorldTransform() * localTransform : localTransform;
}

void Entity::setWorldTransform(const AffineTransform& tm) {
    if (!parent)
        localTransform = tm;
    else {
        AffineTransform parentTm = parent->getWorldTransform();
        localTransform = glm::inverse(parentTm) * tm;
    }
}

Entity::AffineTransform Entity::getLocalTransform() const {
    return localTransform;
}

void Entity::setLocalTransform(const AffineTransform& tm) {
    localTransform = tm;
}

void Entity::addChild(Entity* c) {
    assert(c->parent == this);
    children.push_back(c);
}

void Entity::removeChild(Entity* c) {
    auto itr = std::find(children.begin(), children.end(), c);
    assert(itr != children.end());
    if (itr != children.end()) {
        children.erase(itr);
        assert(c->parent == this);
        c->parent = nullptr;
    }
}
