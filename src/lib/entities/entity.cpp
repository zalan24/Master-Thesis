#include "entity.h"

Entity::Entity(Entity* _parent, const AffineTransform& localTm)
  : parent(_parent), localTransform(localTm) {
    if (parent)
        parent->addChild(this);
}

Entity::~Entity() {
    if (parent)
        parent->removeChild(this);
    assert(children.size() == 0);
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

void Entity::gatherEntries(std::vector<ISerializable::Entry>& entries) const {
    // TODO
    // REGISTER_ENTRY();
    gatherEntityEntries(entries);
}

void Entity::update(const UpdateData& data) {
    if (updateFunctor)
        updateFunctor(this, data);
}

void Entity::start() {
    if (startFunctor)
        startFunctor(this);
}

void Entity::setUpdateFunctor(const UpdateFunctor& functor) {
    updateFunctor = functor;
}

void Entity::setUpdateFunctor(UpdateFunctor&& functor) {
    updateFunctor = std::move(functor);
}

void Entity::setStartFunctor(const StartFunctor& functor) {
    startFunctor = functor;
}

void Entity::setStartFunctor(StartFunctor&& functor) {
    startFunctor = std::move(functor);
}
