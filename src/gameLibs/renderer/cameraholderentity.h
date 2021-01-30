#pragma once

#include <entity.h>

class Renderer;

class CameraHolderEntity : public Entity
{
 public:
    CameraHolderEntity(Renderer* renderer, const Entity::AffineTransform& cameraOffset,
                       Entity* parent = nullptr,
                       const Entity::AffineTransform& localTm = Entity::AffineTransform(1.f));

    bool isActive() const { return active; }
    void activate();
    void deactivate();

    void update(const UpdateData& data) override final;

 protected:
    virtual void _activate() {}
    virtual void _deactivate() {}
    virtual void _update(const UpdateData&) {}

 private:
    Renderer* renderer;
    Entity::AffineTransform cameraOffset;
    bool active = false;
};
