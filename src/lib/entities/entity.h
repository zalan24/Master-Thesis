#pragma once

#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

class Entity
{
 public:
    using AffineTransform = glm::mat3x4;

    void Entity(Entity* parent);
    virtual ~Entity();

    Entity(const Entity&) = delete;
    Entity& operator=(const Entity&) = delete;

    struct UpdateData
    {
        float time;
        float dt;
    };

    virtual update(const UpdateData& data) {}

    AffineTransform getWorldTransform() const;
    void setWorldTransform(const AffineTransform& tm);
    AffineTransform getLocalTransform() const;
    void setLocalTransform(const AffineTransfrm& tm);

    void destroyChildren();

 private:
    Entity* parent = nullptr;
    AffineTransform localTransform;
    std::vector<Entity*> children;

    void addChild(Entity* c);
    void removeChild(Entity* c);
};
