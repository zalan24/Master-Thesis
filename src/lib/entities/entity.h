#pragma once

#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

class Entity
{
 public:
    using AffineTransform = glm::mat4x4;

    Entity(Entity* parent, const AffineTransform& localTm = {});
    virtual ~Entity();

    Entity(const Entity&) = delete;
    Entity& operator=(const Entity&) = delete;

    struct UpdateData
    {
        float time;
        float dt;
    };

    virtual void update(const UpdateData&) {}

    AffineTransform getWorldTransform() const;
    void setWorldTransform(const AffineTransform& tm);
    AffineTransform getLocalTransform() const;
    void setLocalTransform(const AffineTransform& tm);

    void destroyChildren();

 private:
    Entity* parent = nullptr;
    AffineTransform localTransform;
    std::vector<Entity*> children;

    void addChild(Entity* c);
    void removeChild(Entity* c);
};
