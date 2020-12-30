#pragma once

#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <serializable.h>

class Entity : public ISerializable
{
 public:
    using AffineTransform = glm::mat4x4;

    Entity(Entity* parent = nullptr, const AffineTransform& localTm = AffineTransform(1.f));
    virtual ~Entity();

    Entity(const Entity&) = delete;
    Entity& operator=(const Entity&) = delete;

    struct UpdateData
    {
        float time;
        float dt;
    };

    virtual void update(const UpdateData&) {}
    virtual void start() {}

    AffineTransform getWorldTransform() const;
    void setWorldTransform(const AffineTransform& tm);
    AffineTransform getLocalTransform() const;
    void setLocalTransform(const AffineTransform& tm);

    const std::vector<Entity*>& getChildren() const { return children; }

 protected:
    void gatherEntries(std::vector<ISerializable::Entry>& entries) const override final;
    virtual void gatherEntityEntries(std::vector<ISerializable::Entry>& entries) const {}

 private:
    Entity* parent = nullptr;
    AffineTransform localTransform;
    std::vector<Entity*> children;

    void addChild(Entity* c);
    void removeChild(Entity* c);
};
