#pragma once

#include <functional>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

#include <serializable.h>

class Entity  // : public ISerializable
{
 public:
    using AffineTransform = glm::mat4x4;
    struct UpdateData
    {
        float time;
        float dt;
    };
    using UpdateFunctor = std::function<void(Entity* entity, const UpdateData&)>;
    using StartFunctor = std::function<void(Entity*)>;

    Entity(Entity* parent = nullptr, const AffineTransform& localTm = AffineTransform(1.f));
    virtual ~Entity();

    Entity(const Entity&) = delete;
    Entity& operator=(const Entity&) = delete;

    virtual void update(const UpdateData& data);
    virtual void start();

    void setUpdateFunctor(const UpdateFunctor& functor);
    void setUpdateFunctor(UpdateFunctor&& functor);
    void setStartFunctor(const StartFunctor& functor);
    void setStartFunctor(StartFunctor&& functor);

    AffineTransform getWorldTransform() const;
    void setWorldTransform(const AffineTransform& tm);
    AffineTransform getLocalTransform() const;
    void setLocalTransform(const AffineTransform& tm);

    const std::vector<Entity*>& getChildren() const { return children; }

    Entity* getParent() { return parent; }
    const Entity* getParent() const { return parent; }

 private:
    Entity* parent = nullptr;
    AffineTransform localTransform;
    std::vector<Entity*> children;

    void addChild(Entity* c);
    void removeChild(Entity* c);

    // These are used if the update and start methods are not overridden
    std::function<void(Entity* entity, const UpdateData&)> updateFunctor;
    std::function<void(Entity*)> startFunctor;
};
