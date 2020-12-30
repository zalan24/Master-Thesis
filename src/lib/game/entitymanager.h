#pragma once

#include <chrono>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class Entity;

struct EntityQuery
{
    std::function<bool(const Entity*)> selector;
    std::function<void(Entity*)> functor;
    template <typename F1, typename F2>
    EntityQuery(F1&& _selector, F2&& _functor)
      : selector(std::forward<F1>(_selector)), functor(std::forward<F2>(_functor)) {}
};

class EntityManager
{
 public:
    static EntityManager* getSingleton();

    using EntityId = size_t;
    static constexpr EntityId INVALID_ENTITY = std::numeric_limits<EntityId>::max();
    using UpdatePriority = float;

    EntityManager();
    ~EntityManager() noexcept;

    void addEntities(std::vector<std::unique_ptr<Entity>>&& entities, UpdatePriority priority = 0);
    EntityId addEntity(std::unique_ptr<Entity>&& entity, UpdatePriority priority = 0);
    EntityId addEntity(std::unique_ptr<Entity>&& entity, const std::string& name,
                       UpdatePriority priority = 0);
    void removeEntity(const Entity* entity);
    void removeEntity(EntityId id);
    Entity* getById(EntityId id);
    const Entity* getById(EntityId id) const;
    EntityId getByName(const std::string& name) const;
    EntityId getId(const Entity* entity) const;
    std::string getEntityName(EntityId id) const;

    void performQuery(const EntityQuery& query);

    void step();
    void start();

 private:
    using Duration = std::chrono::duration<float>;
    using Clock = std::chrono::high_resolution_clock;
    static EntityManager* instance;

    struct EntityData
    {
        std::unique_ptr<Entity> entity;
        std::string name;
        UpdatePriority priority;
        bool started;
    };

    std::vector<EntityData> entities;
    std::unordered_map<std::string, EntityId> nameMap;  // contains only named entities
    std::vector<EntityId> emptyList;  // indices of removed (currenty nullptr) entities
    std::vector<EntityId> updateOrder;
    std::vector<EntityId> toStart;

    Clock::time_point startTime;
    float uptime = -1;
    bool needReset = false;

    void checkAndResetUpdateOrder();
};
