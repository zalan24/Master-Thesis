#pragma once

#include <chrono>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

class Entity;

class EntityManager
{
 public:
    static EntityManager* getSingleton();

    using EntityId = size_t;
    static constexpr EntityId INVALID_ENTITY = std::numeric_limits<EntityId>::max();
    using UpdatePriority = float;

    EntityManager();
    ~EntityManager() noexcept;

    EntityId addEntity(Entity* entity, UpdatePriority priority = 0);
    EntityId addEntity(Entity* entity, const std::string& name, UpdatePriority priority = 0);
    void removeEntity(const Entity* entity);
    void removeEntity(EntityId id);
    Entity* getById(EntityId id);
    const Entity* getById(EntityId id) const;
    EntityId getByName(const std::string& name) const;
    EntityId getId(const Entity* entity) const;
    std::string getEntityName(EntityId id) const;

    void updateAll();
    void start();

 private:
    using Duration = std::chrono::duration<float>;
    using Clock = std::chrono::high_resolution_clock;
    static EntityManager* instance;

    struct EntityData
    {
        Entity* entity;
        std::string name;
        UpdatePriority priority;
    };

    std::vector<EntityData> entities;
    std::unordered_map<std::string, EntityId> nameMap;  // contains only named entities
    std::vector<EntityId> emptyList;  // indices of removed (currenty nullptr) entities
    std::multimap<UpdatePriority, EntityId> specialPriorities;  // only entities with priority != 0

    Clock::time_point startTime;
    float uptime = -1;
};
