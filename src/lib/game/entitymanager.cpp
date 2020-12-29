#include "entitymanager.h"

#include <algorithm>

#include <entity.h>

EntityManager* EntityManager::instance = nullptr;

EntityManager* EntityManager::getSingleton() {
    assert(instance != nullptr);
    return instance;
}

EntityManager::EntityManager() {
    assert(instance == nullptr);
    instance = this;
}

EntityManager::~EntityManager() noexcept {
    assert(instance == this);
    for (EntityId id = 0; id < entities.size(); ++id)
        if (entities[id].entity != nullptr)
            removeEntity(id);
    instance = nullptr;
}

EntityManager::EntityId EntityManager::addEntity(std::unique_ptr<Entity>&& entity,
                                                 UpdatePriority priority) {
    needReset = true;
    EntityId ret = INVALID_ENTITY;
    if (emptyList.size() > 0) {
        ret = emptyList.back();
        emptyList.resize(emptyList.size() - 1);
        assert(entities[ret].entity == nullptr);
    }
    else {
        ret = entities.size();
        entities.resize(entities.size() + 1);
    }
    entities[ret] = {std::move(entity), "", priority, false};
    toStart.push_back(ret);
    return ret;
}

EntityManager::EntityId EntityManager::addEntity(std::unique_ptr<Entity>&& entity,
                                                 const std::string& name, UpdatePriority priority) {
    if (nameMap.find(name) != nameMap.end())
        throw std::runtime_error("An entity as already added with name: " + name);
    EntityId ret = addEntity(std::move(entity), priority);
    assert(ret != INVALID_ENTITY);
    if (ret == INVALID_ENTITY)
        return ret;
    nameMap[name] = ret;
    entities[ret].name = name;
    return ret;
}

EntityManager::EntityId EntityManager::getId(const Entity* entity) const {
    auto itr = std::find_if(entities.begin(), entities.end(), [entity](const EntityData& data) {
        return data.entity.get() == entity;
    });
    if (itr == entities.end())
        return INVALID_ENTITY;
    return static_cast<EntityId>(std::distance(entities.begin(), itr));
}

void EntityManager::removeEntity(const Entity* entity) {
    removeEntity(getId(entity));
}

std::string EntityManager::getEntityName(EntityId id) const {
    assert(id < entities.size() && entities[id].entity != nullptr);
    return entities[id].name;
}

void EntityManager::removeEntity(EntityId id) {
    needReset = true;
    assert(id < entities.size() && entities[id].entity != nullptr);
    assert(entities[id].started);
    for (Entity* entity : entities[id].entity->getChildren())
        removeEntity(entity);
    std::string name = getEntityName(id);
    if (name != "")
        nameMap.erase(nameMap.find(name));
    entities[id].entity.reset();
    emptyList.push_back(id);
}

Entity* EntityManager::getById(EntityId id) {
    assert(id < entities.size() && entities[id].entity != nullptr);
    return entities[id].entity.get();
}

const Entity* EntityManager::getById(EntityId id) const {
    assert(id < entities.size() && entities[id].entity != nullptr);
    return entities[id].entity.get();
}

EntityManager::EntityId EntityManager::getByName(const std::string& name) const {
    auto itr = nameMap.find(name);
    if (itr == nameMap.end())
        return INVALID_ENTITY;
    return itr->second;
}

void EntityManager::start() {
    uptime = 0;
    startTime = Clock::now();
}

void EntityManager::step() {
    // Start
    if (toStart.size() > 0) {
        std::vector<EntityId> toStartEntities = toStart;
        toStart.clear();
        for (EntityId id : toStartEntities) {
            if (entities[id].entity == nullptr)
                continue;
            if (!entities[id].started) {
                entities[id].entity->start();
                entities[id].started = true;
            }
        }
    }

    // Update
    assert(uptime >= 0);
    float now = std::chrono::duration_cast<Duration>(Clock::now() - startTime).count();
    float dt = now - uptime;
    uptime = now;
    Entity::UpdateData updateData;
    updateData.dt = dt;
    updateData.time = uptime;
    checkAndResetUpdateOrder();
    for (EntityId id : updateOrder)
        if (entities[id].entity != nullptr && entities[id].started)
            entities[id].entity->update(updateData);
}

void EntityManager::checkAndResetUpdateOrder() {
    if (!needReset)
        return;
    updateOrder.clear();
    updateOrder.reserve(entities.size() - emptyList.size());
    for (EntityId id = 0; id < entities.size(); ++id)
        if (entities[id].entity != nullptr)
            updateOrder.push_back(id);
    needReset = false;
}

void EntityManager::performQuery(const EntityQuery& query) {
    for (EntityId id = 0; id < entities.size(); ++id)
        if (entities[id].entity != nullptr && entities[id].started
            && query.selector(entities[id].entity.get()))
            query.functor(entities[id].entity.get());
}
