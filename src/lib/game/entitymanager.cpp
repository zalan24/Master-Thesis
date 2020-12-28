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
    instance = nullptr;
}

EntityManager::EntityId EntityManager::addEntity(Entity* entity, UpdatePriority priority) {
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
    entities[ret] = {entity, "", priority};
    if (priority != 0)
        specialPriorities.insert({priority, ret});
    return ret;
}

EntityManager::EntityId EntityManager::addEntity(Entity* entity, const std::string& name,
                                                 UpdatePriority priority) {
    if (nameMap.find(name) != nameMap.end())
        throw std::runtime_error("An entity as already added with name: " + name);
    EntityId ret = addEntity(entity, priority);
    assert(ret != INVALID_ENTITY);
    if (ret == INVALID_ENTITY)
        return ret;
    nameMap[name] = ret;
    entities[ret].name = name;
    return ret;
}

EntityManager::EntityId EntityManager::getId(const Entity* entity) const {
    auto itr = std::find_if(entities.begin(), entities.end(),
                            [entity](const EntityData& data) { return data.entity == entity; });
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
    assert(id < entities.size() && entities[id].entity != nullptr);
    entities[id].entity = nullptr;
    emptyList.push_back(id);
    std::string name = getEntityName(id);
    if (name != "")
        nameMap.erase(nameMap.find(name));
    auto itr = std::find_if(
      specialPriorities.begin(), specialPriorities.end(),
      [id](const std::pair<UpdatePriority, EntityId>& special) { return special.second == id; });
    if (itr != specialPriorities.end())
        specialPriorities.erase(itr);
}

Entity* EntityManager::getById(EntityId id) {
    assert(id < entities.size() && entities[id].entity != nullptr);
    return entities[id].entity;
}

const Entity* EntityManager::getById(EntityId id) const {
    assert(id < entities.size() && entities[id].entity != nullptr);
    return entities[id].entity;
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

void EntityManager::updateAll() {
    assert(uptime >= 0);
    float now = std::chrono::duration_cast<Duration>(Clock::now() - startTime).count();
    float dt = now - uptime;
    uptime = now;
    Entity::UpdateData updateData;
    updateData.dt = dt;
    updateData.time = uptime;
    for (const auto& [priority, id] : specialPriorities)
        if (priority < 0)
            getById(id)->update(updateData);
        else
            break;
    for (const EntityData& data : entities)
        if (data.priority == 0)
            data.entity->update(updateData);
    for (const auto& [priority, id] : specialPriorities)
        if (priority > 0)
            getById(id)->update(updateData);
}
