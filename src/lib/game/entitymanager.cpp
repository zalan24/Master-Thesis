#include "entitymanager.h"

#include <entity.h>

EntityManager::EntityManager* EntityManager::instance = nullptr;

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
        ret = emptyList.last();
        emptyList.resize(emptyList.size() - 1);
        assert(entities[ret] == nullptr);
    }
    else {
        ret = entities.size();
        entities.resize(entities.size() + 1);
    }
    entities[ret] = entity;
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
    return ret;
}

EntityManager::EntityId EntityManager::getId(const Entity* entity) const {
    auto itr = std::find(entities.begin(), entities.end(), entity);
    if (itr == entities.end())
        return INVALID_ENTITY;
    return static_cast<EntityId>(std::distance(entities.begin(), itr));
}

void EntityManager::removeEntity(const Entity* entity) {
    removeEntity(getId(entity));
}

const std::string* EntityManager::getEntityName(EntityId id) const {
    for (auto& [name, entityId] : nameMap)
        if (entityId == id)
            return &name;
    return nullptr;
}

void EntityManager::removeEntity(EntityId id) {
    assert(id < entities.size() && entities[id] != nullptr);
    entities[id] = nullptr;
    emptyList.push_back(id);
    const std::string* name = getEntityName(id);
    if (name != nullptr)
        nameMap.erase(nameMap.find(*name));
    auto itr = std::find_if(specialPriorities.begin(), specialPriorities.end(),
                            [id](const SpecialPriority& special) { return special.id == id; });
    if (itr != specialPriorities.end())
        specialPriorities.erase(itr);
}

Entity* EntityManager::getById(EntityId id) {
    assert(id < entities.size() && entities[id] != nullptr);
    return entities[id];
}

const Entity* EntityManager::getById(EntityId id) const {
    assert(id < entities.size() && entities[id] != nullptr);
    return entities[id];
}

EntityManager::EntityId EntityManager::getByName(const std::string& name) const {
    auto itr = nameMap.find(name);
    if (itr == nameMap.end())
        return INVALID_ENTITY;
    return itr->second;
}

void EntityManager::updateAll() {
    // TODO
}
