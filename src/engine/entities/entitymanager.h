#pragma once

#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <framegraph.h>

#include "entity.h"

class EntityManager final : public IAutoSerializable<Entity>
{
 public:
    struct EntityType
    {
        uint64_t engineBehaviour = 0;
        uint64_t gameBehaviour = 0;
    };

    struct EntitySystemSignature
    {
        uint64_t engineBehaviour = 0;
        uint64_t gameBehaviour = 0;
        EntityType canModify;
        bool requireZOrder = false;
    };

    EntityManager(const std::string& textureFolder, std::vector<EntityType> entityTypes,
                  std::vector<EntitySystemSignature> esSignatures);

    void initFrameGraph(FrameGraph& frameGraph) const;

    Entity::EntityId addEntity(Entity&& entity);
    void removeEntity(Entity::EntityId id);
    Entity* getById(Entity::EntityId id);
    const Entity* getById(Entity::EntityId id) const;
    Entity::EntityId getByName(const std::string& name) const;
    std::string getEntityName(Entity::EntityId id) const;

    template <typename F>
    void performES(const EntitySystemSignature& signature, F&& functor);

 private:
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;

    REFLECTABLE(
        (std::vector<Entity>) entities
    )

    std::vector<EntityType> entityTypes;
    std::vector<EntitySystemSignature> esSignatures;
    Clock::time_point startTime;
    // write lock only if new entities are added (that's when references can become invalid)
    std::shared_mutex entitiesMutex;
};
