#pragma once

#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <drv_wrappers.h>

#include <framegraph.h>
#include <flexiblearray.hpp>

#include "entity.h"

class Engine;
class Physics;

class EntityManager final : public ISerializable
{
 public:
    struct EntitySystemParams
    {
        float uptime;
        float dt;
        FrameId frameId;
    };

    // return the number of added entities
    using EntitySystemCb = void (*)(EntityManager*, Engine*, FrameGraph::NodeHandle*,
                                    FrameGraph::Stage, const EntitySystemParams&, Entity*,
                                    Entity::EntityId, FlexibleArray<Entity, 4>& outEntities);
    struct EntityTemplate
    {
        uint64_t engineBehaviour = 0;
        uint64_t gameBehaviour = 0;
    };

    struct EntitySystemInfo
    {
        uint64_t flag = 0;
        uint32_t id = 0;
        enum Type
        {
            ENGINE_SYSTEM,
            GAME_SYSTEM
        } type = ENGINE_SYSTEM;
        FrameGraph::Stages stages;
        bool constSystem;
        NodeId nodeId;
        EntitySystemCb entitySystemCb;
    };

    struct EntitySystemSignature
    {
        EntitySystemInfo::Type type = EntitySystemInfo::ENGINE_SYSTEM;
        bool constSystem = false;
    };

    EntityManager(drv::PhysicalDevicePtr physicalDevice, drv::LogicalDevicePtr device,
                  FrameGraph* frameGraph, Physics* physics, const std::string& textureFolder);
    EntityManager(const EntityManager&) = delete;
    EntityManager& operator=(const EntityManager&) = delete;

    ~EntityManager();

    EntitySystemInfo addEntitySystem(std::string name, FrameGraph::Stages stages,
                                     EntitySystemSignature signature,
                                     EntitySystemCb entitySystemCb);
    void addEntityTemplate(std::string name, EntityTemplate entityTemplate);

    void initFrameGraph() const;
    void startFrameGraph(Engine* engine);

    Entity::EntityId addEntity(Entity&& entity);
    void removeEntity(Entity::EntityId id);
    Entity* getById(Entity::EntityId id);
    const Entity* getById(Entity::EntityId id) const;
    Entity::EntityId getByName(const std::string& name) const;
    const std::string& getEntityName(Entity::EntityId id) const;

    template <typename F>
    void performES(const EntitySystemInfo& system, F&& functor) {
        FlexibleArray<Entity, 4> outEntities;
        {
            std::shared_lock<std::shared_mutex> lock(entitiesMutex);
            for (Entity::EntityId id = 0; id < Entity::EntityId(entities.size()); ++id) {
                auto& entity = entities[size_t(id)];
                if (!((system.type == system.ENGINE_SYSTEM ? entity.engineBehaviour
                                                           : entity.gameBehaviour)
                      & system.flag))
                    continue;
                if (system.constSystem) {
                    std::shared_lock<std::shared_mutex> entityLock(entity.mutex);
                    functor(id, &entity, outEntities);
                }
                else {
                    std::unique_lock<std::shared_mutex> entityLock(entity.mutex);
                    functor(id, &entity, outEntities);
                }
            }
        }
        for (uint32_t i = 0; i < outEntities.size(); ++i)
            addEntity(std::move(outEntities[i]));
    }

    template <typename F>
    void performQuery(const std::string& templateName, bool constQuery, F&& functor);

    bool writeBin(std::ostream& out) const override;
    bool readBin(std::istream& in) override;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;

    void clearEntities();

    void setVelocity(Entity::EntityId entityId, const glm::vec3& velocity);
    glm::vec3 getVelocity(Entity::EntityId entityId);

    // void prepareTexture(uint32_t textureId, drv::DrvCmdBufferRecorder* recorder);
    // drv::ImagePtr getTexture(uint32_t textureId) const;

    void setFrozen(bool _frozen) { frozen = _frozen; }

 private:
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;

    drv::PhysicalDevicePtr physicalDevice;
    drv::LogicalDevicePtr device;
    FrameGraph* frameGraph;
    Physics* physics;
    bool frozen = false;

    std::deque<Entity> entities;

    struct EntitySystemData
    {
        std::thread thr;
        // std::mutex mutex;
        // std::condition_variable
    };

    static void node_loop(EntityManager* entityManager, Engine* engine, FrameGraph* frameGraph,
                          const EntitySystemInfo* info);

    std::unordered_map<std::string, EntityTemplate> esTemplates;
    // std::unordered_map<std::string, uint32_t> textureId;
    std::vector<EntitySystemInfo> esSignatures;
    std::vector<EntitySystemData> esSystems;
    uint32_t numGameEs = 0;
    uint32_t numEngineEs = 0;
    Clock::time_point startTime;
    // drv::ImageSet textures;
    // drv::ImageSet textureStager;
    // mutable std::mutex dirtyTextureMutex;
    // std::set<uint32_t> dirtyTextures;

    mutable std::shared_mutex entitiesMutex;  // used when clearing
};
