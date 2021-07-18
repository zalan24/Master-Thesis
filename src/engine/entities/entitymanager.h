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

#include "entity.h"

class Engine;

class EntityManager final : public ISerializable
{
 public:
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
        FrameGraph::NodeId nodeId;
    };

    struct EntitySystemSignature
    {
        EntitySystemInfo::Type type = EntitySystemInfo::ENGINE_SYSTEM;
        bool constSystem = false;
    };

    EntityManager(drv::PhysicalDevicePtr physicalDevice, drv::LogicalDevicePtr device,
                  FrameGraph* frameGraph, const std::string& textureFolder);
    EntityManager(const EntityManager&) = delete;
    EntityManager& operator=(const EntityManager&) = delete;

    ~EntityManager();

    EntitySystemInfo addEntitySystem(std::string name, FrameGraph::Stages stages,
                                     EntitySystemSignature signature);
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
    void performES(const EntitySystemInfo& system, F&& functor);

    template <typename F>
    void performQuery(const std::string& templateName, bool constQuery, F&& functor);

    bool writeBin(std::ostream& out) const override;
    bool readBin(std::istream& in) override;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;

    void clearEntities();

 private:
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;

    drv::PhysicalDevicePtr physicalDevice;
    drv::LogicalDevicePtr device;
    FrameGraph* frameGraph;

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
    std::unordered_map<std::string, uint32_t> textureId;
    std::vector<EntitySystemInfo> esSignatures;
    std::vector<EntitySystemData> esSystems;
    uint32_t numGameEs = 0;
    uint32_t numEngineEs = 0;
    Clock::time_point startTime;
    drv::ImageSet textures;
    drv::ImageSet textureStager;

    mutable std::shared_mutex entitiesMutex;  // used when clearing
};
