#include "entitymanager.h"

#include <algorithm>

#include <namethreads.h>

#include "entity.h"

EntityManager::EntityManager(FrameGraph* _frameGraph, const std::string& textureFolder)
  : frameGraph(_frameGraph) {
    // TODO load textures
}

Entity* EntityManager::getById(Entity::EntityId id) {
    return &entities[id];
}

const Entity* EntityManager::getById(Entity::EntityId id) const {
    return &entities[id];
}

const std::string& EntityManager::getEntityName(Entity::EntityId id) const {
    return getById(id)->name;
}

void EntityManager::initFrameGraph() const {
    for (uint32_t i = 0; i < esSignatures.size(); ++i) {
        for (uint32_t j = 0; j < i; ++j) {
            if (esSignatures[i].constSystem && esSignatures[j].constSystem)
                continue;
            bool conflict = false;
            for (const auto& t : esTemplates)
                if ((esSignatures[i].type == EntitySystemInfo::ENGINE_SYSTEM
                       ? t.second.engineBehaviour
                       : t.second.gameBehaviour)
                      & esSignatures[i].flag
                    && (esSignatures[j].type == EntitySystemInfo::ENGINE_SYSTEM
                          ? t.second.engineBehaviour
                          : t.second.gameBehaviour)
                         & esSignatures[j].flag) {
                    conflict = true;
                    break;
                }
            if (!conflict)
                continue;
            uint32_t minSyncedStageId = 0;
            for (uint32_t dstStageId = 0; dstStageId < FrameGraph::NUM_STAGES; ++dstStageId) {
                FrameGraph::Stage dstStage = FrameGraph::get_stage(dstStageId);
                if (dstStage == FrameGraph::EXECUTION_STAGE)
                    continue;
                if (!(esSignatures[i].stages & dstStage))
                    continue;
                for (uint32_t srcStageId = minSyncedStageId; srcStageId <= dstStageId;
                     ++srcStageId) {
                    FrameGraph::Stage srcStage = FrameGraph::get_stage(srcStageId);
                    if (srcStage == FrameGraph::EXECUTION_STAGE)
                        continue;
                    if (!(esSignatures[j].stages & srcStage))
                        continue;
                    minSyncedStageId = srcStageId + 1;
                    frameGraph->addDependency(
                      esSignatures[i].nodeId,
                      FrameGraph::CpuDependency{esSignatures[j].nodeId, srcStage, dstStage, 0});
                    frameGraph->addDependency(
                      esSignatures[j].nodeId,
                      FrameGraph::CpuDependency{esSignatures[i].nodeId, dstStage, srcStage, 1});
                }
            }
        }
    }
}

bool EntityManager::writeBin(std::ostream& out) const {
    std::shared_lock<std::shared_mutex> lock(entitiesMutex);
    std::vector<Entity> entityCopies;
    entityCopies.reserve(entities.size());
    for (const auto& itr : entities)
        if (itr.engineBehaviour != 0 || itr.gameBehaviour != 0)
            entityCopies.push_back(itr);
    return ISerializable::serializeBin(out, entityCopies);
}

bool EntityManager::readBin(std::istream& in) {
    clearEntities();
    std::vector<Entity> entityCopies;
    if (!ISerializable::serializeBin(in, entityCopies))
        return false;
    for (auto&& itr : entityCopies)
        addEntity(std::move(itr));
    return true;
}

void EntityManager::writeJson(json& out) const {
    std::shared_lock<std::shared_mutex> lock(entitiesMutex);
    std::vector<Entity> entityCopies;
    entityCopies.reserve(entities.size());
    for (const auto& itr : entities)
        if (itr.engineBehaviour != 0 || itr.gameBehaviour != 0)
            entityCopies.push_back(itr);
    ISerializable::serialize(out, entityCopies);
}

void EntityManager::readJson(const json& in) {
    clearEntities();
    std::vector<Entity> entityCopies;
    ISerializable::serialize(in, entityCopies);
    for (auto&& itr : entityCopies)
        addEntity(std::move(itr));
}

void EntityManager::clearEntities() {
    std::unique_lock<std::shared_mutex> lock(entitiesMutex);
    entities.clear();
}

Entity::EntityId EntityManager::addEntity(Entity&& entity) {
    auto templateItr = esTemplates.find(entity.templateName);
    drv::drv_assert(templateItr != esTemplates.end(), "Unknown entity template");
    entity.gameBehaviour = templateItr->second.gameBehaviour;
    entity.engineBehaviour = templateItr->second.engineBehaviour;
    for (Entity::EntityId i = 0; i < entities.size(); ++i) {
        Entity* itr = getById(i);
        std::unique_lock<std::shared_mutex> entityLock(itr->mutex);
        if (itr->gameBehaviour == 0 && itr->engineBehaviour == 0) {
            *itr = std::move(entity);
            return i;
        }
    }
    std::unique_lock<std::shared_mutex> lock(entitiesMutex);
    Entity::EntityId ret = Entity::EntityId(entities.size());
    entities.push_back(std::move(entity));
    return ret;
}

EntityManager::EntitySystemInfo EntityManager::addEntitySystem(std::string name,
                                                               FrameGraph::Stages stages,
                                                               EntitySystemSignature signature) {
    EntitySystemInfo info;
    info.type = signature.type;
    info.id = info.type == EntitySystemInfo::ENGINE_SYSTEM ? (numEngineEs++) : (numGameEs++);
    info.flag = 1 << info.id;
    info.stages = stages;
    info.constSystem = signature.constSystem;
    info.nodeId = frameGraph->addNode(FrameGraph::Node(std::move(name), stages));
    esSignatures.push_back(info);
    return info;
}

void EntityManager::node_loop(EntityManager* entityManager, Engine* engine, FrameGraph* frameGraph,
                              const EntitySystemInfo* info) {
    // std::unique_lock<std::mutex> nodeLock();
    FrameId frameId = 0;
    while (true) {
        for (uint32_t stageId = 0; stageId < FrameGraph::NUM_STAGES; ++stageId) {
            FrameGraph::Stage stage = FrameGraph::get_stage(stageId);
            if (stage == FrameGraph::EXECUTION_STAGE || !(info->stages & stage))
                continue;
            if (FrameGraph::NodeHandle nodeHandle =
                  frameGraph->acquireNode(info->nodeId, stage, frameId);
                nodeHandle) {
            }
            else
                return;
        }
        frameId++;
    }
}

EntityManager::~EntityManager() {
    if (!esSystems.empty()) {
        for (auto& itr : esSystems)
            itr.thr.join();
        esSystems.clear();
    }
}

void EntityManager::startFrameGraph(Engine* engine) {
    esSystems.reserve(esSignatures.size());
    for (const auto& itr : esSignatures) {
        esSystems.push_back(
          {std::thread(&EntityManager::node_loop, this, engine, frameGraph, &itr)});
        set_thread_name(&esSystems.back().thr, frameGraph->getNode(itr.nodeId)->getName().c_str());
    }
}
