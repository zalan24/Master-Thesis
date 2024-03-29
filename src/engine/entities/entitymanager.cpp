#include "entitymanager.h"

#include <algorithm>
#include <filesystem>

// #define STB_IMAGE_IMPLEMENTATION
// #include <stb_image.h>

#include <namethreads.h>

#include <framegraph.h>
#include <logger.h>
#include <physics.h>

#include "entity.h"

namespace fs = std::filesystem;

EntityManager::EntityManager(drv::PhysicalDevicePtr _physicalDevice, drv::LogicalDevicePtr _device,
                             FrameGraph* _frameGraph, Physics* _physics,
                             const std::string& /*textureFolder*/)
  : physicalDevice(_physicalDevice), device(_device), frameGraph(_frameGraph), physics(_physics) {
    // std::vector<drv::ImageSet::ImageInfo> imageInfos;
    // std::vector<drv::ImageSet::ImageInfo> imageStagerInfos;
    // std::vector<unsigned char*> imageData;
    // std::vector<int> imageChannels;
    // drv::drv_assert(fs::exists(fs::path{textureFolder}), "Textures folders does not exist");
    // for (auto& p : fs::directory_iterator(fs::path{textureFolder})) {
    //     const std::string filename = p.path().filename().string();
    //     const std::string ext = p.path().extension().string();
    //     if (ext != ".png" && ext != ".jpg" && ext != ".bmp")
    //         continue;
    //     int width, height, channels;
    //     unsigned char* image = stbi_load(p.path().string().c_str(), &width, &height, &channels, 0);
    //     drv::drv_assert(image != nullptr, "Could not load image");
    //     imageData.push_back(image);
    //     imageChannels.push_back(channels);
    //     drv::ImageSet::ImageInfo imageInfo;
    //     imageInfo.imageId = drv::ImageId(p.path().filename().string());
    //     imageInfo.type = imageInfo.TYPE_2D;
    //     imageInfo.format = drv::ImageFormat::R8G8B8A8_UNORM;
    //     imageInfo.extent = {uint32_t(width), uint32_t(height), 1};
    //     imageInfo.mipLevels = 1;
    //     imageInfo.arrayLayers = 1;
    //     imageInfo.sampleCount = drv::SampleCount::SAMPLE_COUNT_1;
    //     imageInfo.tiling = imageInfo.TILING_LINEAR;
    //     imageInfo.usage = imageInfo.TRANSFER_DST_BIT | imageInfo.TRANSFER_SRC_BIT;
    //     textureId[p.path().filename().string()] = uint32_t(imageInfos.size());
    //     dirtyTextures.insert(uint32_t(imageInfos.size()));
    //     imageInfos.push_back(imageInfo);
    //     imageStagerInfos.push_back(imageInfo);
    //     LOG_F(INFO, "Texture loaded in entity manager: %s", p.path().filename().string().c_str());
    // }
    // textures = drv::ImageSet(physicalDevice, device, std::move(imageInfos),
    //                          drv::ImageSet::PreferenceSelector(drv::MemoryType::DEVICE_LOCAL_BIT,
    //                                                            drv::MemoryType::DEVICE_LOCAL_BIT));
    // textureStager =
    //   drv::ImageSet(physicalDevice, device, std::move(imageInfos),
    //                 drv::ImageSet::PreferenceSelector(drv::MemoryType::HOST_COHERENT_BIT
    //                                                     | drv::MemoryType::HOST_CACHED_BIT
    //                                                     | drv::MemoryType::HOST_VISIBLE_BIT,
    //                                                   drv::MemoryType::HOST_VISIBLE_BIT));
    // for (uint32_t i = 0; i < imageData.size(); ++i) {
    //     drv::TextureInfo texInfo = drv::get_texture_info(textureStager.getImage(i));

    //     drv::DeviceSize offset;
    //     drv::DeviceSize size;
    //     drv::DeviceSize rowPitch;
    //     drv::DeviceSize arrayPitch;
    //     drv::DeviceSize depthPitch;
    //     drv::drv_assert(drv::get_image_memory_data(device, textureStager.getImage(i), 0, 0, offset,
    //                                                size, rowPitch, arrayPitch, depthPitch),
    //                     "Could not get image memory data");

    //     StackMemory::MemoryHandle<uint32_t> srcData(size, TEMPMEM);
    //     for (uint32_t y = 0; y < texInfo.extent.height; ++y) {
    //         for (uint32_t x = 0; x < texInfo.extent.width; ++x) {
    //             uint32_t r, g, b, a;
    //             r = g = b = 0;
    //             a = 255;
    //             uint32_t numChannels = uint32_t(imageChannels[i]);
    //             r = imageData[i][(x + texInfo.extent.width * y) * numChannels + 0];
    //             if (numChannels > 1)
    //                 g = imageData[i][(x + texInfo.extent.width * y) * numChannels + 1];
    //             if (numChannels > 2)
    //                 b = imageData[i][(x + texInfo.extent.width * y) * numChannels + 2];
    //             if (numChannels > 3)
    //                 a = imageData[i][(x + texInfo.extent.width * y) * numChannels + 3];
    //             srcData[StackMemory::size_t(x + y * rowPitch / 4)] =
    //               (a << 24) + (b << 16) + (g << 8) + r;
    //         }
    //     }
    //     TemporalResourceLockerDescriptor resourceDesc;
    //     resourceDesc.addImage(textureStager.getImage(i), 1, 0, 0, drv::AspectFlagBits::COLOR_BIT,
    //                           drv::ResourceLockerDescriptor::WRITE);
    //     auto lock = frameGraph->getResourceLocker()->lock(&resourceDesc).getLock();
    //     drv::write_image_memory(device, textureStager.getImage(i), 0, 0, lock, srcData);
    //     stbi_image_free(imageData[i]);
    // }
}

Entity* EntityManager::getById(Entity::EntityId id) {
    return &entities[size_t(id)];
}

const Entity* EntityManager::getById(Entity::EntityId id) const {
    return &entities[size_t(id)];
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
    return ISerializable::serializeBin(out, const_cast<const std::vector<Entity>&>(entityCopies));
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
    out = ISerializable::serialize(entityCopies);
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
    if (entity.templateName == "")
        return Entity::INVALID_ENTITY;
    auto templateItr = esTemplates.find(entity.templateName);
    drv::drv_assert(templateItr != esTemplates.end(), "Unknown entity template");
    entity.gameBehaviour = templateItr->second.gameBehaviour;
    entity.engineBehaviour = templateItr->second.engineBehaviour;
    // if (entity.textureName != "") {
    //     auto textureItr = textureId.find(entity.textureName);
    //     drv::drv_assert(textureItr != textureId.end(), "Could not find a texture for an entity");
    //     entity.textureId = textureItr->second;
    // }
    if (entity.parentName != "") {
        entity.parent = getByName(entity.parentName);
        drv::drv_assert(entity.parent != Entity::INVALID_ENTITY, "Could not find parent entity");
    }
    if (entity.mass >= 0) {
        Physics::Shape physicsShape;
        if (entity.modelName == "box")
            physicsShape = Physics::CUBE;
        else if (entity.modelName == "sphere")
            physicsShape = Physics::SPHERE;
        else
            throw std::runtime_error("Unknown shape: " + entity.modelName);
        entity.rigidBody = physics->addRigidBody(physicsShape, entity.mass, entity.scale,
                                                 entity.position, entity.rotation, entity.velocity);
    }
    {
        std::shared_lock<std::shared_mutex> lock(entitiesMutex);
        for (Entity::EntityId i = 0; i < Entity::EntityId(entities.size()); ++i) {
            Entity* itr = getById(i);
            std::unique_lock<std::shared_mutex> entityLock(itr->mutex);
            if (itr->gameBehaviour == 0 && itr->engineBehaviour == 0) {
                *itr = std::move(entity);
                return i;
            }
        }
    }
    std::unique_lock<std::shared_mutex> lock(entitiesMutex);
    Entity::EntityId ret = Entity::EntityId(entities.size());
    entities.push_back(std::move(entity));
    return ret;
}

void EntityManager::removeEntity(Entity::EntityId id) {
    if (id == Entity::INVALID_ENTITY)
        return;
    // std::shared_lock<std::shared_mutex> lock(entitiesMutex);
    auto& entity = entities[size_t(id)];
    if (entity.gameBehaviour == 0 && entity.engineBehaviour == 0)
        return;
    if (entity.rigidBody) {
        physics->removeRigidBody(static_cast<RigidBodyPtr>(entity.rigidBody));
        entity.rigidBody = nullptr;
    }
    entity.gameBehaviour = 0;
    entity.engineBehaviour = 0;
    entity.templateName = "";
}

EntityManager::EntitySystemInfo EntityManager::addEntitySystem(std::string name,
                                                               FrameGraph::Stages stages,
                                                               EntitySystemSignature signature,
                                                               EntitySystemCb entitySystemCb) {
    EntitySystemInfo info;
    info.type = signature.type;
    info.id = info.type == EntitySystemInfo::ENGINE_SYSTEM ? (numEngineEs++) : (numGameEs++);
    info.flag = 1 << info.id;
    info.stages = stages;
    info.constSystem = signature.constSystem;
    info.nodeId = frameGraph->addNode(FrameGraph::Node(std::move(name), stages));
    info.entitySystemCb = entitySystemCb;
    esSignatures.push_back(info);
    return info;
}

void EntityManager::node_loop(EntityManager* entityManager, Engine* engine, FrameGraph* frameGraph,
                              const EntitySystemInfo* info) {
    // std::unique_lock<std::mutex> nodeLock();
    FrameId frameId = 0;
    Clock::time_point startTime = Clock::now();
    Clock::time_point prevTime[FrameGraph::NUM_STAGES] = {startTime};
    while (true) {
        for (uint32_t stageId = 0; stageId < FrameGraph::NUM_STAGES; ++stageId) {
            FrameGraph::Stage stage = FrameGraph::get_stage(stageId);
            if (stage == FrameGraph::EXECUTION_STAGE || !(info->stages & stage))
                continue;
            if (FrameGraph::NodeHandle nodeHandle =
                  frameGraph->acquireNode(info->nodeId, stage, frameId);
                nodeHandle) {
                EntitySystemParams params;
                Clock::time_point currTime = Clock::now();
                params.frameId = frameId;
                params.uptime =
                  float(std::chrono::duration_cast<Duration>(currTime - startTime).count());
                params.dt =
                  float(std::chrono::duration_cast<Duration>(currTime - prevTime[stageId]).count());
                // Avoid huge jumps on freezes and debug breaks
                if (entityManager->frozen)
                    params.dt = 0;
                if (params.dt > 0.5f)
                    params.dt = 0.016f;
                prevTime[stageId] = currTime;
                entityManager->performES(*info, [&](Entity::EntityId id, Entity* entity,
                                                    FlexibleArray<Entity, 4>& outEntities) {
                    info->entitySystemCb(entityManager, engine, &nodeHandle, stage, params, entity,
                                         id, outEntities);
                });
            }
            else
                return;
        }
        frameId++;
    }
}

EntityManager::~EntityManager() {
    for (Entity::EntityId i = 0; i < Entity::EntityId(entities.size()); ++i)
        removeEntity(i);
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

void EntityManager::addEntityTemplate(std::string name, EntityTemplate entityTemplate) {
    esTemplates[name] = entityTemplate;
}

// void EntityManager::prepareTexture(uint32_t texId, drv::DrvCmdBufferRecorder* recorder) {
//     std::unique_lock<std::mutex> lock(dirtyTextureMutex);
//     if (dirtyTextures.count(texId)) {
//         dirtyTextures.extract(texId);

//         recorder->cmdImageBarrier({textureStager.getImage(texId), drv::IMAGE_USAGE_TRANSFER_SOURCE,
//                                    drv::ImageMemoryBarrier::AUTO_TRANSITION, false});
//         recorder->cmdImageBarrier({textures.getImage(texId), drv::IMAGE_USAGE_TRANSFER_DESTINATION,
//                                    drv::ImageMemoryBarrier::AUTO_TRANSITION, true});
//         drv::ImageCopyRegion region;
//         region.srcSubresource.aspectMask = drv::COLOR_BIT;
//         region.srcSubresource.baseArrayLayer = 0;
//         region.srcSubresource.layerCount = 1;
//         region.srcSubresource.mipLevel = 0;
//         region.srcOffset = {0, 0, 0};
//         region.dstSubresource.aspectMask = drv::COLOR_BIT;
//         region.dstSubresource.baseArrayLayer = 0;
//         region.dstSubresource.layerCount = 1;
//         region.dstSubresource.mipLevel = 0;
//         region.dstOffset = {0, 0, 0};
//         region.extent = drv::get_texture_info(textureStager.getImage(texId)).extent;
//         recorder->cmdCopyImage(textureStager.getImage(texId), textures.getImage(texId), 1, &region);
//     }
// }

// drv::ImagePtr EntityManager::getTexture(uint32_t texId) const {
//     return textureStager.getImage(texId);
// }

Entity::EntityId EntityManager::getByName(const std::string& name) const {
    // std::shared_lock<std::shared_mutex> lock(entitiesMutex);
    for (size_t id = 0; id < entities.size(); ++id)
        if ((entities[id].gameBehaviour != 0 || entities[id].engineBehaviour != 0)
            && entities[id].name == name)
            return Entity::EntityId(id);
    return Entity::INVALID_ENTITY;
}

// void EntityManager::setVelocity(Entity::EntityId entityId, const glm::vec3& velocity) {

// }
// glm::vec3 EntityManager::getVelocity(Entity::EntityId entityId) {

// }
