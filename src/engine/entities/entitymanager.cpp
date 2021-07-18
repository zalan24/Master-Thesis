#include "entitymanager.h"

#include <algorithm>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <namethreads.h>

#include <framegraph.h>
#include <logger.h>

#include "entity.h"

namespace fs = std::filesystem;

EntityManager::EntityManager(drv::PhysicalDevicePtr _physicalDevice, drv::LogicalDevicePtr _device,
                             FrameGraph* _frameGraph, const std::string& textureFolder)
  : physicalDevice(_physicalDevice), device(_device), frameGraph(_frameGraph) {
    std::vector<drv::ImageSet::ImageInfo> imageInfos;
    std::vector<drv::ImageSet::ImageInfo> imageStagerInfos;
    std::vector<unsigned char*> imageData;
    std::vector<int> imageChannels;
    drv::drv_assert(fs::exists(fs::path{textureFolder}), "Textures folders does not exist");
    for (auto& p : fs::directory_iterator(fs::path{textureFolder})) {
        const std::string filename = p.path().filename().string();
        const std::string ext = p.path().extension().string();
        if (ext != ".png" && ext != ".jpg" && ext != ".bmp")
            continue;
        int width, height, channels;
        unsigned char* image = stbi_load(p.path().string().c_str(), &width, &height, &channels, 0);
        drv::drv_assert(image != nullptr, "Could not load image");
        imageData.push_back(image);
        imageChannels.push_back(channels);
        drv::ImageSet::ImageInfo imageInfo;
        imageInfo.imageId = drv::ImageId(p.path().filename().string());
        imageInfo.type = imageInfo.TYPE_2D;
        imageInfo.format = drv::ImageFormat::R8G8B8A8_UNORM;
        imageInfo.extent = {uint32_t(width), uint32_t(height), 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.sampleCount = drv::SampleCount::SAMPLE_COUNT_1;
        imageInfo.tiling = imageInfo.TILING_LINEAR;
        imageInfo.usage = imageInfo.TRANSFER_DST_BIT | imageInfo.TRANSFER_SRC_BIT;
        textureId[p.path().filename().string()] = uint32_t(imageInfos.size());
        imageInfos.push_back(imageInfo);
        imageStagerInfos.push_back(imageInfo);
        LOG_F(INFO, "Texture loaded in entity manager: %s", p.path().filename().string().c_str());
    }
    textures = drv::ImageSet(physicalDevice, device, std::move(imageInfos),
                             drv::ImageSet::PreferenceSelector(drv::MemoryType::DEVICE_LOCAL_BIT,
                                                               drv::MemoryType::DEVICE_LOCAL_BIT));
    textureStager =
      drv::ImageSet(physicalDevice, device, std::move(imageInfos),
                    drv::ImageSet::PreferenceSelector(drv::MemoryType::HOST_COHERENT_BIT
                                                        | drv::MemoryType::HOST_CACHED_BIT
                                                        | drv::MemoryType::HOST_VISIBLE_BIT,
                                                      drv::MemoryType::HOST_VISIBLE_BIT));
    for (uint32_t i = 0; i < imageData.size(); ++i) {
        drv::TextureInfo texInfo = drv::get_texture_info(textureStager.getImage(i));

        drv::DeviceSize offset;
        drv::DeviceSize size;
        drv::DeviceSize rowPitch;
        drv::DeviceSize arrayPitch;
        drv::DeviceSize depthPitch;
        drv::drv_assert(drv::get_image_memory_data(device, textureStager.getImage(i), 0, 0, offset,
                                                   size, rowPitch, arrayPitch, depthPitch),
                        "Could not get image memory data");

        StackMemory::MemoryHandle<uint32_t> srcData(size, TEMPMEM);
        for (uint32_t y = 0; y < texInfo.extent.height; ++y) {
            for (uint32_t x = 0; x < texInfo.extent.width; ++x) {
                uint32_t r, g, b, a;
                r = g = b = 0;
                a = 255;
                uint32_t numChannels = uint32_t(imageChannels[i]);
                r = imageData[i][(x + texInfo.extent.width * y) * numChannels + 0];
                if (numChannels > 1)
                    g = imageData[i][(x + texInfo.extent.width * y) * numChannels + 1];
                if (numChannels > 2)
                    b = imageData[i][(x + texInfo.extent.width * y) * numChannels + 2];
                if (numChannels > 3)
                    a = imageData[i][(x + texInfo.extent.width * y) * numChannels + 3];
                srcData[StackMemory::size_t(x + y * rowPitch / 4)] =
                  (a << 24) + (b << 16) + (g << 8) + r;
            }
        }
        TemporalResourceLockerDescriptor resourceDesc;
        resourceDesc.addImage(textureStager.getImage(i), 1, 0, 0, drv::AspectFlagBits::COLOR_BIT,
                              drv::ResourceLockerDescriptor::WRITE);
        auto lock = frameGraph->getResourceLocker()->lock(&resourceDesc).getLock();
        drv::write_image_memory(device, textureStager.getImage(i), 0, 0, lock, srcData);
        stbi_image_free(imageData[i]);
    }
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
    auto templateItr = esTemplates.find(entity.templateName);
    drv::drv_assert(templateItr != esTemplates.end(), "Unknown entity template");
    entity.gameBehaviour = templateItr->second.gameBehaviour;
    entity.engineBehaviour = templateItr->second.engineBehaviour;
    if (entity.textureName != "") {
        auto textureItr = textureId.find(entity.textureName);
        drv::drv_assert(textureItr != textureId.end(), "Could not find a texture for an entity");
        entity.textureId = textureItr->second;
    }
    for (Entity::EntityId i = 0; i < Entity::EntityId(entities.size()); ++i) {
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
    while (true) {
        for (uint32_t stageId = 0; stageId < FrameGraph::NUM_STAGES; ++stageId) {
            FrameGraph::Stage stage = FrameGraph::get_stage(stageId);
            if (stage == FrameGraph::EXECUTION_STAGE || !(info->stages & stage))
                continue;
            if (FrameGraph::NodeHandle nodeHandle =
                  frameGraph->acquireNode(info->nodeId, stage, frameId);
                nodeHandle) {
                entityManager->performES(*info, [&](Entity* entity) {
                    info->entitySystemCb(entityManager, engine, &nodeHandle, stage, entity);
                });
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

void EntityManager::addEntityTemplate(std::string name, EntityTemplate entityTemplate) {
    esTemplates[name] = entityTemplate;
}
