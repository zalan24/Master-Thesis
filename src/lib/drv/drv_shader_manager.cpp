#include "drv_shader_manager.h"

#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <drverror.h>

#include "drv.h"

using namespace drv;

namespace drv
{
struct ShaderData
{
    struct ShaderModuleInfo
    {
        ShaderModulePtr module;
        ShaderStage::FlagType stage;
        std::vector<DescriptorSetLayoutPtr> layouts;
    };
    LogicalDevicePtr device;
    std::unordered_map<ShaderIdType, ShaderInfo> shaders;
    std::vector<std::vector<DescriptorSetLayoutCreateInfo>> descriptorSetLayoutCreateInfos;
    std::vector<std::vector<DescriptorSetLayoutCreateInfo::Binding>> bindings;
    std::vector<DescriptorSetLayoutPtr> layouts;
    std::unordered_map<ShaderIdType, ShaderModuleInfo> shaderModules;
    ShaderData(LogicalDevicePtr _device = NULL_HANDLE) : device(_device) {}
    ~ShaderData() {
        for (auto& itr : shaderModules)
            drv_assert(destroy_shader_module(device, itr.second.module));
        for (auto& itr : shaders)
            drv_assert(destroy_shader_create_info(itr.second.createInfo),
                       "Could not destroy shader create info");
    }

    ShaderData(const ShaderData&) = delete;
    ShaderData& operator=(const ShaderData&) = delete;

    ShaderData(ShaderData&&) = default;
    ShaderData& operator=(ShaderData&&) = default;
};

struct DeviceData
{
    ShaderData shaders;
};
static std::unordered_map<LogicalDevicePtr, DeviceData>* deviceData = nullptr;
}  // namespace drv

void drv::init_shader_manager(LogicalDevicePtr device) {
    if (deviceData == nullptr) {
        deviceData = new std::remove_pointer_t<decltype(deviceData)>;  // ...
    }
    drv_assert(deviceData->find(device) == deviceData->end(),
               "Shader manager on device already initialized");
    (*deviceData)[device] = DeviceData{ShaderData{device}};
}

void drv::register_shader(LogicalDevicePtr device, ShaderIdType name,
                          const ShaderInfo& shaderInfo) {
    drv_assert(deviceData != nullptr, "Shader manager has not beet initialized");
    auto itr = deviceData->find(device);
    drv_assert(itr != deviceData->end(), "Shader manager has not beet initialized");
    auto shaderItr = itr->second.shaders.shaders.find(name);
    drv_assert(shaderItr == itr->second.shaders.shaders.end(),
               "A shader already exists with this name");
    ShaderInfo& info = itr->second.shaders.shaders[name];
    info = shaderInfo;
    itr->second.shaders.descriptorSetLayoutCreateInfos.push_back({});
    std::vector<DescriptorSetLayoutCreateInfo>& descriptorSetLayoutCreateInfos =
      itr->second.shaders.descriptorSetLayoutCreateInfos.back();
    descriptorSetLayoutCreateInfos.reserve(info.numDescriptorSetLayouts);
    for (unsigned int i = 0; i < info.numDescriptorSetLayouts; ++i) {
        descriptorSetLayoutCreateInfos[i] = info.descriptorSetLayoutInfos[i];
        itr->second.shaders.bindings.push_back({});
        std::vector<DescriptorSetLayoutCreateInfo::Binding>& bindings =
          itr->second.shaders.bindings.back();
        bindings.reserve(info.descriptorSetLayoutInfos[i].numBindings);
        for (unsigned int j = 0; j < info.descriptorSetLayoutInfos[i].numBindings; ++j)
            bindings.push_back(info.descriptorSetLayoutInfos[i].bindings[j]);
        descriptorSetLayoutCreateInfos[i].bindings = bindings.data();
    }
    info.descriptorSetLayoutInfos = descriptorSetLayoutCreateInfos.data();
}

namespace drv
{
struct descriptor_layout_info_less
{
    bool operator()(const drv::DescriptorSetLayoutCreateInfo* a,
                    const drv::DescriptorSetLayoutCreateInfo* b) const {
        if (a->numBindings != b->numBindings)
            return a->numBindings < b->numBindings;
        for (unsigned int i = 0; i < a->numBindings; ++i) {
            if (a->bindings[i].slot != b->bindings[i].slot)
                return a->bindings[i].slot < b->bindings[i].slot;
            if (a->bindings[i].type != b->bindings[i].type)
                return a->bindings[i].type < b->bindings[i].type;
            if (a->bindings[i].count != b->bindings[i].count)
                return a->bindings[i].count < b->bindings[i].count;
            if (a->bindings[i].stages != b->bindings[i].stages)
                return a->bindings[i].stages < b->bindings[i].stages;
            if (a->bindings[i].immutableSamplers != b->bindings[i].immutableSamplers)
                return a->bindings[i].immutableSamplers < b->bindings[i].immutableSamplers;
        }
        return false;
    }
};
}  // namespace drv

void drv::build_shaders(LogicalDevicePtr device) {
    drv_assert(deviceData != nullptr, "Shader manager has not beet initialized");
    auto itr = deviceData->find(device);
    drv_assert(itr != deviceData->end(), "Shader manager has not beet initialized");
    std::map<const drv::DescriptorSetLayoutCreateInfo*, drv::DescriptorSetLayoutPtr,
             descriptor_layout_info_less>
      layouts;
    for (auto& shaderInfo : itr->second.shaders.shaders) {
        ShaderData::ShaderModuleInfo& moduleInfo =
          itr->second.shaders.shaderModules[shaderInfo.first];
        for (unsigned int j = 0; j < shaderInfo.second.numDescriptorSetLayouts; ++j) {
            auto layoutItr = layouts.find(&shaderInfo.second.descriptorSetLayoutInfos[j]);
            if (layoutItr != layouts.end())
                continue;  // this layout already exists
            DescriptorSetLayoutPtr layout =
              create_descriptor_set_layout(device, &shaderInfo.second.descriptorSetLayoutInfos[j]);
            itr->second.shaders.layouts.push_back(layout);
            moduleInfo.layouts.push_back(layout);
        }
        moduleInfo.stage = shaderInfo.second.stage;
        moduleInfo.module = create_shader_module(device, shaderInfo.second.createInfo);
        drv_assert(moduleInfo.module != NULL_HANDLE, "Could not create shader module");
    }
}

void drv::destroy_shader_mananger(LogicalDevicePtr device) {
    drv_assert(deviceData != nullptr,
               "Shader manager has not beet initialized, but it's destroyed");
    auto itr = deviceData->find(device);
    drv_assert(itr != deviceData->end(),
               "Shader manager has not beet initialized, but it's destroyed");
    deviceData->erase(itr);
    if (deviceData->size() == 0)
        delete deviceData;
}

ShaderModulePtr drv::get_shader_module(LogicalDevicePtr device, ShaderIdType shaderId) {
    drv_assert(deviceData != nullptr, "Shader manager has not beet initialized");
    auto itr = deviceData->find(device);
    drv_assert(itr != deviceData->end(), "Shader manager has not beet initialized");
    auto shaderItr = itr->second.shaders.shaderModules.find(shaderId);
    if (shaderItr == itr->second.shaders.shaderModules.end())
        return NULL_HANDLE;
    return shaderItr->second.module;
}

unsigned int drv::get_num_shader_descriptor_set_layouts(LogicalDevicePtr device,
                                                        ShaderIdType shaderId) {
    drv_assert(deviceData != nullptr, "Shader manager has not beet initialized");
    auto itr = deviceData->find(device);
    drv_assert(itr != deviceData->end(), "Shader manager has not beet initialized");
    auto shaderItr = itr->second.shaders.shaderModules.find(shaderId);
    if (shaderItr == itr->second.shaders.shaderModules.end())
        return 0;
    return static_cast<unsigned int>(shaderItr->second.layouts.size());
}

DescriptorSetLayoutPtr* drv::get_shader_descriptor_set_layouts(LogicalDevicePtr device,
                                                               ShaderIdType shaderId) {
    drv_assert(deviceData != nullptr, "Shader manager has not beet initialized");
    auto itr = deviceData->find(device);
    drv_assert(itr != deviceData->end(), "Shader manager has not beet initialized");
    auto shaderItr = itr->second.shaders.shaderModules.find(shaderId);
    if (shaderItr == itr->second.shaders.shaderModules.end())
        return nullptr;
    return shaderItr->second.layouts.data();
}
