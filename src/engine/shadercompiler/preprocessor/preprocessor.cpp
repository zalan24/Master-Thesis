#include "preprocessor.h"

#include <cassert>
#include <cctype>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>

#include <simplecpp.h>

#include <shadertypes.h>

#include <blockfile.h>
#include <shaderbin.h>
#include <uncomment.h>
#include <util.hpp>

namespace fs = std::filesystem;

Resources& Resources::operator+=(const Resources& rhs) {
    for (const auto& [name, type] : rhs.variables) {
        if (variables.find(name) != variables.end())
            throw std::runtime_error("Variable already exists in this resources object");
        variables[name] = type;
    }
    return *this;
}

static void read_variants(const BlockFile* blockFile, Variants& variants) {
    variants = {};
    if (blockFile->hasNodes())
        throw std::runtime_error("variants block cannot contain nested blocks");
    if (!blockFile->hasContent())
        return;
    const std::string* variantContent = blockFile->getContent();
    std::regex paramReg{"(\\w+)\\s*:((\\s*\\w+\\s*,)+\\s*\\w+\\s*);"};
    std::regex valueReg{"\\s*(\\w+)\\s*(,)?"};
    auto variantsBegin =
      std::sregex_iterator(variantContent->begin(), variantContent->end(), paramReg);
    auto variantsEnd = std::sregex_iterator();
    for (std::sregex_iterator regI = variantsBegin; regI != variantsEnd; ++regI) {
        std::string paramName = (*regI)[1];
        std::string values = (*regI)[2];
        auto& vec = variants.values[paramName] = {};
        auto valuesBegin = std::sregex_iterator(values.begin(), values.end(), valueReg);
        auto valuesEnd = std::sregex_iterator();
        for (std::sregex_iterator regJ = valuesBegin; regJ != valuesEnd; ++regJ) {
            std::string value = (*regJ)[1];
            vec.push_back(value);
        }
    }
}

static void read_resources(const BlockFile* blockFile, Resources& resources) {
    if (blockFile->hasNodes())
        throw std::runtime_error("resources block cannot contain nested blocks");
    if (!blockFile->hasContent())
        return;
    const std::string* resourcesContent = blockFile->getContent();
    std::regex varReg{
      "(uint|uvec2|uvec3|uvec4|int|ivec2|ivec3|ivec4|float|vec2|vec3|vec4|mat4)\\s+(\\w+)\\s*;"};
    auto resourcesBegin =
      std::sregex_iterator(resourcesContent->begin(), resourcesContent->end(), varReg);
    auto resourcesEnd = std::sregex_iterator();
    for (std::sregex_iterator regI = resourcesBegin; regI != resourcesEnd; ++regI) {
        std::string varType = (*regI)[1];
        std::string varName = (*regI)[2];
        if (resources.variables.find(varName) != resources.variables.end())
            throw std::runtime_error("A variable already exists with this name: " + varName);
        resources.variables[varName] = varType;
    }
}

template <typename F>
static void collect_shader_f(const BlockFile& blockFile, const std::string& type, F&& f) {
    for (size_t i = 0; i < blockFile.getBlockCount("stages"); ++i) {
        const BlockFile* b = blockFile.getNode("stages", i);
        if (b->hasNodes()) {
            for (size_t j = 0; j < b->getBlockCount(type); ++j) {
                const BlockFile* b2 = b->getNode(type, j);
                f(i, b, j, b2);
            }
        }
        else if (b->hasContent())
            // completely empty block is allowed
            throw std::runtime_error(
              "A shader block (stages) contains direct content instead of separate blocks for stages");
    }
}

static void collect_shader_cfg(const BlockFile& blockFile, std::ostream& stagesOut,
                               const std::string& type) {
    collect_shader_f(blockFile, type, [&](size_t, const BlockFile*, size_t, const BlockFile* node) {
        if (node->hasContent()) {
            stagesOut << *node->getContent();
        }
        else if (node->hasNodes())
            // completely empty block is allowed
            throw std::runtime_error("A shader block (stages/" + type
                                     + ") contains nested blocks instead of content");
    });
}

static void collect_shader(const BlockFile& blockFile, std::ostream* shaderOut,
                           std::ostream* cfgOut, const std::string& type) {
    if (cfgOut)
        collect_shader_cfg(blockFile, *cfgOut, type);
    if (shaderOut) {
        for (size_t i = 0; i < blockFile.getBlockCount("global"); ++i) {
            const BlockFile* b = blockFile.getNode("global", i);
            if (b->hasContent()) {
                *shaderOut << *b->getContent();
            }
            else if (b->hasNodes()) {
                // completely empty block is allowed
                throw std::runtime_error(
                  "A shader block (global) contains nested blocks instead of content");
            }
        }
        for (size_t i = 0; i < blockFile.getBlockCount(type); ++i) {
            const BlockFile* b = blockFile.getNode(type, i);
            if (b->hasContent()) {
                *shaderOut << *b->getContent();
            }
            else if (b->hasNodes()) {
                // completely empty block is allowed
                throw std::runtime_error("A shader block (" + type
                                         + ") contains nested blocks instead of content.");
            }
        }
    }
}

static std::array<std::string, ShaderBin::NUM_STAGES> getBlockNames() {
    std::array<std::string, ShaderBin::NUM_STAGES> blockNames;
    blockNames[ShaderBin::VS] = "vs";
    blockNames[ShaderBin::PS] = "ps";
    blockNames[ShaderBin::CS] = "cs";
    static_assert(ShaderBin::NUM_STAGES == 3, "Update this code as well");
    return blockNames;
}

static void read_gen_input(const BlockFile& shaderFile, ShaderGenerationInput& genInput) {
    collect_shader_cfg(shaderFile, genInput.statesCfg, "states");
    for (uint32_t i = 0; i < ShaderBin::NUM_STAGES; ++i)
        collect_shader(shaderFile, nullptr, &genInput.stageConfigs[i], getBlockNames()[i]);
    // collect_shader(shaderFile, genInput.ps, genInput.psCfg, "ps");
    // collect_shader(shaderFile, genInput.cs, genInput.csCfg, "cs");
    collect_shader_f(
      shaderFile, "attachments", [&](size_t, const BlockFile*, size_t, const BlockFile* node) {
          if (node->hasNodes()) {
              for (size_t i = 0; i < node->getBlockCount(); ++i) {
                  const BlockFile* attachment = node->getNode(i);
                  if (attachment->hasNodes())
                      throw std::runtime_error(
                        "Attachment infos inside the attachments block must not contain other blocks");
                  if (attachment->hasContent())
                      genInput.attachments[node->getBlockName(i)] << *attachment->getContent();
                  else
                      genInput.attachments[node->getBlockName(i)] << "";
              }
          }
          else if (node->hasContent())
              throw std::runtime_error(
                "The attachments block in shader stage infos must contain a block for each used attachment, not raw content");
          return true;
      });
}

ShaderGenerationInput ShaderObjectData::readGenInput() const {
    std::stringstream cu;
    includeHeaders(cu);
    BlockFile cuBlocks(cu, false);

    ShaderGenerationInput ret;
    read_gen_input(cuBlocks, ret);
    return ret;
}

ShaderObjectData::ComputeUnit ShaderObjectData::readComputeUnite(
  ShaderGenerationInput* outCfg) const {
    std::stringstream cu;
    includeHeaders(cu);
    BlockFile cuBlocks(cu, false);
    if (outCfg)
        read_gen_input(cuBlocks, *outCfg);
    ComputeUnit ret;
    for (uint32_t i = 0; i < ShaderBin::NUM_STAGES; ++i)
        collect_shader(cuBlocks, &ret.stages[i], nullptr, getBlockNames()[i]);
    return ret;
}

void ShaderObjectData::includeHeaders(std::ostream& out) const {
    for (const auto& header : allIncludes) {
        auto itr = headerLocations.find(header);
        if (itr == headerLocations.end())
            throw std::runtime_error("Unknown shader header: " + header);
        const std::string& filename = itr->second;
        std::ifstream in(filename.c_str());
        if (!in.is_open())
            throw std::runtime_error("Could not open included file: " + filename);
        uncomment(in, out);
    }
}

static std::string get_variant_enum_name(std::string name) {
    for (char& c : name)
        c = static_cast<char>(tolower(c));
    name[0] = static_cast<char>(toupper(name[0]));
    return name;
}

static std::string get_variant_enum_val_name(std::string name) {
    for (char& c : name)
        c = static_cast<char>(tolower(c));
    return name;
}

static std::string get_variant_enum_value(std::string value) {
    for (char& c : value)
        c = static_cast<char>(toupper(c));
    return value;
}

static VariantConfig get_variant_config(
  uint32_t variantId, const std::vector<Variants>& variants,
  const std::map<std::string, uint32_t>& variantParamMultiplier) {
    VariantConfig ret;
    for (const Variants& v : variants) {
        for (const auto& [name, values] : v.values) {
            auto itr = variantParamMultiplier.find(name);
            assert(itr != variantParamMultiplier.end());
            size_t valueId = (variantId / itr->second) % values.size();
            ret.variantValues[name] = valueId;
        }
    }
    return ret;
}

std::string format_variant(uint32_t variantId, const std::vector<Variants>& variants,
                           const std::stringstream& text,
                           const std::map<std::string, uint32_t>& variantParamMultiplier) {
    VariantConfig config = get_variant_config(variantId, variants, variantParamMultiplier);
    simplecpp::DUI dui;
    for (const Variants& v : variants)
        for (const auto& [key, values] : v.values)
            for (size_t ind = 0; ind < values.size(); ++ind)
                dui.defines.push_back(values[ind] + "=" + std::to_string(ind));
    for (const auto& [key, value] : config.variantValues)
        dui.defines.push_back(key + "=" + std::to_string(value));
    std::stringstream shaderCopy(text.str());
    std::vector<std::string> files;
    simplecpp::TokenList rawtokens(shaderCopy, files);
    std::map<std::string, simplecpp::TokenList*> included = simplecpp::load(rawtokens, files, dui);
    simplecpp::TokenList outputTokens(files);
    simplecpp::preprocess(outputTokens, rawtokens, files, included, dui);
    return outputTokens.stringify();
}

template <typename T>
static const T& translate_input(const std::string& str, const std::map<std::string, T>& values) {
    auto itr = values.find(str);
    if (itr != values.end())
        return itr->second;
    std::stringstream message;
    message << "Invalid value: <" << str << ">. Valid values are {";
    for (const auto& value : values)
        message << " " << value.first;
    message << " }";
    throw std::runtime_error(message.str());
}

static std::unordered_map<std::string, std::string> read_values(const std::string& s) {
    std::regex valueReg{"\\s*(\\w+)\\s*=\\s*(\\w+)\\s*(;|\\Z)"};
    std::unordered_map<std::string, std::string> ret;
    auto begin = std::sregex_iterator(s.begin(), s.end(), valueReg);
    auto end = std::sregex_iterator();
    for (std::sregex_iterator regI = begin; regI != end; ++regI)
        ret[(*regI)[1]] = (*regI)[2];
    return ret;
}

static ResourceUsage read_used_resources(const std::string& s, const Resources& resources) {
    std::regex resourceReg{"\\s*use\\s+(\\w+)\\s*;"};
    ResourceUsage ret;
    auto begin = std::sregex_iterator(s.begin(), s.end(), resourceReg);
    auto end = std::sregex_iterator();
    for (std::sregex_iterator regI = begin; regI != end; ++regI) {
        std::string name = (*regI)[1];
        if (resources.variables.find(name) != resources.variables.end())
            ret.usedVars.insert(name);
        else
            throw std::runtime_error("Unknown resource: " + name);
    }
    return ret;
}

ShaderBin::StageConfig read_stage_configs(
  const Resources& resources, uint32_t variantId, const std::vector<Variants>& variants,
  const std::map<std::string, uint32_t>& variantParamMultiplier, const ShaderGenerationInput& input,
  PipelineResourceUsage& resourceUsage) {
    std::string statesInfo =
      format_variant(variantId, variants, input.statesCfg, variantParamMultiplier);
    std::string stageInfos[ShaderBin::NUM_STAGES];
    std::unordered_map<std::string, std::string> stageValues[ShaderBin::NUM_STAGES];
    std::unordered_map<std::string, std::string> statesValues = read_values(statesInfo);
    std::unordered_set<std::string> usedValue;
    for (uint32_t i = 0; i < ShaderBin::NUM_STAGES; ++i) {
        stageInfos[i] =
          format_variant(variantId, variants, input.stageConfigs[i], variantParamMultiplier);
        stageValues[i] = read_values(stageInfos[i]);
        resourceUsage.usages[i] = read_used_resources(stageInfos[i], resources);
    }
    auto read_value = [&](std::unordered_map<std::string, std::string>& values,
                          const std::string& name) -> const std::string* {
        usedValue.insert(name);
        if (values.count(name) > 1)
            throw std::runtime_error("Shater state overwrites are currently not supported");
        auto itr = values.find(name);
        if (itr != values.end())
            return &itr->second;
        return nullptr;
    };
    ShaderBin::StageConfig ret;

    for (uint32_t i = 0; i < ShaderBin::NUM_STAGES; ++i)
        if (auto value = read_value(stageValues[i], "entry"))
            ret.entryPoints[i] = *value;
    if (auto value = read_value(statesValues, "polygonMode"))
        ret.polygonMode =
          translate_input<drv::PolygonMode>(*value, {{"fill", drv::PolygonMode::FILL},
                                                     {"line", drv::PolygonMode::LINE},
                                                     {"point", drv::PolygonMode::POINT}});
    if (auto value = read_value(statesValues, "cull"))
        ret.cullMode =
          translate_input<drv::CullMode>(*value, {{"none", drv::CullMode::NONE},
                                                  {"front", drv::CullMode::FRONT},
                                                  {"back", drv::CullMode::BACK},
                                                  {"all", drv::CullMode::FRONT_AND_BACK}});
    if (auto value = read_value(statesValues, "depthCompare"))
        ret.depthCompare = translate_input<drv::CompareOp>(
          *value, {{"never", drv::CompareOp::NEVER},
                   {"less", drv::CompareOp::LESS},
                   {"equal", drv::CompareOp::EQUAL},
                   {"less_or_equal", drv::CompareOp::LESS_OR_EQUAL},
                   {"greater", drv::CompareOp::GREATER},
                   {"not_equal", drv::CompareOp::NOT_EQUAL},
                   {"greater_or_equal", drv::CompareOp::GREATER_OR_EQUAL},
                   {"always", drv::CompareOp::ALWAYS}});
    if (auto value = read_value(statesValues, "useDepthClamp"))
        ret.useDepthClamp = translate_input<bool>(*value, {{"true", true}, {"false", false}});
    if (auto value = read_value(statesValues, "depthBiasEnable"))
        ret.depthBiasEnable = translate_input<bool>(*value, {{"true", true}, {"false", false}});
    if (auto value = read_value(statesValues, "depthTest"))
        ret.depthTest = translate_input<bool>(*value, {{"true", true}, {"false", false}});
    if (auto value = read_value(statesValues, "depthWrite"))
        ret.depthWrite = translate_input<bool>(*value, {{"true", true}, {"false", false}});
    if (auto value = read_value(statesValues, "stencilTest"))
        ret.stencilTest = translate_input<bool>(*value, {{"true", true}, {"false", false}});
    for (const auto& [name, cfg] : input.attachments) {
        std::string attachments = format_variant(variantId, variants, cfg, variantParamMultiplier);
        std::unordered_map<std::string, std::string> values = read_values(attachments);
        ShaderBin::AttachmentInfo attachmentInfo;
        if (auto itr = values.find("location"); itr != values.end()) {
            int loc = std::atoi(itr->second.c_str());
            if (loc < 0)
                continue;
            attachmentInfo.location = safe_cast<uint8_t>(loc);
        }
        else
            throw std::runtime_error(
              "An attachment description must contain a 'location' parameter: " + name);
        attachmentInfo.name = name;
        attachmentInfo.info = 0;
        if (auto itr = values.find("channels"); itr != values.end()) {
            if (itr->second.find('a') != std::string::npos)
                attachmentInfo.info |= ShaderBin::AttachmentInfo::USE_ALPHA;
            if (itr->second.find('r') != std::string::npos)
                attachmentInfo.info |= ShaderBin::AttachmentInfo::USE_RED;
            if (itr->second.find('g') != std::string::npos)
                attachmentInfo.info |= ShaderBin::AttachmentInfo::USE_GREEN;
            if (itr->second.find('b') != std::string::npos)
                attachmentInfo.info |= ShaderBin::AttachmentInfo::USE_BLUE;
            if (attachmentInfo.info == 0)
                throw std::runtime_error("No channels are used by an attachment: " + name);
        }
        else
            throw std::runtime_error("Missing 'channels' parameter from attachment description: "
                                     + name);
        if (auto itr = values.find("type"); itr != values.end()) {
            // TODO depth stencil
            if (itr->second == "output")
                attachmentInfo.info |= ShaderBin::AttachmentInfo::WRITE;
            else if (itr->second != "input")
                throw std::runtime_error("Unknown attachment type: " + itr->second + "  (" + name
                                         + ")");
        }
        ret.attachments.push_back(std::move(attachmentInfo));
    }
    for (const auto& itr : statesValues)
        if (usedValue.count(itr.first) == 0)
            throw std::runtime_error("Unknown value in shader config: " + itr.first);
    for (uint32_t i = 0; i < ShaderBin::NUM_STAGES; ++i)
        for (const auto& itr : stageValues[i])
            if (usedValue.count(itr.first) == 0)
                throw std::runtime_error("Unknown value in shader config: " + itr.first);
    return ret;
}

void Preprocessor::readIncludes(const BlockFile& b, std::set<std::string>& directIncludes) const {
    std::regex headerReg{"((\\w+\\/)*(\\w+))"};
    for (size_t i = 0; i < b.getBlockCount("include"); ++i) {
        const BlockFile* inc = b.getNode("include", i);
        if (!inc->hasContent())
            throw std::runtime_error("Invalid include block");
        const std::string* headerContent = inc->getContent();
        auto headersBegin =
          std::sregex_iterator(headerContent->begin(), headerContent->end(), headerReg);
        auto headersEnd = std::sregex_iterator();
        for (std::sregex_iterator regI = headersBegin; regI != headersEnd; ++regI) {
            std::string headerId = (*regI)[0];
            auto itr = data.headers.find(headerId);
            if (itr == data.headers.end())
                throw std::runtime_error("Could not find header: " + headerId);
            directIncludes.insert(headerId);
        }
    }
}

std::string Preprocessor::collectIncludes(const std::string& header,
                                          std::vector<std::string>& includes) const {
    if (std::find(includes.begin(), includes.end(), header) != includes.end())
        return "";
    auto itr = data.headers.find(header);
    if (itr == data.headers.end())
        throw std::runtime_error("Unkown header: " + header);
    std::string ret = itr->second.fileHash;
    includes.push_back(header);
    for (const auto& h : itr->second.includes)
        ret += collectIncludes(h, includes);
    return hash_string(ret);
}

static ResourceObject generate_resource_object(const Resources& resources,
                                               const PipelineResourceUsage& usages) {
    ResourceObject ret;
    for (const auto& itr : resources.variables) {
        const std::string& name = itr.first;
        if (usages.usages[ShaderBin::CS].usedVars.count(name))
            ret.computeResources.shaderVars.insert(name);
        if (usages.usages[ShaderBin::VS].usedVars.count(name))
            ret.graphicsResources.shaderVars.insert(name);
        if (usages.usages[ShaderBin::PS].usedVars.count(name))
            ret.graphicsResources.shaderVars.insert(name);
    }
    return ret;
}

struct TypeInfo
{
    std::string glslType;
    std::string cxxType;
    uint32_t size;
    uint32_t align;
};

static TypeInfo get_type_info(const std::string& type) {
#define RET_TYPE(cxxType, align) \
    return { type, #cxxType, sizeof(cxxType), align }
    if (type == "int")
        RET_TYPE(int32_t, 4);
    if (type == "ivec2")
        RET_TYPE(ivec2, 8);
    if (type == "ivec3")
        RET_TYPE(ivec3, 16);
    if (type == "ivec4")
        RET_TYPE(ivec4, 16);
    if (type == "uint")
        RET_TYPE(uint32_t, 4);
    if (type == "uvec2")
        RET_TYPE(uvec2, 8);
    if (type == "uvec3")
        RET_TYPE(uvec3, 16);
    if (type == "uvec4")
        RET_TYPE(uvec4, 16);
    if (type == "float")
        RET_TYPE(float, 4);
    if (type == "vec2")
        RET_TYPE(vec2, 8);
    if (type == "vec3")
        RET_TYPE(vec3, 16);
    if (type == "vec4")
        RET_TYPE(vec4, 16);
    if (type == "mat4")
        RET_TYPE(mat4, 16);
#undef RET_TYPE
    throw std::runtime_error("Unkown type: " + type);
}

PushConstObjData ResourcePack::generateCXX(const std::string& structName,
                                           const Resources& resources, std::ostream& out,
                                           std::vector<PushConstEntry>& pushConstEntries) const {
    PushConstObjData ret;
    ret.name = structName;
    std::vector<std::string> initOrder;
    initOrder.reserve(shaderVars.size());
    std::map<std::string, TypeInfo> vars;
    std::vector<std::string> sizeOrder;
    std::map<std::string, size_t> offsets;
    sizeOrder.reserve(shaderVars.size());
    for (const std::string& var : shaderVars) {
        auto typeItr = resources.variables.find(var);
        if (typeItr == resources.variables.end())
            throw std::runtime_error(
              "Variable registered in resource pack is not found in the resource list");
        vars[var] = get_type_info(typeItr->second);
        sizeOrder.push_back(var);
    }
    std::sort(sizeOrder.begin(), sizeOrder.end(),
              [&](const std::string& lhs, const std::string& rhs) {
                  return vars[lhs].size > vars[rhs].size;
              });
    uint32_t predictedSize = 0;
    uint32_t separatorId = 0;
    auto gen_separator = [&](uint32_t align) {
        uint32_t offset = predictedSize % align;
        if (offset != 0) {
            out << "    // padding of " << (align - offset) << "bytes\n";
            for (uint32_t i = offset; i < align; ++i)
                out << "    uint8_t __sep" << separatorId++ << ";\n";
            predictedSize += align - offset;
        }
    };
    uint32_t firstAlignment = 0;
    auto export_var = [&](auto itr) {
        const std::string& var = *itr;
        if (firstAlignment == 0)
            firstAlignment = vars[var].align;
        gen_separator(vars[var].align);
        out << "    " << vars[var].cxxType << " " << var << "; // offset: " << predictedSize
            << "\n";
        pushConstEntries.push_back({predictedSize, vars[var].glslType, var});
        offsets[var] = predictedSize;
        predictedSize += vars[var].size;
        initOrder.push_back(var);
        sizeOrder.erase(itr);
    };
    constexpr uint32_t structAlignas = 4;
    out << "struct alignas(" << structAlignas << ") " << structName << " {\n";
    while (sizeOrder.size() > 0) {
        uint32_t requiredOffset = vars[sizeOrder[0]].align;
        if (predictedSize % requiredOffset != 0) {
            uint32_t padding = requiredOffset - (predictedSize % requiredOffset);
            auto itr =
              std::find_if(sizeOrder.begin(), sizeOrder.end(),
                           [&](const std::string& var) { return vars[var].size <= padding; });
            if (itr != sizeOrder.end()) {
                export_var(itr);
                continue;
            }
        }
        export_var(sizeOrder.begin());
    }
    if (firstAlignment == 0)
        firstAlignment = 1;
    out << "    // size without padding at the end\n";
    out << "    static constexpr uint32_t CONTENT_SIZE = " << predictedSize << ";\n";
    out << "    // based on the first member\n";
    out << "    static constexpr uint32_t REQUIRED_ALIGNMENT = " << firstAlignment << ";\n";
    ret.effectiveSize = predictedSize;
    ret.structAlignment = firstAlignment;
    gen_separator(structAlignas);
    out << "    " << structName << "(";
    bool first = true;
    for (const auto& [name, type] : resources.variables) {
        if (!first)
            out << ", ";
        first = false;
        if (std::find(initOrder.begin(), initOrder.end(), name) != initOrder.end())
            out << "const " << type << " &_" << name;
        else
            out << "const " << type << " & /*" << name << "*/";
    }
    out << ")\n";
    first = true;
    for (const std::string& var : initOrder) {
        out << (first ? "      : " : "      , ");
        first = false;
        out << var << "(_" << var << ")\n";
    }
    out << "    {\n";
    out << "    }\n";
    out << "};\n";
    for (const auto& [var, offset] : offsets) {
        out << "static_assert(offsetof(" << structName << ", " << var << ") == " << offset
            << ");\n";
    }
    out << "static_assert(sizeof(" << structName << ") == " << predictedSize << ");\n\n";
    ret.structSize = predictedSize;
    return ret;
}

void Preprocessor::processHeader(const fs::path& file, const fs::path& outdir) {
    std::string name = file.stem().string();
    for (char& c : name)
        c = static_cast<char>(tolower(c));
    usedHeaders.insert(name);
    std::ifstream in(file.c_str());
    if (!in.is_open())
        throw std::runtime_error("Could not open shader header file: " + file.string());
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::string hash = hash_string(content);
    content = "";
    in.seekg(0, std::ios::beg);
    std::string headerHash = "";
    std::string cxxHash = "";
    if (auto itr = data.headers.find(name); itr != data.headers.end()) {
        if (itr->second.fileHash == hash) {
            itr->second.filePath = fs::absolute(file).string();
            return;
        }
        headerHash = itr->second.headerHash;
        cxxHash = itr->second.cxxHash;
    }
    std::cout << "Preprocessing shader header '" << name << "' (" << file.string() << ")\n";
    ShaderHeaderData incData;
    incData.name = name;
    incData.filePath = fs::absolute(file).string();
    incData.fileHash = hash;
    incData.headerHash = headerHash;
    incData.cxxHash = cxxHash;

    BlockFile b(in);
    in.close();
    if (b.hasContent())
        throw std::runtime_error("Shader file has content on the root level (no blocks present)");
    size_t descriptorCount = b.getBlockCount("descriptor");
    if (descriptorCount > 1)
        throw std::runtime_error("A shader file may only contain one 'descriptor' block");
    const BlockFile* descBlock = b.getNode("descriptor");
    if (descBlock->hasContent())
        throw std::runtime_error("The descriptor block must not have direct content");
    size_t variantsBlockCount = descBlock->hasNodes() ? descBlock->getBlockCount("variants") : 0;
    size_t resourcesBlockCount = descBlock->hasNodes() ? descBlock->getBlockCount("resources") : 0;
    if (variantsBlockCount > 1 || resourcesBlockCount > 1)
        throw std::runtime_error(
          "The descriptor block can only have up to one variants and resources blocks");
    std::stringstream header;
    std::stringstream cxx;
    if (variantsBlockCount == 1) {
        const BlockFile* variantBlock = descBlock->getNode("variants");
        read_variants(variantBlock, incData.variants);
    }
    if (resourcesBlockCount == 1) {
        const BlockFile* resourcesBlock = descBlock->getNode("resources");
        read_resources(resourcesBlock, incData.resources);
    }

    readIncludes(b, incData.includes);

    const std::string className = "shader_" + name + "_descriptor";
    const std::string registryClassName = "shader_" + name + "_registry";
    fs::path headerFileName = fs::path("shader_header_" + name + ".h");
    fs::path cxxFileName = fs::path("shader_header_" + name + ".cpp");
    incData.descriptorClassName = className;
    incData.descriptorRegistryClassName = registryClassName;
    incData.name = name;
    header << "#pragma once\n\n";
    header << "#include <memory>\n\n";
    header << "#include <shaderdescriptor.h>\n";
    header << "#include <shadertypes.h>\n";
    header << "#include <drvshader.h>\n\n";

    cxx << "#include \"" << headerFileName.string() << "\"\n\n";
    cxx << "#include <drv.h>\n\n";

    uint32_t variantMul = 1;
    for (const auto& [variantName, values] : incData.variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        incData.variantMultiplier[variantName] = variantMul;
        variantMul *= values.size();
    }
    incData.totalVariantMultiplier = variantMul;

    ShaderGenerationInput genInput;
    read_gen_input(b, genInput);

    incData.variantToResourceUsage.resize(incData.totalVariantMultiplier);
    for (uint32_t i = 0; i < incData.totalVariantMultiplier; ++i) {
        PipelineResourceUsage resourceUsage;
        ShaderBin::StageConfig cfg =
          read_stage_configs(incData.resources, i, {incData.variants}, incData.variantMultiplier,
                             genInput, resourceUsage);
        if (incData.resourceObjects.find(resourceUsage) == incData.resourceObjects.end())
            incData.resourceObjects[resourceUsage] =
              generate_resource_object(incData.resources, resourceUsage);
        // const ResourceObject& resourceObj = incData.resourceObjects[resourceUsage];
        incData.variantToResourceUsage[i] = resourceUsage;
    }

    uint32_t structId = 0;
    std::map<std::string, uint32_t> graphicsPushConstStructNameToId;
    std::map<std::string, uint32_t> computePushConstStructNameToId;
    std::vector<std::string> structIdToName;
    for (const auto& itr : incData.resourceObjects) {
        if (itr.second.graphicsResources) {
            // TODO
            std::string structName =
              "PushConstants_header_" + name + "_graphics_" + std::to_string(structId);
            graphicsPushConstStructNameToId[structName] = structId;
            structIdToName.push_back(structName);
            incData.structIdToGlslStructDesc.emplace_back();
            incData.exportedPacks[itr.second.graphicsResources] =
              itr.second.graphicsResources.generateCXX(structName, incData.resources, cxx,
                                                       incData.structIdToGlslStructDesc.back());
            structId++;
        }
        if (itr.second.computeResources) {
            std::string structName =
              "PushConstants_header_" + name + "_compute_" + std::to_string(structId);
            computePushConstStructNameToId[structName] = structId;
            structIdToName.push_back(structName);
            incData.structIdToGlslStructDesc.emplace_back();
            incData.exportedPacks[itr.second.computeResources] =
              itr.second.computeResources.generateCXX(structName, incData.resources, cxx,
                                                      incData.structIdToGlslStructDesc.back());
            structId++;
        }
    }

    incData.localVariantToStructIdGraphics.clear();
    incData.localVariantToStructIdCompute.clear();
    incData.localVariantToStructIdGraphics.reserve(incData.totalVariantMultiplier);
    incData.localVariantToStructIdCompute.reserve(incData.totalVariantMultiplier);
    for (uint32_t i = 0; i < incData.totalVariantMultiplier; ++i) {
        PipelineResourceUsage resourceUsage = incData.variantToResourceUsage[i];
        ResourceObject resObj = incData.resourceObjects[resourceUsage];
        if (auto itr = incData.exportedPacks.find(resObj.graphicsResources);
            itr != incData.exportedPacks.end()) {
            if (auto itr2 = graphicsPushConstStructNameToId.find(itr->second.name);
                itr2 != graphicsPushConstStructNameToId.end())
                incData.localVariantToStructIdGraphics.push_back(itr2->second);
            else
                incData.localVariantToStructIdGraphics.push_back(INVALID_STRUCT_ID);
        }
        else
            incData.localVariantToStructIdGraphics.push_back(INVALID_STRUCT_ID);
        if (auto itr = incData.exportedPacks.find(resObj.computeResources);
            itr != incData.exportedPacks.end()) {
            if (auto itr2 = computePushConstStructNameToId.find(itr->second.name);
                itr2 != computePushConstStructNameToId.end())
                incData.localVariantToStructIdCompute.push_back(itr2->second);
            else
                incData.localVariantToStructIdCompute.push_back(INVALID_STRUCT_ID);
        }
        else
            incData.localVariantToStructIdCompute.push_back(INVALID_STRUCT_ID);
    }

    cxx << "static const uint32_t LOCAL_VARIANT_TO_PUSH_CONST_STRUCT_ID_GRAPHICS[] = {";
    for (uint32_t i = 0; i < incData.totalVariantMultiplier; ++i)
        cxx << incData.localVariantToStructIdGraphics[i] << ", ";
    cxx << "};\n";
    cxx << "static const uint32_t LOCAL_VARIANT_TO_PUSH_CONST_STRUCT_ID_COMPUTE[] = {";
    for (uint32_t i = 0; i < incData.totalVariantMultiplier; ++i)
        cxx << incData.localVariantToStructIdCompute[i] << ", ";
    cxx << "};\n";

    header << "class " << registryClassName << " final : public ShaderDescriptorReg {\n";
    header << "  public:\n";
    header << "    " << registryClassName << "(drv::LogicalDevicePtr device);\n";
    cxx << registryClassName << "::" << registryClassName << "(drv::LogicalDevicePtr device)\n";
    cxx << "  : reg(drv::create_shader_header_registry(device))\n";
    cxx << "{\n";
    cxx << "}\n\n";
    header << "    friend class " << className << ";\n";
    header << "  private:\n";
    header << "    std::unique_ptr<drv::DrvShaderHeaderRegistry> reg;\n";
    header << "};\n\n";

    header << "class " << className << " final : public ShaderDescriptor\n";
    header << "{\n";
    header << "  public:\n";
    header << "    ~" << className << "() override {}\n";
    for (const auto& [variantName, values] : incData.variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        std::string enumName = get_variant_enum_name(variantName);
        header << "    enum class " << enumName << " {\n";
        for (size_t i = 0; i < values.size(); ++i) {
            std::string val = get_variant_enum_value(values[i]);
            header << "        " << val << " = " << i << ",\n";
        }
        header << "    };\n";
        std::string valName = get_variant_enum_val_name(variantName);
        header << "    void setVariant_" << variantName << "(" << enumName << " value);\n";
        cxx << "void " << className << "::setVariant_" << variantName << "(" << enumName
            << " value) {\n";
        cxx << "    variantDesc." << valName << " = value;\n";
        cxx << "}\n";
    }
    header << "    struct VariantDesc {\n";
    for (const auto& [variantName, values] : incData.variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        std::string enumName = get_variant_enum_name(variantName);
        std::string valName = get_variant_enum_val_name(variantName);
        header << "        " << enumName << " " << valName << " = " << enumName
               << "::" << get_variant_enum_value(values[0]) << ";\n";
    }
    header << "        uint32_t getLocalVariantId() const;\n";
    cxx << "uint32_t " << className << "::VariantDesc::getLocalVariantId() const {\n";
    cxx << "    uint32_t ret = 0;\n";
    for (const auto& [variantName, values] : incData.variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        std::string valName = get_variant_enum_val_name(variantName);
        cxx << "    ret += static_cast<uint32_t>(" << valName << ") * "
            << incData.variantMultiplier[variantName] << ";\n";
    }
    cxx << "    return ret;\n";
    cxx << "}\n";
    header << "    };\n";
    for (const auto& [varName, varType] : incData.resources.variables) {
        header << "    " << varType << " " << varName << " = " << varType << "_default_value;\n";
        header << "    void set_" << varName << "(const " << varType << " &_" << varName << ");\n";
        cxx << "void " << className << "::set_" << varName << "(const " << varType << " &_"
            << varName << ") {\n";
        cxx << "    if (" << varName << " != _" << varName << ") {\n";
        cxx << "        " << varName << " = _" << varName << ";\n";
        // cxx << "        if ()\n"; // TODo
        cxx << "            invalidatePushConsts();\n";
        cxx << "    }\n";
        cxx << "}\n";
    }
    header
      << "    void setVariant(const std::string& variantName, const std::string& value) override;\n";
    cxx << "void " << className
        << "::setVariant(const std::string& variantName, const std::string& value) {\n";
    std::string ifString = "if";
    for (const auto& [variantName, values] : incData.variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        std::string enumName = get_variant_enum_name(variantName);
        std::string valName = get_variant_enum_val_name(variantName);
        cxx << "    " << ifString << " (variantName == \"" << variantName << "\") {\n";
        std::string ifString2 = "if";
        for (const std::string& variantVal : values) {
            const std::string val = get_variant_enum_value(variantVal);
            cxx << "        " << ifString2 << " (value == \"" << variantVal << "\")\n";
            cxx << "            variantDesc." << valName << " = " << enumName << "::" << val
                << ";\n";
            ifString2 = "else if";
        }
        if (ifString2 != "if")
            cxx << "        else\n    ";
        cxx
          << "        throw std::runtime_error(\"Unknown value (\" + value + \") for shader variant param: "
          << variantName << "\");\n";
        cxx << "    }";
        ifString = " else if";
    }
    if (ifString != "if")
        cxx << " else\n    ";
    else
        cxx << "\n";
    cxx << "    throw std::runtime_error(\"Unknown variant param: \" + variantName);\n";
    cxx << "}\n";
    header << "    void setVariant(const std::string& variantName, int value) override;\n";
    cxx << "void " << className << "::setVariant(const std::string& variantName, int value) {\n";
    ifString = "if";
    for (const auto& [variantName, values] : incData.variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        std::string enumName = get_variant_enum_name(variantName);
        std::string valName = get_variant_enum_val_name(variantName);
        cxx << "    " << ifString << " (variantName == \"" << variantName << "\")\n";
        cxx << "        variantDesc." << valName << " = static_cast<" << enumName << ">(value);\n";
        ifString = "else if";
    }
    if (ifString != "if")
        cxx << "    else\n    ";
    cxx << "    throw std::runtime_error(\"Unknown variant param: \" + variantName);\n";
    cxx << "}\n";
    header << "    std::vector<std::string> getVariantParamNames() const override;\n";
    cxx << "std::vector<std::string> " << className << "::getVariantParamNames() const {\n";
    cxx << "    return {\n";
    for (const auto& [variantName, values] : incData.variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        cxx << "        \"" << variantName << "\",\n";
    }
    cxx << "    };\n";
    cxx << "}\n";
    header << "    const VariantDesc &getVariantDesc() const { return variantDesc; }\n";
    header << "    uint32_t getLocalVariantId() const override;\n";
    cxx << "uint32_t " << className << "::getLocalVariantId() const {\n";
    cxx << "    return variantDesc.getLocalVariantId();\n";
    cxx << "}\n";
    // std::map<std::string, uint32_t> graphicsPushConstStructNameToId;
    // std::map<std::string, uint32_t> computePushConstStructNameToId;
    // incData.exportedPacks[itr.second.graphicsResources]
    header << "    bool hasPushConstsGraphics() const override;\n";
    cxx << "bool " << className << "::hasPushConstsGraphics() const {\n";
    cxx << "    return LOCAL_VARIANT_TO_PUSH_CONST_STRUCT_ID_GRAPHICS[getLocalVariantId()] != "
        << INVALID_STRUCT_ID << ";\n";
    cxx << "}\n";
    header << "    uint32_t getPushConstStructIdGraphics() const override;\n";
    cxx << "uint32_t " << className << "::getPushConstStructIdGraphics() const {\n";
    cxx << "    return LOCAL_VARIANT_TO_PUSH_CONST_STRUCT_ID_GRAPHICS[getLocalVariantId()];\n";
    cxx << "}\n";
    header << "    uint32_t getPushConstStructIdCompute() const override;\n";
    cxx << "uint32_t " << className << "::getPushConstStructIdCompute() const {\n";
    cxx << "    return LOCAL_VARIANT_TO_PUSH_CONST_STRUCT_ID_COMPUTE[getLocalVariantId()];\n";
    cxx << "}\n";
    header << "    void pushGraphicsConsts(void *dst) const override;\n";
    cxx << "void " << className << "::pushGraphicsConsts(void *dst) const {\n";
    cxx << "    switch (getPushConstStructIdGraphics()) {\n";
    for (const auto& pack : incData.exportedPacks) {
        auto idItr = graphicsPushConstStructNameToId.find(pack.second.name);
        if (idItr == graphicsPushConstStructNameToId.end())
            continue;
        cxx << "      case " << idItr->second << ": {\n";
        cxx << "        " << pack.second.name << " pack(";
        bool first = true;
        for (const auto& [name, type] : incData.resources.variables) {
            if (!first)
                cxx << ", ";
            cxx << name;
            first = false;
        }
        cxx << ");\n";
        cxx << "        std::memcpy(dst, &pack, " << pack.second.name << "::CONTENT_SIZE);\n";
        cxx << "        break;\n";
        cxx << "      }\n";
    }
    cxx << "      default:\n";
    cxx << "        throw std::runtime_error(\"Invalid push const struct id\");\n";
    cxx << "    }\n";
    cxx << "}\n";
    header << "    " << className << "(drv::LogicalDevicePtr device, const " << registryClassName
           << " *_reg);\n";
    cxx << className << "::" << className << "(drv::LogicalDevicePtr device, const "
        << registryClassName << " *_reg)\n";
    cxx << "  : ShaderDescriptor(\"" << name << "\")\n";
    cxx << "  , reg(_reg)\n";
    cxx << "  , header(drv::create_shader_header(device, reg->reg.get()))\n";
    cxx << "{\n";
    cxx << "}\n";
    header << "    const ShaderDescriptorReg* getReg() const override { return reg; }\n";
    header << "  private:\n";
    header << "    VariantDesc variantDesc;\n";
    header << "    const " << registryClassName << " *reg;\n";
    header << "    std::unique_ptr<drv::DrvShaderHeader> header;\n";
    header << "};\n";

    incData.headerFileName = headerFileName.string();
    const std::string h = hash_string(header.str());
    if (headerHash != h) {
        fs::path headerFilePath = outdir / headerFileName;
        std::cout << "Generating " << headerFilePath << std::endl;
        std::ofstream outHeaderFile(headerFilePath.string());
        if (!outHeaderFile.is_open())
            throw std::runtime_error("Could not open output file: " + headerFileName.string());
        incData.headerHash = h;
        changedAnyCppHeader = true;
        outHeaderFile << header.str();
    }
    const std::string ch = hash_string(cxx.str());
    if (cxxHash != ch) {
        fs::path cxxFilePath = outdir / cxxFileName;
        std::cout << "Generating " << cxxFilePath << std::endl;
        std::ofstream outCxxFile(cxxFilePath.string());
        if (!outCxxFile.is_open())
            throw std::runtime_error("Could not open output file: " + cxxFileName.string());
        outCxxFile << cxx.str();
        incData.cxxHash = ch;
    }
    data.headers[name] = std::move(incData);
}

std::map<std::string, uint32_t> Preprocessor::getHeaderLocalVariants(
  uint32_t variantId, const ShaderObjectData& objData) const {
    std::map<std::string, uint32_t> ret;
    for (const auto& header : objData.allIncludes)
        ret[header] = (variantId / objData.headerVariantIdMultiplier.find(header)->second)
                      % data.headers.find(header)->second.totalVariantMultiplier;
    return ret;
}

// std::vector<PipelineResourceUsage> Preprocessor::generateShaderVariantToResourceUsages(
//   const ShaderObjectData& objData) const {
//     std::vector<PipelineResourceUsage> ret;
//     ret.reserve(objData.variantCount);
//     for (uint32_t i = 0; i < objData.variantCount; ++i) {
//         TODO;
//     }
//     return ret;
// }

void Preprocessor::processSource(const fs::path& file, const fs::path& outdir,
                                 const drv::DeviceLimits& drvLimits) {
    std::string name = file.stem().string();
    for (char& c : name)
        c = static_cast<char>(tolower(c));
    usedShaders.insert(name);
    std::ifstream in(file.c_str());
    if (!in.is_open())
        throw std::runtime_error("Could not open shader file: " + file.string());
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::string hash = hash_string(content);
    content = "";
    in.seekg(0, std::ios::beg);
    std::string headerHash = "";
    std::string cxxHash = "";
    if (auto itr = data.sources.find(name); itr != data.sources.end()) {
        if (itr->second.fileHash == hash) {
            std::vector<std::string> includes;
            if (collectIncludes(name, includes) == itr->second.headersHash) {
                itr->second.filePath = fs::absolute(file).string();
                return;
            }
        }
        headerHash = itr->second.headerHash;
        cxxHash = itr->second.cxxHash;
    }
    std::cout << "Preprocessing shader '" << name << "' (" << file.string() << ")\n";
    ShaderObjectData objData;
    objData.name = name;
    objData.filePath = fs::absolute(file).string();
    objData.fileHash = hash;
    objData.headerHash = headerHash;
    objData.cxxHash = cxxHash;
    objData.headersHash = collectIncludes(name, objData.allIncludes);
    std::reverse(objData.allIncludes.begin(), objData.allIncludes.end());
    objData.headerLocations.clear();
    for (const auto& header : objData.allIncludes)
        objData.headerLocations[header] = data.headers.find(header)->second.filePath;

    BlockFile b(in);
    in.close();
    if (b.hasContent())
        throw std::runtime_error("Shader file has content on the root level (no blocks present)");

    std::stringstream cu;
    std::stringstream header;
    std::stringstream cxx;
    fs::path headerFileName = fs::path("shader_" + name + ".h");
    fs::path cxxFileName = fs::path("shader_" + name + ".cpp");
    objData.headerFileName = headerFileName.string();
    header << "#pragma once\n\n";
    header << "#include <cstddef>\n";
    header << "#include <drvshader.h>\n";
    header << "#include <shaderbin.h>\n";
    header << "#include <shaderobject.h>\n";
    header << "#include <drvrenderpass.h>\n";
    header << "#include <shaderobjectregistry.h>\n\n";
    cxx << "#include \"" << headerFileName.string() << "\"\n\n";
    cxx << "#include <drv.h>\n";
    objData.includeHeaders(cu);
    for (const auto& h : objData.allIncludes) {
        auto itr = data.headers.find(h);
        assert(itr != data.headers.end());
        header << "#include \"" << itr->second.headerFileName << "\"\n";
    }
    BlockFile cuBlocks(cu, false);
    if (!cuBlocks.hasNodes())
        throw std::runtime_error("Compilation unit doesn't have any blocks");
    ShaderGenerationInput genInput;
    read_gen_input(cuBlocks, genInput);

    //     directIncludes.push_back(shaderName);  // include it's own header as well
    //     std::vector<std::string> allIncludes;
    // std::unordered_map<std::string, std::string> variantParamToDescriptor;
    uint32_t variantIdMul = 1;
    objData.variants.clear();
    objData.resources = {};
    for (const auto& h : objData.allIncludes) {
        auto itr = data.headers.find(h);
        if (itr == data.headers.end())
            throw std::runtime_error("Unknown shader header: " + h);
        objData.headerVariantIdMultiplier[h] = variantIdMul;
        for (const auto& [name, value] : itr->second.variantMultiplier)
            objData.variantIdMultiplier[name] = value * objData.headerVariantIdMultiplier[h];
        variantIdMul *= itr->second.totalVariantMultiplier;
        objData.variants.push_back(itr->second.variants);
        // for (const auto& itr2 : itr->second.variantMultiplier) {
        //     if (variantParamToDescriptor.find(itr2.first) != variantParamToDescriptor.end())
        //         throw std::runtime_error("Variant param names must be unique");
        //     variantParamToDescriptor[itr2.first] = inc;
        // }
        objData.resources += itr->second.resources;
    }
    objData.variantCount = variantIdMul;
    // objData.stageConfigs = ShaderBin::StageConfig cfg = read_stage_configs(
    //   resources, i, {incData.variants}, incData.variantMultiplier, genInput, resourceUsage);

    //     std::vector<Variants> variants;
    //     size_t descriptorCount = cuBlocks.getBlockCount("descriptor");
    //     Resources resources;
    //     std::unordered_map<std::string, uint32_t> variantParamMultiplier;
    //     for (size_t i = 0; i < descriptorCount; ++i) {
    //         const BlockFile* descriptor = cuBlocks.getNode("descriptor", i);
    //         if (descriptor->getBlockCount("variants") == 1) {
    //             Variants v;
    //             read_variants(descriptor->getNode("variants"), v);
    //             for (const auto& [name, values] : v.values) {
    //                 auto desc = variantParamToDescriptor.find(name);
    //                 assert(desc != variantParamToDescriptor.end());
    //                 auto descMulItr = variantIdMultiplier.find(desc->second);
    //                 assert(descMulItr != variantIdMultiplier.end());
    //                 auto inc = compileData.includeData.find(desc->second);
    //                 assert(inc != compileData.includeData.end());
    //                 auto variantMul = inc->second.variantMultiplier.find(name);
    //                 assert(variantMul != inc->second.variantMultiplier.end());
    //                 variantParamMultiplier[name] =
    //                   safe_cast<uint32_t>(descMulItr->second * variantMul->second);
    //             }
    //             variants.push_back(std::move(v));
    //         }
    //         if (descriptor->getBlockCount("resources") == 1) {
    //             const BlockFile* resourcesBlock = descriptor->getNode("resources");
    //             if (!read_resources(resourcesBlock, resources)) {
    //                 std::cerr << "Could not read resources: " << shaderFile << std::endl;
    //                 return false;
    //             }
    //         }
    //     }
    //     ShaderBin::ShaderData shaderData;
    std::map<PipelineResourceUsage, ResourceObject> resourceObjects;
    // std::vector<PipelineResourceUsage> variantToResourceUsage =
    //   generateShaderVariantToResourceUsages(objData);
    //     std::string genFile = "";
    //     if (compileData.genFolder != "")
    //         genFile = (fs::path{compileData.genFolder} / fs::path{shaderName + ".glsl"}).string();
    //     if (!generate_binary(compileData.debugPath, compileData.compiler, resources, shaderData,
    //                          variants, std::move(genInput), variantParamMultiplier, resourceObjects,
    //                          variantToResourceUsage, genFile)) {
    //         std::cerr << "Could not generate binary: " << shaderFile << std::endl;
    //         return false;
    //     }

    //     compileData.shaderBin->addShader(shaderName, std::move(shaderData));

    const std::string className = "shader_" + name;
    const std::string registryClassName = "shader_obj_registry_" + name;
    objData.className = className;
    objData.registryClassName = registryClassName;

    header << "\n";

    // uint32_t structId = 0;
    // std::map<ResourcePack, std::string> exportedPacks;
    // for (const auto& itr : resourceObjects) {
    //     for (const auto& [stages, pack] : itr.second.packs) {
    //         if (exportedPacks.find(pack) != exportedPacks.end())
    //             continue;
    //         std::string structName = "PushConstants_" + name + "_" + std::to_string(structId++);
    //         exportedPacks[pack] = structName;
    //         pack.generateCXX(structName, objData.resources, cxx);
    //     }
    // }

    header << "class " << registryClassName << " final : public ShaderObjectRegistry {\n";
    header << "  public:\n";
    header << "    " << registryClassName
           << "(drv::LogicalDevicePtr device, const ShaderBin &shaderBin";
    cxx << registryClassName << "::" << registryClassName
        << "(drv::LogicalDevicePtr device, const ShaderBin &shaderBin";
    for (const std::string& inc : objData.allIncludes) {
        auto itr = data.headers.find(inc);
        assert(itr != data.headers.end());
        header << ", const " << itr->second.descriptorRegistryClassName << " *reg_"
               << itr->second.name;
        cxx << ", const " << itr->second.descriptorRegistryClassName << " *reg_"
            << itr->second.name;
    }
    header << ");\n";
    cxx << ")\n";
    cxx << "  : ShaderObjectRegistry(device)\n";
    cxx << "{\n";
    cxx << "    const ShaderBin::ShaderData *shader = shaderBin.getShader(\"" << name << "\");\n";
    cxx << "    if (shader == nullptr)\n";
    cxx << "        throw std::runtime_error(\"Shader not found: " << name << "\");\n";
    cxx << "    loadShader(shaderBin, *shader);\n";

    struct PipelineData
    {
        std::map<std::string, ResourceObject> headerToResObj;
        bool operator<(const PipelineData& rhs) const {
            if (headerToResObj.size() != rhs.headerToResObj.size())
                return headerToResObj.size() < rhs.headerToResObj.size();
            {
                auto itr1 = headerToResObj.begin();
                auto itr2 = rhs.headerToResObj.begin();
                while (itr1 != headerToResObj.end()) {
                    if (itr1->first != itr2->first)
                        return itr1->first < itr2->first;
                    itr1++;
                    itr2++;
                }
            }
            {
                auto itr1 = headerToResObj.begin();
                auto itr2 = rhs.headerToResObj.begin();
                while (itr1 != headerToResObj.end()) {
                    if (itr1->second != itr2->second)
                        return itr1->second < itr2->second;
                    itr1++;
                    itr2++;
                }
            }
            return false;
        }
    };

    std::vector<std::string> resourceShaderOrder;
    resourceShaderOrder.reserve(objData.allIncludes.size());
    // TODO this is just a dummy shader order
    // usage statistics would work best here (probably)
    // Headers append their resources in this order
    for (auto itr = objData.allIncludes.rbegin(); itr != objData.allIncludes.rend(); ++itr)
        resourceShaderOrder.push_back(*itr);

    std::map<PipelineData, uint32_t> resourceUsageToConfigId;
    objData.variantToConfigId.reserve(objData.variantCount);
    uint32_t configId = 0;
    for (uint32_t shaderVariant = 0; shaderVariant < objData.variantCount; ++shaderVariant) {
        PipelineData pipelineData;
        for (const auto& h : objData.allIncludes) {
            auto itr = data.headers.find(h);
            assert(itr != data.headers.end());
            uint32_t headerVariant = (shaderVariant / objData.headerVariantIdMultiplier[h])
                                     % itr->second.totalVariantMultiplier;
            PipelineResourceUsage headerResourceUsage =
              itr->second.variantToResourceUsage[headerVariant];
            ResourceObject headerResObj = itr->second.resourceObjects[headerResourceUsage];
            pipelineData.headerToResObj[h] = headerResObj;
        }
        if (auto itr = resourceUsageToConfigId.find(pipelineData);
            itr != resourceUsageToConfigId.end()) {
            objData.variantToConfigId.push_back(itr->second);
            continue;
        }

        uint32_t computeSize = 0;
        uint32_t graphicsSize = 0;
        for (const auto& h : resourceShaderOrder) {
            auto itr = data.headers.find(h);
            assert(itr != data.headers.end());
            const ResourceObject& resUsage = pipelineData.headerToResObj[h];
            ShaderHeaderResInfo graphicsInfo;
            ShaderHeaderResInfo computeInfo;

            if (resUsage.graphicsResources) {
                const PushConstObjData& pushConstData =
                  itr->second.exportedPacks.find(resUsage.graphicsResources)->second;
                uint32_t size = pushConstData.effectiveSize;
                uint32_t structAlignment = pushConstData.structAlignment;
                if ((graphicsSize % structAlignment) != 0)
                    graphicsSize += structAlignment - (graphicsSize % structAlignment);
                graphicsInfo.pushConstOffset = graphicsSize;
                graphicsInfo.pushConstSize = size;
                graphicsSize += size;
            }
            objData.headerToConfigToResinfosGraphics[h].push_back(graphicsInfo);
            if (resUsage.computeResources) {
                const PushConstObjData& pushConstData =
                  itr->second.exportedPacks.find(resUsage.computeResources)->second;
                uint32_t size = pushConstData.effectiveSize;
                uint32_t structAlignment = pushConstData.structAlignment;
                if ((computeSize % structAlignment) != 0)
                    computeSize += structAlignment - (computeSize % structAlignment);
                computeInfo.pushConstOffset = computeSize;
                computeInfo.pushConstSize = size;
                computeSize += size;
            }
            objData.headerToConfigToResinfosCompute[h].push_back(computeInfo);
        }
        uint32_t rangeCount = 0;
        if (computeSize)
            rangeCount++;
        if (graphicsSize)
            rangeCount++;
        if (graphicsSize > drvLimits.maxPushConstantsSize)
            throw std::runtime_error(
              "The graphics pipeline has exceeded the push const range size limit ("
              + std::to_string(graphicsSize) + " > "
              + std::to_string(drvLimits.maxPushConstantsSize) + ")");
        if (computeSize > drvLimits.maxPushConstantsSize)
            throw std::runtime_error(
              "The compute pipeline has exceeded the push const range size limit ("
              + std::to_string(computeSize) + " > " + std::to_string(drvLimits.maxPushConstantsSize)
              + ")");

        cxx << "    {\n";
        cxx << "        drv::DrvShaderObjectRegistry::PushConstantRange ranges[" << rangeCount
            << "];\n";
        uint32_t rangeId = 0;
        if (graphicsSize) {
            cxx << "        ranges[" << rangeId
                << "].stages = drv::ShaderStage::VERTEX_BIT | drv::ShaderStage::FRAGMENT_BIT;\n";
            cxx << "        ranges[" << rangeId << "].offset = 0;\n";
            cxx << "        ranges[" << rangeId << "].size = " << graphicsSize << ";\n";
            rangeId++;
        }
        if (computeSize) {
            cxx << "        ranges[" << rangeId << "].stages = drv::ShaderStage::COMPUTE_BIT;\n";
            cxx << "        ranges[" << rangeId << "].offset = 0;\n";
            cxx << "        ranges[" << rangeId << "].size = " << computeSize << ";\n";
            rangeId++;
        }
        cxx << "        drv::DrvShaderObjectRegistry::ConfigInfo config;\n";
        cxx << "        config.numRanges = " << rangeId << ";\n";
        cxx << "        config.ranges = ranges;\n";
        cxx << "        reg->addConfig(std::move(config));\n";
        cxx << "    }\n";
        resourceUsageToConfigId[pipelineData] = configId;
        objData.variantToConfigId.push_back(configId++);
    }
    cxx << "}\n\n";

    cxx << "static uint32_t CONFIG_INDEX[] = {";
    for (uint32_t i = 0; i < objData.variantCount; ++i)
        cxx << objData.variantToConfigId[i] << ", ";
    cxx << "};\n\n";
    for (const auto& [header, infos] : objData.headerToConfigToResinfosGraphics) {
        cxx << "static ShaderHeaderResInfo CONFIG_TO_RES_GRAPHICS_" << header << "[] = {";
        for (const auto& info : infos)
            cxx << "{" << info.pushConstOffset << ", " << info.pushConstSize << "}, ";
        cxx << "};\n\n";
    }
    for (const auto& [header, infos] : objData.headerToConfigToResinfosCompute) {
        cxx << "static ShaderHeaderResInfo CONFIG_TO_RES_COMPUTE_" << header << "[] = {";
        for (const auto& info : infos)
            cxx << "{" << info.pushConstOffset << ", " << info.pushConstSize << "}, ";
        cxx << "};\n\n";
    }

    header << "    static VariantId get_variant_id(";
    cxx << "ShaderObjectRegistry::VariantId " << registryClassName << "::get_variant_id(";
    bool first = true;
    for (const std::string& inc : objData.allIncludes) {
        auto itr = data.headers.find(inc);
        assert(itr != data.headers.end());
        if (!first) {
            header << ", ";
            cxx << ", ";
        }
        else
            first = false;
        header << "const " << itr->second.descriptorClassName << "::VariantDesc &"
               << itr->second.name;
        cxx << "const " << itr->second.descriptorClassName << "::VariantDesc &" << itr->second.name;
    }
    header << ");\n";
    cxx << ") {\n";
    cxx << "    uint32_t ret = 0;\n";
    for (const std::string& inc : objData.allIncludes) {
        auto itr = data.headers.find(inc);
        assert(itr != data.headers.end());
        auto mulItr = objData.headerVariantIdMultiplier.find(inc);
        assert(mulItr != objData.headerVariantIdMultiplier.end());
        cxx << "    ret += " << itr->second.name << ".getLocalVariantId() * " << mulItr->second
            << ";\n";
    }
    cxx << "    return ret;\n";
    cxx << "}\n";
    header << "    static uint32_t get_config_id(ShaderObjectRegistry::VariantId variantId);\n";
    cxx << "uint32_t " << registryClassName
        << "::get_config_id(ShaderObjectRegistry::VariantId variantId) {\n";
    cxx << "    return CONFIG_INDEX[variantId];\n";
    cxx << "}\n";
    header
      << "    drv::PipelineLayoutPtr getPipelineLayout(ShaderObjectRegistry::VariantId variantId) const override;";
    cxx << "drv::PipelineLayoutPtr " << registryClassName
        << "::getPipelineLayout(ShaderObjectRegistry::VariantId variantId) const {\n";
    cxx << "    uint32_t configId = CONFIG_INDEX[variantId];\n";
    cxx << "    return reg->getPipelineLayout(configId);\n";
    cxx << "}\n";
    header << "    friend class " << className << ";\n";
    header << "  protected:\n";
    // shaderObj
    //   << "    VariantId getShaderVariant(const ShaderDescriptorCollection* descriptors) const override {\n";
    // shaderObj
    //   << "        return static_cast<const Descriptor*>(descriptors)->getLocalVariantId();\n";
    // shaderObj << "    }\n";
    header << "};\n\n";

    header << "class " << className << " final : public ShaderObject {\n";
    header << "  public:\n";
    header << "    using Registry = " << registryClassName << ";\n";
    header << "    " << className << "(drv::LogicalDevicePtr device, const " << registryClassName
           << " *reg, drv::DrvShader::DynamicStates dynamicStates);\n";
    cxx << className << "::" << className << "(drv::LogicalDevicePtr _device, const "
        << registryClassName << " *_reg, drv::DrvShader::DynamicStates dynamicStates)\n";
    cxx << "  : ShaderObject(_device, _reg, \"" << name << "\", std::move(dynamicStates))\n";
    cxx << "{\n";
    cxx << "}\n\n";
    header << "    ~" << className << "() override {}\n";
    header
      << "    uint32_t prepareGraphicalPipeline(const drv::RenderPass *renderPass, drv::SubpassId subpass, const DynamicState &dynamicStates";
    cxx
      << "uint32_t " << className
      << "::prepareGraphicalPipeline(const drv::RenderPass *renderPass, drv::SubpassId subpass, const DynamicState &dynamicStates";
    std::stringstream variantIdInput;
    first = true;
    for (const std::string& inc : objData.allIncludes) {
        auto itr = data.headers.find(inc);
        assert(itr != data.headers.end());
        header << ", const " << itr->second.descriptorClassName << "::VariantDesc &"
               << itr->second.name;
        cxx << ", const " << itr->second.descriptorClassName << "::VariantDesc &"
            << itr->second.name;
        if (!first)
            variantIdInput << ", ";
        else
            first = false;
        variantIdInput << itr->second.name;
    }
    header << ", const GraphicsPipelineStates &overrideStates = {});\n";
    cxx << ", const GraphicsPipelineStates &overrideStates) {\n";
    cxx << "    GraphicsPipelineDescriptor desc;\n";
    cxx << "    desc.renderPass = renderPass;\n";
    cxx << "    desc.subpass = subpass;\n";
    cxx << "    desc.variantId = static_cast<const " << registryClassName
        << "*>(reg)->get_variant_id(" << variantIdInput.str() << ");\n";
    cxx << "    desc.configIndex = static_cast<const " << registryClassName
        << "*>(reg)->get_config_id(desc.variantId);\n";
    cxx << "    desc.states = overrideStates;\n";
    cxx << "    desc.dynamicStates = dynamicStates;\n";
    cxx << "    return getGraphicsPipeline(desc);\n";
    cxx << "}\n";
    for (const std::string& inc : objData.allIncludes) {
        auto itr = data.headers.find(inc);
        assert(itr != data.headers.end());
        header
          << "    static ShaderHeaderResInfo getGraphicsResInfo(ShaderObjectRegistry::VariantId variantId, const "
          << itr->second.descriptorClassName << " *" << itr->second.name << ");\n";
        cxx << "ShaderHeaderResInfo " << className
            << "::getGraphicsResInfo(ShaderObjectRegistry::VariantId variantId, const "
            << itr->second.descriptorClassName << " *" << itr->second.name << ") {\n";
        cxx << "    uint32_t configId = CONFIG_INDEX[variantId];\n";
        cxx << "    return CONFIG_TO_RES_GRAPHICS_" << inc << "[configId];\n";
        cxx << "}\n";
        header
          << "    static ShaderHeaderResInfo getComputeResInfo(ShaderObjectRegistry::VariantId variantId, const "
          << itr->second.descriptorClassName << " *" << itr->second.name << ");\n";
        cxx << "ShaderHeaderResInfo " << className
            << "::getComputeResInfo(ShaderObjectRegistry::VariantId variantId, const "
            << itr->second.descriptorClassName << " *" << itr->second.name << ") {\n";
        cxx << "    uint32_t configId = CONFIG_INDEX[variantId];\n";
        cxx << "    return CONFIG_TO_RES_COMPUTE_" << inc << "[configId];\n";
        cxx << "}\n";
    }
    header
      << "    void bindGraphicsInfo(drv::CmdRenderPass &renderPass, const DynamicState &dynamicStates";
    cxx << "void " << className
        << "::bindGraphicsInfo(drv::CmdRenderPass &renderPass, const DynamicState &dynamicStates";
    std::stringstream pipelineInput;
    for (const std::string& inc : objData.allIncludes) {
        auto itr = data.headers.find(inc);
        assert(itr != data.headers.end());
        header << ", const " << itr->second.descriptorClassName << " *" << itr->second.name;
        cxx << ", const " << itr->second.descriptorClassName << " *" << itr->second.name;
        pipelineInput << "        " << itr->second.name << "->getVariantDesc(),\n";
    }
    header << ", const GraphicsPipelineStates &overrideStates = {});\n";
    cxx << ", const GraphicsPipelineStates &overrideStates) {\n";
    cxx
      << "    uint32_t pipelineId = prepareGraphicalPipeline(renderPass.getRenderPass(), renderPass.getSubpass(), dynamicStates,\n"
      << pipelineInput.str() << "        overrideStates);\n";
    cxx << "    drv::GraphicsPipelineBindInfo info;\n";
    cxx << "    info.shader = getShader();\n";
    cxx << "    info.pipelineId = pipelineId;\n";
    cxx << "    renderPass.bindGraphicsPipeline(info);\n";
    cxx << "}\n";
    header
      << "    drv::PipelineLayoutPtr getPipelineLayout(ShaderObjectRegistry::VariantId variantId) const override;";
    cxx << "drv::PipelineLayoutPtr " << className
        << "::getPipelineLayout(ShaderObjectRegistry::VariantId variantId) const {\n";
    cxx << "    return reg->getPipelineLayout(variantId);\n";
    cxx << "}\n";
    header << "};\n";

    objData.headerFileName = headerFileName.string();
    const std::string h = hash_string(header.str());
    if (headerHash != h) {
        fs::path headerFilePath = outdir / headerFileName;
        std::cout << "Generating " << headerFilePath << std::endl;
        std::ofstream outHeaderFile(headerFilePath.string());
        if (!outHeaderFile.is_open())
            throw std::runtime_error("Could not open output file: " + headerFileName.string());
        objData.headerHash = h;
        changedAnyCppHeader = true;
        outHeaderFile << header.str();
    }
    const std::string ch = hash_string(cxx.str());
    if (cxxHash != ch) {
        fs::path cxxFilePath = outdir / cxxFileName;
        std::cout << "Generating " << cxxFilePath << std::endl;
        std::ofstream outCxxFile(cxxFilePath.string());
        if (!outCxxFile.is_open())
            throw std::runtime_error("Could not open output file: " + cxxFileName.string());
        outCxxFile << cxx.str();
        objData.cxxHash = ch;
    }
    data.sources[name] = std::move(objData);
}

void Preprocessor::generateRegistryFile(const fs::path& file) const {
    if (!changedAnyCppHeader && fs::exists(file))
        return;
    std::cout << "Generating " << file.string() << std::endl;
    if (!fs::exists(file))
        fs::create_directories(file.parent_path());
    std::ofstream reg(file.c_str());
    if (!reg.is_open())
        throw std::runtime_error("Could not open file: " + file.string());

    reg << "#pragma once\n\n";
    reg << "#include <drvtypes.h>\n";
    reg << "#include <shaderbin.h>\n";
    for (const auto& itr : data.headers)
        reg << "#include <" << itr.second.headerFileName << ">\n";
    for (const auto& itr : data.sources)
        reg << "#include <" << itr.second.headerFileName << ">\n";
    reg << "\n";
    reg << "struct ShaderHeaderRegistry {\n";
    reg << "    ShaderHeaderRegistry(const ShaderHeaderRegistry&) = delete;\n";
    reg << "    ShaderHeaderRegistry& operator=(const ShaderHeaderRegistry&) = delete;\n";
    for (const auto& itr : data.headers)
        reg << "    " << itr.second.descriptorRegistryClassName << " " << itr.second.name << ";\n";
    reg << "    ShaderHeaderRegistry(drv::LogicalDevicePtr device)\n";
    bool firstHeader = true;
    for (const auto& itr : data.headers) {
        reg << "      " << (firstHeader ? ':' : ',') << " " << itr.second.name << "(device)\n";
        firstHeader = false;
    }
    reg << "    {\n    }\n";
    reg << "};\n\n";
    reg << "struct ShaderObjRegistry {\n";
    reg << "    ShaderObjRegistry(const ShaderObjRegistry&) = delete;\n";
    reg << "    ShaderObjRegistry& operator=(const ShaderObjRegistry&) = delete;\n";
    for (const auto& itr : data.sources)
        reg << "    " << itr.second.registryClassName << " " << itr.second.name << ";\n";

    reg
      << "    ShaderObjRegistry(drv::LogicalDevicePtr device, const ShaderBin &shaderBin, const ShaderHeaderRegistry& headers)\n";
    bool firstSource = true;
    for (const auto& itr : data.sources) {
        reg << "      " << (firstSource ? ':' : ',') << " " << itr.second.name
            << "(device, shaderBin";
        for (const std::string& inc : itr.second.allIncludes) {
            auto header = data.headers.find(inc);
            assert(header != data.headers.end());
            reg << ", &headers." << header->second.name;
        }
        reg << ")\n";
        firstSource = false;
    }
    reg << "    {\n    }\n";
    reg << "};\n";
}

void Preprocessor::cleanUp() {
    std::unordered_set<std::string> unusedHeaders;
    std::unordered_set<std::string> unusedShaders;
    for (const auto& itr : data.headers)
        if (usedHeaders.count(itr.second.name) == 0)
            unusedHeaders.insert(itr.first);
    for (const auto& itr : data.sources)
        if (usedShaders.count(itr.second.name) == 0)
            unusedShaders.insert(itr.first);

    for (const auto& itr : unusedHeaders)
        data.headers.erase(data.headers.find(itr));
    for (const auto& itr : unusedShaders)
        data.sources.erase(data.sources.find(itr));
}
