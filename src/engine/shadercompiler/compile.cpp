#include "compile.h"

#include <cassert>
#include <cctype>
#include <fstream>
#include <iterator>
#include <set>

#include <HashLib4CPP.h>
#include <simplecpp.h>

#include <blockfile.h>
#include <features.h>
#include <shadertypes.h>
#include <uncomment.h>
#include <util.hpp>

#include "spirvcompiler.h"

namespace fs = std::filesystem;

using ShaderHash = std::string;

static ShaderHash hash_string(const std::string& data) {
    IHash hash = HashLib4CPP::Hash128::CreateMurmurHash3_x64_128();
    IHashResult res = hash->ComputeString(data);
    return res->ToString();
}

static ShaderHash hash_code(const std::string& data) {
    return hash_string(data);
}

static ShaderHash hash_binary(size_t len, const uint32_t* data) {
    IHash hash = HashLib4CPP::Hash128::CreateMurmurHash3_x64_128();
    IHashResult res = hash->ComputeUntyped(data, static_cast<int64_t>(len * sizeof(data[0])));
    return res->ToString();
}

static bool include_headers(const std::string& filename, std::ostream& out,
                            std::set<std::string>& includes, std::set<std::string>& filesInProgress,
                            std::unordered_map<std::string, IncludeData>& includeData,
                            std::vector<std::string>& directIncludes) {
    if (filesInProgress.count(filename) != 0) {
        std::cerr << "File recursively included: " << filename << std::endl;
        return false;
    }
    if (includes.count(filename) > 0)
        return true;
    // directIncludes.clear();
    includes.insert(filename);
    std::ifstream in(filename.c_str());
    if (!in.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }
    std::stringstream content;
    uncomment(in, content);
    std::string contentStr = content.str();
    filesInProgress.insert(filename);
    bool ret = true;
    std::regex headerReg{"((\\w+\\/)*(\\w+))"};
    try {
        BlockFile blocks(content);
        if (!blocks.hasNodes()) {
            std::cerr << "Shader must only contian blocks" << std::endl;
            ret = false;
        }
        else {
            for (size_t i = 0; i < blocks.getBlockCount("include") && ret; ++i) {
                const BlockFile* inc = blocks.getNode("include", i);
                if (!inc->hasContent()) {
                    std::cerr << "Invalid include block in file: " << filename << std::endl;
                    ret = false;
                    break;
                }
                const std::string* headerContent = inc->getContent();
                auto headersBegin =
                  std::sregex_iterator(headerContent->begin(), headerContent->end(), headerReg);
                auto headersEnd = std::sregex_iterator();
                for (std::sregex_iterator regI = headersBegin; regI != headersEnd; ++regI) {
                    std::string headerId = (*regI)[0];
                    auto itr = includeData.find(headerId);
                    if (itr == includeData.end()) {
                        std::cerr << "Could not find header: " << headerId << std::endl;
                        ret = false;
                        break;
                    }
                    directIncludes.push_back(headerId);
                    if (!include_headers(itr->second.shaderFileName.string(), out, includes,
                                         filesInProgress, includeData, itr->second.included)) {
                        std::cerr << "Error in header " << headerId << " ("
                                  << itr->second.shaderFileName.string() << "), included from "
                                  << filename << std::endl;
                        ret = false;
                        break;
                    }
                }
            }
        }
    }
    catch (...) {
        filesInProgress.erase(filename);
        throw;
    }
    filesInProgress.erase(filename);
    out << contentStr;
    return ret;
}

template <typename F>
static bool collect_shader_f(const BlockFile& blockFile, const std::string& type, F&& f) {
    for (size_t i = 0; i < blockFile.getBlockCount("stages"); ++i) {
        const BlockFile* b = blockFile.getNode("stages", i);
        if (b->hasNodes()) {
            for (size_t j = 0; j < b->getBlockCount(type); ++j) {
                const BlockFile* b2 = b->getNode(type, j);
                if (!f(i, b, j, b2))
                    return false;
            }
        }
        else if (b->hasContent()) {
            // completely empty block is allowed
            std::cerr
              << "A shader block (stages) contains direct content instead of separate blocks for stages."
              << std::endl;
            return false;
        }
    }
    return true;
}

static bool collect_shader_cfg(const BlockFile& blockFile, std::ostream& stagesOut,
                               const std::string& type) {
    return collect_shader_f(
      blockFile, type, [&](size_t, const BlockFile*, size_t, const BlockFile* node) {
          if (node->hasContent()) {
              stagesOut << *node->getContent();
          }
          else if (node->hasNodes()) {
              // completely empty block is allowed
              std::cerr << "A shader block (stages/" << type
                        << ") contains nested blocks instead of content." << std::endl;
              return false;
          }
          return true;
      });
}

static bool collect_shader(const BlockFile& blockFile, std::ostream& out, std::ostream& cfgOut,
                           const std::string& type) {
    if (!collect_shader_cfg(blockFile, cfgOut, type))
        return false;
    for (size_t i = 0; i < blockFile.getBlockCount("global"); ++i) {
        const BlockFile* b = blockFile.getNode("global", i);
        if (b->hasContent()) {
            out << *b->getContent();
        }
        else if (b->hasNodes()) {
            // completely empty block is allowed
            std::cerr << "A shader block (global) contains nested blocks instead of content."
                      << std::endl;
            return false;
        }
    }
    for (size_t i = 0; i < blockFile.getBlockCount(type); ++i) {
        const BlockFile* b = blockFile.getNode(type, i);
        if (b->hasContent()) {
            out << *b->getContent();
        }
        else if (b->hasNodes()) {
            // completely empty block is allowed
            std::cerr << "A shader block (" << type
                      << ") contains nested blocks instead of content." << std::endl;
            return false;
        }
    }
    return true;
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

bool generate_header(Cache& cache, ShaderRegistryOutput& registry, const std::string& shaderFile,
                     const std::string& outputFolder,
                     std::unordered_map<std::string, IncludeData>& includeData) {
    if (!fs::exists(fs::path(outputFolder)) && !fs::create_directories(fs::path(outputFolder))) {
        std::cerr << "Could not create directory for shader headers: " << outputFolder << std::endl;
        return false;
    }
    IncludeData incData;
    incData.shaderFileName = fs::path{shaderFile};
    std::ifstream shaderInput(shaderFile);
    if (!shaderInput.is_open()) {
        std::cerr << "Could not open file: " << shaderFile << std::endl;
        return false;
    }
    BlockFile b(shaderInput);
    shaderInput.close();
    if (b.hasContent()) {
        std::cerr << "Shader file has content on the root level (no blocks present): " << shaderFile
                  << std::endl;
        return false;
    }
    if (!b.hasNodes())
        return true;
    size_t descriptorCount = b.getBlockCount("descriptor");
    if (descriptorCount == 0)
        return true;
    if (descriptorCount > 1) {
        std::cerr << "A shader file may only contain one 'descriptor' block: " << shaderFile
                  << std::endl;
        return false;
    }
    const BlockFile* descBlock = b.getNode("descriptor");
    if (descBlock->hasContent()) {
        std::cerr << "The descriptor block must not have direct content." << std::endl;
        return false;
    }
    size_t variantsBlockCount = descBlock->hasNodes() ? descBlock->getBlockCount("variants") : 0;
    size_t resourcesBlockCount = descBlock->hasNodes() ? descBlock->getBlockCount("resources") : 0;
    if (variantsBlockCount > 1 || resourcesBlockCount > 1) {
        std::cerr << "The descriptor block can only have up to one variants and resources blocks"
                  << std::endl;
        return false;
    }
    std::string name = fs::path(shaderFile).stem().string();
    for (char& c : name)
        c = static_cast<char>(tolower(c));
    std::stringstream header;
    std::stringstream cxx;
    Variants variants;
    if (variantsBlockCount == 1) {
        const BlockFile* variantBlock = descBlock->getNode("variants");
        if (!read_variants(variantBlock, variants)) {
            std::cerr << "Could not read variants: " << shaderFile << std::endl;
            return false;
        }
    }
    Resources resources;
    if (resourcesBlockCount == 1) {
        const BlockFile* resourcesBlock = descBlock->getNode("resources");
        if (!read_resources(resourcesBlock, resources)) {
            std::cerr << "Could not read resources: " << shaderFile << std::endl;
            return false;
        }
    }
    const std::string className = "shader_" + name + "_descriptor";
    const std::string registryClassName = "shader_" + name + "_registry";
    fs::path headerFileName = fs::path("shader_header_" + name + ".h");
    fs::path cxxFileName = fs::path("shader_header_" + name + ".cpp");
    incData.desriptorClassName = className;
    incData.desriptorRegistryClassName = registryClassName;
    incData.name = name;
    header << "#pragma once\n\n";
    header << "#include <memory>\n\n";
    header << "#include <shaderdescriptor.h>\n";
    header << "#include <shadertypes.h>\n";
    header << "#include <drvshader.h>\n\n";

    cxx << "#include \"" << headerFileName.string() << "\"\n\n";
    cxx << "#include <drv.h>\n\n";

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
    uint64_t variantMul = 1;
    for (const auto& [variantName, values] : variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        incData.variantMultiplier[variantName] = variantMul;
        variantMul *= values.size();
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
    for (const auto& [variantName, values] : variants.values) {
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
    for (const auto& [variantName, values] : variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        std::string valName = get_variant_enum_val_name(variantName);
        cxx << "    ret += static_cast<uint32_t>(" << valName << ") * "
            << incData.variantMultiplier[variantName] << ";\n";
    }
    cxx << "    return ret;\n";
    cxx << "}\n";
    header << "    };\n";
    for (const auto& [varName, varType] : resources.variables) {
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
    incData.totalVarintMultiplier = variantMul;
    header
      << "    void setVariant(const std::string& variantName, const std::string& value) override;\n";
    cxx << "void " << className
        << "::setVariant(const std::string& variantName, const std::string& value) {\n";
    std::string ifString = "if";
    for (const auto& [variantName, values] : variants.values) {
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
    for (const auto& [variantName, values] : variants.values) {
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
    for (const auto& [variantName, values] : variants.values) {
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

    registry.headersStart << "    " << registryClassName << " " << name << ";\n";
    registry.headersCtor << "      " << (registry.firstHeader ? ':' : ',') << " " << name
                         << "(device)\n";
    registry.firstHeader = false;

    fs::path headerFilePath = fs::path(outputFolder) / headerFileName;
    fs::path cxxFilePath = fs::path(outputFolder) / cxxFileName;
    registry.includes << "#include <" << headerFileName.string() << ">\n";
    incData.headerFileName = fs::path{headerFileName};
    const std::string h = hash_string(header.str() + cxx.str());
    if (auto itr = cache.headerHashes.find(name);
        itr == cache.headerHashes.end() || itr->second != h) {
        std::ofstream outHeaderFile(headerFilePath.string());
        if (!outHeaderFile.is_open()) {
            std::cerr << "Could not open output file: " << headerFileName.string() << std::endl;
            return false;
        }
        outHeaderFile << header.str();
        outHeaderFile.close();
        std::ofstream outCxxFile(cxxFilePath.string());
        if (!outCxxFile.is_open()) {
            std::cerr << "Could not open output file: " << cxxFileName.string() << std::endl;
            return false;
        }
        outCxxFile << cxx.str();
        cache.headerHashes[name] = h;
    }
    includeData[name] = std::move(incData);

    return true;
}

bool read_variants(const BlockFile* blockFile, Variants& variants) {
    variants = {};
    if (blockFile->hasNodes()) {
        std::cerr << "variants block cannot contain nested blocks" << std::endl;
        return false;
    }
    if (!blockFile->hasContent())
        return true;
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
    return true;
}

bool read_resources(const BlockFile* blockFile, Resources& resources) {
    if (blockFile->hasNodes()) {
        std::cerr << "resources block cannot contain nested blocks" << std::endl;
        return false;
    }
    if (!blockFile->hasContent())
        return true;
    const std::string* resourcesContent = blockFile->getContent();
    std::regex varReg{
      "(uint|uint2|uint3|uint4|int|int2|int3|int4|float|vec2|vec3|vec4|mat44)\\s+(\\w+)\\s*;"};
    auto resourcesBegin =
      std::sregex_iterator(resourcesContent->begin(), resourcesContent->end(), varReg);
    auto resourcesEnd = std::sregex_iterator();
    for (std::sregex_iterator regI = resourcesBegin; regI != resourcesEnd; ++regI) {
        std::string varType = (*regI)[1];
        std::string varName = (*regI)[2];
        if (resources.variables.find(varName) != resources.variables.end()) {
            std::cerr << "A variable already exists with this name: " << varName << std::endl;
            return false;
        }
        resources.variables[varName] = varType;
    }
    return true;
}

static VariantConfig get_variant_config(
  uint32_t variantId, const std::vector<Variants>& variants,
  const std::unordered_map<std::string, uint32_t>& variantParamMultiplier) {
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

static void generate_shader_code(std::ostream& out /* resources */) {
    out << "#version 450\n";
    out << "#extension GL_ARB_separate_shader_objects : enable\n";
    if constexpr (featureconfig::params.shaderPrint) {
        out << "#extension GL_EXT_debug_printf : enable\n";
    }
    out << "\n";
}

static void generate_shader_attachments(const ShaderBin::StageConfig& configs,
                                        std::stringstream& code) {
    for (const auto& attachment : configs.attachments) {
        code << "layout(location = " << static_cast<uint32_t>(attachment.location) << ") ";
        if (attachment.info & ShaderBin::AttachmentInfo::WRITE)
            code << "out ";
        else
            throw std::runtime_error("Implement input attachments as well");
        // TODO handle depth attachment
        code << "vec4 " << attachment.name << ";\n";
    }
    code << "\n";
}

static std::vector<uint32_t> compile_shader_binary(const fs::path& debugPath,
                                                   const Compiler* compiler, ShaderBin::Stage stage,
                                                   size_t len, const char* code) {
    UNUSED(len);
    try {
        return compiler->GLSLtoSPV(stage, code);
    }
    catch (...) {
        fs::path debugFile = debugPath / fs::path{"error_compiled_shader.glsl"};
        std::ofstream debugOut(debugFile.c_str());
        debugOut << code;
        throw;
    }
}

static std::string format_variant(
  uint32_t variantId, const std::vector<Variants>& variants, const std::stringstream& text,
  const std::unordered_map<std::string, uint32_t>& variantParamMultiplier) {
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

static bool generate_binary(
  const fs::path& debugPath, const Compiler* compiler, ShaderBin::ShaderData& shaderData,
  const ResourceObject& resourceObj, uint32_t variantId, const std::vector<Variants>& variants,
  const std::stringstream& shader,
  std::unordered_map<ShaderHash, std::pair<size_t, size_t>>& codeOffsets,
  std::unordered_map<ShaderHash, std::pair<size_t, size_t>>& binaryOffsets, ShaderBin::Stage stage,
  const std::unordered_map<std::string, uint32_t>& variantParamMultiplier, std::ostream* genOut) {
    // TODO add resources to shader code
    std::stringstream shaderCodeSS;
    generate_shader_code(shaderCodeSS);
    if (stage == ShaderBin::PS)
        generate_shader_attachments(shaderData.stages[variantId].configs, shaderCodeSS);
    std::string shaderCode =
      shaderCodeSS.str() + format_variant(variantId, variants, shader, variantParamMultiplier);
    if (genOut) {
        *genOut << "// Stage: ";
        switch (stage) {
            case ShaderBin::PS:
                *genOut << "PS";
                break;
            case ShaderBin::VS:
                *genOut << "VS";
                break;
            case ShaderBin::CS:
                *genOut << "CS";
                break;
            case ShaderBin::NUM_STAGES:
                break;
        }
        *genOut << "\n\n";
        *genOut << shaderCode << std::endl;
    }

    ShaderHash codeHash = hash_code(shaderCode);
    if (auto itr = codeOffsets.find(codeHash); itr != codeOffsets.end()) {
        shaderData.stages[variantId].stageOffsets[stage] = itr->second.first;
        shaderData.stages[variantId].stageCodeSizes[stage] = itr->second.second;
        return true;
    }
    std::vector<uint32_t> binary =
      compile_shader_binary(debugPath, compiler, stage, shaderCode.length(), shaderCode.c_str());
    ShaderHash binHash = hash_binary(binary.size(), binary.data());
    if (auto itr = binaryOffsets.find(binHash); itr != binaryOffsets.end()) {
        shaderData.stages[variantId].stageOffsets[stage] = itr->second.first;
        shaderData.stages[variantId].stageCodeSizes[stage] = itr->second.second;
        codeOffsets[codeHash] = itr->second;
        return true;
    }
    size_t offset = shaderData.codes.size();
    size_t codeSize = binary.size();
    codeOffsets[codeHash] = std::make_pair(offset, codeSize);
    binaryOffsets[binHash] = std::make_pair(offset, codeSize);
    std::copy(binary.begin(), binary.end(),
              std::inserter(shaderData.codes, shaderData.codes.end()));
    shaderData.stages[variantId].stageOffsets[stage] = offset;
    shaderData.stages[variantId].stageCodeSizes[stage] = codeSize;
    return true;
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

static ShaderBin::StageConfig read_stage_configs(
  const Resources& resources, uint32_t variantId, const std::vector<Variants>& variants,
  const std::unordered_map<std::string, uint32_t>& variantParamMultiplier,
  const ShaderGenerationInput& input, PipelineResourceUsage& resourceUsage) {
    // TODO report an error for unknown values
    std::string statesInfo =
      format_variant(variantId, variants, input.statesCfg, variantParamMultiplier);
    std::string vsInfo = format_variant(variantId, variants, input.vsCfg, variantParamMultiplier);
    std::string psInfo = format_variant(variantId, variants, input.psCfg, variantParamMultiplier);
    std::string csInfo = format_variant(variantId, variants, input.csCfg, variantParamMultiplier);
    std::unordered_map<std::string, std::string> statesValues = read_values(statesInfo);
    std::unordered_map<std::string, std::string> vsValues = read_values(vsInfo);
    std::unordered_map<std::string, std::string> psValues = read_values(psInfo);
    std::unordered_map<std::string, std::string> csValues = read_values(csInfo);
    resourceUsage.vsUsage = read_used_resources(vsInfo, resources);
    resourceUsage.psUsage = read_used_resources(psInfo, resources);
    resourceUsage.csUsage = read_used_resources(csInfo, resources);
    ShaderBin::StageConfig ret;
    if (vsValues.count("entry") > 1 || psValues.count("entry") > 1 || csValues.count("entry") > 1
        || statesValues.count("polygonMode") > 1 || statesValues.count("cull") > 1
        || statesValues.count("depthCompare") > 1 || statesValues.count("useDepthClamp") > 1
        || statesValues.count("depthBiasEnable") > 1 || statesValues.count("depthTest") > 1
        || statesValues.count("depthWrite") > 1 || statesValues.count("stencilTest") > 1)
        throw std::runtime_error("Shater state overwrites are currently not supported");

    if (auto itr = vsValues.find("entry"); itr != vsValues.end())
        ret.vsEntryPoint = itr->second;
    if (auto itr = psValues.find("entry"); itr != psValues.end())
        ret.psEntryPoint = itr->second;
    if (auto itr = csValues.find("entry"); itr != csValues.end())
        ret.csEntryPoint = itr->second;
    if (auto itr = statesValues.find("polygonMode"); itr != statesValues.end())
        ret.polygonMode =
          translate_input<drv::PolygonMode>(itr->second, {{"fill", drv::PolygonMode::FILL},
                                                          {"line", drv::PolygonMode::LINE},
                                                          {"point", drv::PolygonMode::POINT}});
    if (auto itr = statesValues.find("cull"); itr != statesValues.end())
        ret.cullMode =
          translate_input<drv::CullMode>(itr->second, {{"none", drv::CullMode::NONE},
                                                       {"front", drv::CullMode::FRONT_BIT},
                                                       {"back", drv::CullMode::BACK_BIT},
                                                       {"all", drv::CullMode::FRONT_AND_BACK}});
    if (auto itr = statesValues.find("depthCompare"); itr != statesValues.end())
        ret.depthCompare = translate_input<drv::CompareOp>(
          itr->second, {{"never", drv::CompareOp::NEVER},
                        {"less", drv::CompareOp::LESS},
                        {"equal", drv::CompareOp::EQUAL},
                        {"less_or_equal", drv::CompareOp::LESS_OR_EQUAL},
                        {"greater", drv::CompareOp::GREATER},
                        {"not_equal", drv::CompareOp::NOT_EQUAL},
                        {"greater_or_equal", drv::CompareOp::GREATER_OR_EQUAL},
                        {"always", drv::CompareOp::ALWAYS}});
    if (auto itr = statesValues.find("useDepthClamp"); itr != statesValues.end())
        ret.useDepthClamp = translate_input<bool>(itr->second, {{"true", true}, {"false", false}});
    if (auto itr = statesValues.find("depthBiasEnable"); itr != statesValues.end())
        ret.depthBiasEnable =
          translate_input<bool>(itr->second, {{"true", true}, {"false", false}});
    if (auto itr = statesValues.find("depthTest"); itr != statesValues.end())
        ret.depthTest = translate_input<bool>(itr->second, {{"true", true}, {"false", false}});
    if (auto itr = statesValues.find("depthWrite"); itr != statesValues.end())
        ret.depthWrite = translate_input<bool>(itr->second, {{"true", true}, {"false", false}});
    if (auto itr = statesValues.find("stencilTest"); itr != statesValues.end())
        ret.stencilTest = translate_input<bool>(itr->second, {{"true", true}, {"false", false}});
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
    return ret;
}

static ResourceObject generate_resource_object(const Resources& resources,
                                               const PipelineResourceUsage& usages) {
    ResourceObject ret;
    for (const auto& itr : resources.variables) {
        const std::string& name = itr.first;
        if (usages.csUsage.usedVars.count(name))
            ret.packs[ResourceObject::CS].shaderVars.insert(name);
        if (usages.vsUsage.usedVars.count(name))
            ret.packs[ResourceObject::VS].shaderVars.insert(name);
        if (usages.psUsage.usedVars.count(name))
            ret.packs[ResourceObject::PS].shaderVars.insert(name);
    }
    // TODO push constant ranges could be combined
    // overlaps should also be supported
    return ret;
}

static bool generate_binary(const fs::path& debugPath, const Compiler* compiler,
                            const Resources& resources, ShaderBin::ShaderData& shaderData,
                            const std::vector<Variants>& variants, ShaderGenerationInput&& input,
                            const std::unordered_map<std::string, uint32_t>& variantParamMultiplier,
                            std::map<PipelineResourceUsage, ResourceObject>& resourceObjects,
                            std::vector<PipelineResourceUsage>& varintToResourceUsage,
                            const std::string& genFile) {
    std::set<std::string> variantParams;
    size_t count = 1;
    shaderData.variantParamNum = 0;
    for (const Variants& v : variants) {
        for (const auto& [name, values] : v.values) {
            for (const std::string& value : values) {
                // try to find value name in registered param names
                auto itr = variantParamMultiplier.find(value);
                if (itr != variantParamMultiplier.end()) {
                    std::cerr << "Name collision of variant param name (" << value
                              << ") and variant param: " << name << " / " << value << std::endl;
                    return false;
                }
            }
            if (shaderData.variantParamNum >= ShaderBin::MAX_VARIANT_PARAM_COUNT) {
                std::cerr
                  << "A shader has exceeded the current limit for max shader variant parameters ("
                  << ShaderBin::MAX_VARIANT_PARAM_COUNT << ")" << std::endl;
                return false;
            }
            shaderData.variantValues[shaderData.variantParamNum] =
              safe_cast<std::decay_t<decltype(shaderData.variantValues[0])>>(values.size());
            shaderData.variantParamNum++;
            if (variantParams.count(name) > 0) {
                std::cerr << "A shader variant param name is used multiple times: " << name
                          << std::endl;
                return false;
            }
            count *= values.size();
            variantParams.insert(name);
        }
    }
    shaderData.totalVariantCount = safe_cast<uint32_t>(count);
    shaderData.stages.clear();
    shaderData.stages.resize(shaderData.totalVariantCount);
    std::unordered_map<ShaderHash, std::pair<size_t, size_t>> codeOffsets;
    std::unordered_map<ShaderHash, std::pair<size_t, size_t>> binaryOffsets;
    varintToResourceUsage.resize(shaderData.totalVariantCount);
    std::ostream* genOut = nullptr;
    std::ofstream genOutF;
    if (genFile != "") {
        genOutF.open(genFile.c_str());
        if (genOutF.is_open())
            genOut = &genOutF;
    }
    for (uint32_t variantId = 0; variantId < shaderData.totalVariantCount; ++variantId) {
        PipelineResourceUsage resourceUsage;
        ShaderBin::StageConfig cfg = read_stage_configs(
          resources, variantId, variants, variantParamMultiplier, input, resourceUsage);
        if (resourceObjects.find(resourceUsage) == resourceObjects.end())
            resourceObjects[resourceUsage] = generate_resource_object(resources, resourceUsage);
        const ResourceObject& resourceObj = resourceObjects[resourceUsage];
        varintToResourceUsage[variantId] = resourceUsage;
        // TODO add resource packs to shader codes
        shaderData.stages[variantId].configs = cfg;
        if (genOut) {
            *genOut << "// ---------------------- Variant " << variantId
                    << " ----------------------\n";
            const VariantConfig config =
              get_variant_config(variantId, variants, variantParamMultiplier);
            for (const Variants& v : variants) {
                for (const auto& [name, values] : v.values) {
                    auto itr = config.variantValues.find(name);
                    assert(itr != config.variantValues.end());
                    *genOut << "// " << name << " = " << values[itr->second] << ";\n";
                }
            }
            *genOut << " ---\n";
        }
        if (cfg.vsEntryPoint != ""
            && !generate_binary(debugPath, compiler, shaderData, resourceObj, variantId, variants,
                                input.ps, codeOffsets, binaryOffsets, ShaderBin::PS,
                                variantParamMultiplier, genOut)) {
            std::cerr << "Could not generate PS binary." << std::endl;
            return false;
        }
        if (cfg.psEntryPoint != ""
            && !generate_binary(debugPath, compiler, shaderData, resourceObj, variantId, variants,
                                input.vs, codeOffsets, binaryOffsets, ShaderBin::VS,
                                variantParamMultiplier, genOut)) {
            std::cerr << "Could not generate VS binary." << std::endl;
            return false;
        }
        if (cfg.csEntryPoint != ""
            && !generate_binary(debugPath, compiler, shaderData, resourceObj, variantId, variants,
                                input.cs, codeOffsets, binaryOffsets, ShaderBin::CS,
                                variantParamMultiplier, genOut)) {
            std::cerr << "Could not generate CS binary." << std::endl;
            return false;
        }
    }
    return true;
}

static void include_all(std::ostream& out, const fs::path& root,
                        const std::unordered_map<std::string, IncludeData>& includeData,
                        const std::vector<std::string>& directIncludes,
                        std::vector<std::string>& allIncludes,
                        std::unordered_map<std::string, uint32_t>& variantIdMultiplier,
                        std::unordered_map<std::string, std::string>& variantParamToDescriptor,
                        uint32_t& variantIdMul) {
    for (const std::string& inc : directIncludes) {
        auto itr = includeData.find(inc);
        if (itr == includeData.end())
            throw std::runtime_error("Could not find include file for: " + inc);
        out << "#include \"" << fs::relative(itr->second.headerFileName, root).string() << "\"\n";
        allIncludes.push_back(inc);
        for (const auto& itr2 : itr->second.variantMultiplier) {
            if (variantParamToDescriptor.find(itr2.first) != variantParamToDescriptor.end())
                throw std::runtime_error("Variant param names must be unique");
            variantParamToDescriptor[itr2.first] = inc;
        }
        include_all(out, root, includeData, itr->second.included, allIncludes, variantIdMultiplier,
                    variantParamToDescriptor, variantIdMul);
        variantIdMultiplier[inc] = variantIdMul;
        variantIdMul *= itr->second.totalVarintMultiplier;
    }
}

struct TypeInfo
{
    std::string cxxType;
    size_t size;
    size_t align;
};

static TypeInfo get_type_info(const std::string& type) {
#define RET_TYPE(type) \
    return { #type, sizeof(type), alignof(type) }
    if (type == "int")
        RET_TYPE(int32_t);
    if (type == "int2")
        RET_TYPE(int2);
    if (type == "int3")
        RET_TYPE(int3);
    if (type == "int4")
        RET_TYPE(int4);
    if (type == "uint")
        RET_TYPE(uint32_t);
    if (type == "uint2")
        RET_TYPE(uint2);
    if (type == "uint3")
        RET_TYPE(uint3);
    if (type == "uint4")
        RET_TYPE(uint4);
    if (type == "float")
        RET_TYPE(float);
    if (type == "vec2")
        RET_TYPE(vec2);
    if (type == "vec3")
        RET_TYPE(vec3);
    if (type == "vec4")
        RET_TYPE(vec4);
    if (type == "mat44")
        RET_TYPE(mat44);
#undef RET_TYPE
    throw std::runtime_error("Unkown type: " + type);
}

void ResourcePack::generateCXX(const std::string& structName, const Resources& resources,
                               std::ostream& out) const {
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
    size_t predictedSize = 0;
    auto gen_separator = [&](size_t align) {
        size_t offset = predictedSize % align;
        if (offset != 0) {
            out << "    // padding of " << (align - offset) << "bytes\n";
            predictedSize += align - offset;
        }
    };
    auto export_var = [&](auto itr) {
        const std::string& var = *itr;
        gen_separator(vars[var].align);
        out << "    " << vars[var].cxxType << " " << var << "; // offset: " << predictedSize
            << "\n";
        offsets[var] = predictedSize;
        predictedSize += vars[var].size;
        initOrder.push_back(var);
        sizeOrder.erase(itr);
    };
    constexpr size_t structAlignas = 16;
    out << "struct alignas(" << structAlignas << ") " << structName << " {\n";
    while (sizeOrder.size() > 0) {
        size_t requiredOffset = vars[sizeOrder[0]].align;
        if (predictedSize % requiredOffset != 0) {
            size_t padding = requiredOffset - (predictedSize % requiredOffset);
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
    out << "    // size without padding at the end\n";
    out << "    static constexpr size_t CONTENT_SIZE = " << predictedSize << ";\n";
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
    out << "    " << structName << "(";
    first = true;
    for (const std::string& var : initOrder) {
        if (!first)
            out << ", ";
        first = false;
        auto itr = resources.variables.find(var);
        assert(itr != resources.variables.end());
        out << "const " << itr->second << " &_" << itr->first;
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
}

bool compile_shader(const fs::path& debugPath, const Compiler* compiler, ShaderBin& shaderBin,
                    Cache& cache, ShaderRegistryOutput& registry, const std::string& shaderFile,
                    const std::string& outputFolder,
                    std::unordered_map<std::string, IncludeData>& includeData,
                    const std::string& genFolder) {
    std::stringstream cu;
    std::stringstream header;
    std::stringstream cxx;
    const std::string shaderName = fs::path{shaderFile}.stem().string();
    fs::path headerFileName = fs::path("shader_" + shaderName + ".h");
    fs::path cxxFileName = fs::path("shader_" + shaderName + ".cpp");
    header << "#pragma once\n\n";
    header << "#include <cstddef>\n";
    header << "#include <drvshader.h>\n";
    header << "#include <shaderbin.h>\n";
    header << "#include <shaderobject.h>\n";
    header << "#include <drvrenderpass.h>\n";
    header << "#include <shaderobjectregistry.h>\n\n";
    cxx << "#include \"" << headerFileName.string() << "\"\n\n";
    cxx << "#include <drv.h>\n";
    std::set<std::string> includes;
    std::set<std::string> progress;
    std::vector<std::string> directIncludes;
    if (!include_headers(shaderFile, cu, includes, progress, includeData, directIncludes)) {
        std::cerr << "Could not collect headers for shader: " << shaderFile << std::endl;
        return false;
    }
    BlockFile cuBlocks(cu, false);
    if (!cuBlocks.hasNodes()) {
        std::cerr << "Compilation unit doesn't have any blocks: " << shaderFile << std::endl;
        return false;
    }
    ShaderGenerationInput genInput;

    if (!collect_shader_cfg(cuBlocks, genInput.statesCfg, "states")) {
        std::cerr << "Could not collect shader stages/states content in: " << shaderFile
                  << std::endl;
        return false;
    }
    if (!collect_shader(cuBlocks, genInput.vs, genInput.vsCfg, "vs")) {
        std::cerr << "Could not collect vs shader content in: " << shaderFile << std::endl;
        return false;
    }
    if (!collect_shader(cuBlocks, genInput.ps, genInput.psCfg, "ps")) {
        std::cerr << "Could not collect ps shader content in: " << shaderFile << std::endl;
        return false;
    }
    if (!collect_shader(cuBlocks, genInput.cs, genInput.csCfg, "cs")) {
        std::cerr << "Could not collect cs shader content in: " << shaderFile << std::endl;
        return false;
    }
    if (
      !collect_shader_f(
        cuBlocks, "attachments", [&](size_t, const BlockFile*, size_t, const BlockFile* node) {
            if (node->hasNodes()) {
                for (size_t i = 0; i < node->getBlockCount(); ++i) {
                    const BlockFile* attachment = node->getNode(i);
                    if (attachment->hasNodes()) {
                        std::cerr
                          << "Attachment infos inside the attachments block must not contain other blocks: "
                          << shaderFile << std::endl;
                        return false;
                    }
                    if (attachment->hasContent())
                        genInput.attachments[node->getBlockName(i)] << *attachment->getContent();
                    else
                        genInput.attachments[node->getBlockName(i)] << "";
                }
            }
            else if (node->hasContent()) {
                std::cerr
                  << "The attachments block in shader stage infos must contain a block for each used attachment, not raw content: "
                  << shaderFile << std::endl;
                return false;
            }
            return true;
        })) {
        std::cerr << "Could not collect attachments in: " << shaderFile << std::endl;
        return false;
    }

    directIncludes.push_back(shaderName);  // include it's own header as well
    std::vector<std::string> allIncludes;
    std::unordered_map<std::string, std::string> variantParamToDescriptor;
    std::unordered_map<std::string, uint32_t> variantIdMultiplier;
    uint32_t variantIdMul = 1;
    include_all(header, fs::path{outputFolder}, includeData, directIncludes, allIncludes,
                variantIdMultiplier, variantParamToDescriptor, variantIdMul);

    std::vector<Variants> variants;
    size_t descriptorCount = cuBlocks.getBlockCount("descriptor");
    Resources resources;
    std::unordered_map<std::string, uint32_t> variantParamMultiplier;
    for (size_t i = 0; i < descriptorCount; ++i) {
        const BlockFile* descriptor = cuBlocks.getNode("descriptor", i);
        if (descriptor->getBlockCount("variants") == 1) {
            Variants v;
            read_variants(descriptor->getNode("variants"), v);
            for (const auto& [name, values] : v.values) {
                auto desc = variantParamToDescriptor.find(name);
                assert(desc != variantParamToDescriptor.end());
                auto descMulItr = variantIdMultiplier.find(desc->second);
                assert(descMulItr != variantIdMultiplier.end());
                auto inc = includeData.find(desc->second);
                assert(inc != includeData.end());
                auto variantMul = inc->second.variantMultiplier.find(name);
                assert(variantMul != inc->second.variantMultiplier.end());
                variantParamMultiplier[name] =
                  safe_cast<uint32_t>(descMulItr->second * variantMul->second);
            }
            variants.push_back(std::move(v));
        }
        if (descriptor->getBlockCount("resources") == 1) {
            const BlockFile* resourcesBlock = descriptor->getNode("resources");
            if (!read_resources(resourcesBlock, resources)) {
                std::cerr << "Could not read resources: " << shaderFile << std::endl;
                return false;
            }
        }
    }
    ShaderBin::ShaderData shaderData;
    std::map<PipelineResourceUsage, ResourceObject> resourceObjects;
    std::vector<PipelineResourceUsage> varintToResourceUsage;
    std::string genFile = "";
    if (genFolder != "")
        genFile = (fs::path{genFolder} / fs::path{shaderName + ".glsl"}).string();
    if (!generate_binary(debugPath, compiler, resources, shaderData, variants, std::move(genInput),
                         variantParamMultiplier, resourceObjects, varintToResourceUsage, genFile)) {
        std::cerr << "Could not generate binary: " << shaderFile << std::endl;
        return false;
    }

    shaderBin.addShader(shaderName, std::move(shaderData));

    const std::string className = "shader_" + shaderName;
    const std::string registryClassName = "shader_obj_registry_" + shaderName;

    registry.objectsStart << "    " << registryClassName << " " << shaderName << ";\n";
    registry.objectsCtor << "      " << (registry.firstObj ? ':' : ',') << " " << shaderName
                         << "(device, shaderBin";
    for (const std::string& inc : allIncludes) {
        auto itr = includeData.find(inc);
        assert(itr != includeData.end());
        registry.objectsCtor << ", &headers." << itr->second.name;
    }
    registry.objectsCtor << ")\n";
    registry.firstObj = false;

    header << "\n";

    uint32_t structId = 0;
    std::map<ResourcePack, std::string> exportedPacks;
    for (const auto& itr : resourceObjects) {
        for (const auto& [stages, pack] : itr.second.packs) {
            if (exportedPacks.find(pack) != exportedPacks.end())
                continue;
            std::string structName =
              "PushConstants_" + shaderName + "_" + std::to_string(structId++);
            exportedPacks[pack] = structName;
            pack.generateCXX(structName, resources, cxx);
        }
    }

    header << "class " << registryClassName << " final : public ShaderObjectRegistry {\n";
    header << "  public:\n";
    header << "    " << registryClassName
           << "(drv::LogicalDevicePtr device, const ShaderBin &shaderBin";
    cxx << registryClassName << "::" << registryClassName
        << "(drv::LogicalDevicePtr device, const ShaderBin &shaderBin";
    for (const std::string& inc : allIncludes) {
        auto itr = includeData.find(inc);
        assert(itr != includeData.end());
        header << ", const " << itr->second.desriptorRegistryClassName << " *reg_"
               << itr->second.name;
        cxx << ", const " << itr->second.desriptorRegistryClassName << " *reg_" << itr->second.name;
    }
    header << ");\n";
    cxx << ")\n";
    cxx << "  : ShaderObjectRegistry(device)\n";
    cxx << "{\n";
    cxx << "    const ShaderBin::ShaderData *shader = shaderBin.getShader(\"" << shaderName
        << "\");\n";
    cxx << "    if (shader == nullptr)\n";
    cxx << "        throw std::runtime_error(\"Shader not found: " << shaderName << "\");\n";
    cxx << "    loadShader(*shader);\n";

    std::map<PipelineResourceUsage, uint32_t> resourceUsageToConfigId;
    uint32_t configId = 0;
    for (const auto& [usage, object] : resourceObjects) {
        cxx << "    {\n";
        cxx << "        drv::DrvShaderObjectRegistry::PushConstantRange ranges["
            << object.packs.size() << "];\n";
        uint32_t rangeId = 0;
        for (const auto& [stages, pack] : object.packs) {
            if (!pack.shaderVars.empty()) {
                cxx << "        ranges[" << rangeId << "].stages = 0";
                if (stages & ResourceObject::VS)
                    cxx << " | drv::ShaderStage::VERTEX_BIT";
                if (stages & ResourceObject::PS)
                    cxx << " | drv::ShaderStage::FRAGMENT_BIT";
                if (stages & ResourceObject::CS)
                    cxx << " | drv::ShaderStage::COMPUTE_BIT";
                cxx << ";\n";
                cxx << "        ranges[" << rangeId << "].offset = ";
                if (rangeId == 0)
                    cxx << "0";
                else
                    cxx << "ranges[" << rangeId - 1 << "].offset + ranges[" << rangeId - 1
                        << "].size";
                cxx << ";\n";
                cxx << "        ranges[" << rangeId << "].size = sizeof("
                    << exportedPacks.find(pack)->second << ");\n";
                rangeId++;
            }
        }
        cxx << "        drv::DrvShaderObjectRegistry::ConfigInfo config;\n";
        cxx << "        config.numRanges = " << rangeId << ";\n";
        cxx << "        config.ranges = ranges;\n";
        cxx << "        reg->addConfig(config);\n";
        cxx << "    }\n";
        resourceUsageToConfigId[usage] = configId++;
    }
    cxx << "}\n\n";

    cxx << "static uint32_t CONFIG_INDEX[] = {";
    for (uint32_t i = 0; i < varintToResourceUsage.size(); ++i) {
        if (i > 0)
            cxx << ", ";
        auto itr = resourceUsageToConfigId.find(varintToResourceUsage[i]);
        assert(itr != resourceUsageToConfigId.end());
        cxx << itr->second;
    }
    cxx << "};\n\n";

    header << "    static VariantId get_variant_id(";
    cxx << "ShaderObjectRegistry::VariantId " << registryClassName << "::get_variant_id(";
    bool first = true;
    for (const std::string& inc : allIncludes) {
        auto itr = includeData.find(inc);
        assert(itr != includeData.end());
        if (!first) {
            header << ", ";
            cxx << ", ";
        }
        else
            first = false;
        header << "const " << itr->second.desriptorClassName << "::VariantDesc &"
               << itr->second.name;
        cxx << "const " << itr->second.desriptorClassName << "::VariantDesc &" << itr->second.name;
    }
    header << ");\n";
    cxx << ") {\n";
    cxx << "    uint32_t ret = 0;\n";
    for (const std::string& inc : allIncludes) {
        auto itr = includeData.find(inc);
        assert(itr != includeData.end());
        auto mulItr = variantIdMultiplier.find(inc);
        assert(mulItr != variantIdMultiplier.end());
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
    cxx << "  : ShaderObject(_device, _reg, \"" << shaderName << "\", std::move(dynamicStates))\n";
    cxx << "{\n";
    cxx << "}\n\n";
    header << "    ~" << className << "() override {}\n";
    header
      << "    uint32_t prepareGraphicalPipeline(const drv::RenderPass *renderPass, drv::SubpassId subpass, const DynamicState &dynamicStates";
    cxx
      << "uint32_t " << className
      << "::prepareGraphicalPipeline(const drv::RenderPass *renderPass, drv::SubpassId subpass, const DynamicState &dynamicStates";
    std::stringstream varintIdInput;
    first = true;
    for (const std::string& inc : allIncludes) {
        auto itr = includeData.find(inc);
        assert(itr != includeData.end());
        header << ", const " << itr->second.desriptorClassName << "::VariantDesc &"
               << itr->second.name;
        cxx << ", const " << itr->second.desriptorClassName << "::VariantDesc &"
            << itr->second.name;
        if (!first)
            varintIdInput << ", ";
        else
            first = false;
        varintIdInput << itr->second.name;
    }
    header << ", const GraphicsPipelineStates &overrideStates = {});\n";
    cxx << ", const GraphicsPipelineStates &overrideStates) {\n";
    cxx << "    GraphicsPipelineDescriptor desc;\n";
    cxx << "    desc.renderPass = renderPass;\n";
    cxx << "    desc.subpass = subpass;\n";
    cxx << "    desc.variantId = static_cast<const " << registryClassName
        << "*>(reg)->get_variant_id(" << varintIdInput.str() << ");\n";
    cxx << "    desc.configIndex = static_cast<const " << registryClassName
        << "*>(reg)->get_config_id(desc.variantId);\n";
    cxx << "    desc.states = overrideStates;\n";
    cxx << "    desc.dynamicStates = dynamicStates;\n";
    cxx << "    return getGraphicsPipeline(desc);\n";
    cxx << "}\n";
    for (const std::string& inc : allIncludes) {
        auto itr = includeData.find(inc);
        assert(itr != includeData.end());
        header
          << "    static size_t getPushConstOffset(ShaderObjectRegistry::VariantId variantId, const "
          << itr->second.desriptorClassName << " *" << itr->second.name << ");\n";
        cxx << "size_t " << className
            << "::getPushConstOffset(ShaderObjectRegistry::VariantId variantId, const "
            << itr->second.desriptorClassName << " *" << itr->second.name << ") {\n";
        cxx << "    // TODO\n";
        cxx << "    return 0;\n";
        cxx << "}\n";
        header
          << "    static size_t getPushConstSize(ShaderObjectRegistry::VariantId variantId, const "
          << itr->second.desriptorClassName << " *" << itr->second.name << ");\n";
        cxx << "size_t " << className
            << "::getPushConstSize(ShaderObjectRegistry::VariantId variantId, const "
            << itr->second.desriptorClassName << " *" << itr->second.name << ") {\n";
        cxx << "    // TODO\n";
        cxx << "    return 0;\n";
        cxx << "}\n";
    }
    header
      << "    void bindGraphicsInfo(drv::CmdRenderPass &renderPass, const DynamicState &dynamicStates";
    cxx << "void " << className
        << "::bindGraphicsInfo(drv::CmdRenderPass &renderPass, const DynamicState &dynamicStates";
    std::stringstream pipelineInput;
    for (const std::string& inc : allIncludes) {
        auto itr = includeData.find(inc);
        assert(itr != includeData.end());
        header << ", const " << itr->second.desriptorClassName << " *" << itr->second.name;
        cxx << ", const " << itr->second.desriptorClassName << " *" << itr->second.name;
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
    header << "};\n";

    registry.includes << "#include <" << headerFileName.string() << ">\n";

    const std::string h = hash_string(header.str() + cxx.str());
    const std::string cacheEntry = shaderName + "_obj";
    if (auto itr = cache.headerHashes.find(cacheEntry);
        itr == cache.headerHashes.end() || itr->second != h) {
        fs::path headerFilePath = fs::path(outputFolder) / headerFileName;
        fs::path cxxFilePath = fs::path(outputFolder) / cxxFileName;
        std::ofstream headerFile(headerFilePath.string());
        if (!headerFile.is_open()) {
            std::cerr << "Could not open output file: " << headerFilePath.string() << std::endl;
            return false;
        }
        headerFile << header.str();
        headerFile.close();
        std::ofstream cxxFile(cxxFilePath.string());
        if (!cxxFile.is_open()) {
            std::cerr << "Could not open output file: " << cxxFilePath.string() << std::endl;
            return false;
        }
        cxxFile << cxx.str();
        cache.headerHashes[cacheEntry] = h;
    }

    return true;
}

void Cache::writeJson(json& out) const {
    WRITE_OBJECT(headerHashes, out);
}

void Cache::readJson(const json& in) {
    READ_OBJECT_OPT(headerHashes, in, {});
}

void init_registry(ShaderRegistryOutput& registry) {
    registry.includes << "#pragma once\n\n";
    registry.includes << "#include <drvtypes.h>\n";
    registry.includes << "#include <shaderbin.h>\n";
    registry.headersStart << "struct ShaderHeaderRegistry {\n";
    registry.headersStart << "    ShaderHeaderRegistry(const ShaderHeaderRegistry&) = delete;\n";
    registry.headersStart
      << "    ShaderHeaderRegistry& operator=(const ShaderHeaderRegistry&) = delete;\n";
    registry.headersCtor << "    ShaderHeaderRegistry(drv::LogicalDevicePtr device)\n";
    registry.objectsStart << "struct ShaderObjRegistry {\n";
    registry.objectsStart << "    ShaderObjRegistry(const ShaderObjRegistry&) = delete;\n";
    registry.objectsStart
      << "    ShaderObjRegistry& operator=(const ShaderObjRegistry&) = delete;\n";
    registry.objectsCtor
      << "    ShaderObjRegistry(drv::LogicalDevicePtr device, const ShaderBin &shaderBin, const ShaderHeaderRegistry& headers)\n";
}

void finish_registry(ShaderRegistryOutput& registry) {
    registry.includes << "\n";
    registry.headersEnd << "};\n\n";
    registry.objectsEnd << "};\n";
    registry.objectsCtor << "    {\n    }\n";
    registry.headersCtor << "    {\n    }\n";
}
