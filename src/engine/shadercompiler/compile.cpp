#include "compile.h"

#include <cassert>
#include <cctype>
#include <fstream>
#include <iterator>
#include <set>

#include <HashLib4CPP.h>
#include <simplecpp.h>

#include <blockfile.h>
#include <uncomment.h>

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
    IHashResult res = hash->ComputeUntyped(data, len * sizeof(data[0]));
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
    directIncludes.clear();
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

static bool collect_shader(const BlockFile& blockFile, std::ostream& out, std::ostream& cfgOut,
                           const std::string& type) {
    for (size_t i = 0; i < blockFile.getBlockCount("stages"); ++i) {
        const BlockFile* b = blockFile.getNode("stages", i);
        if (b->hasNodes()) {
            for (size_t j = 0; j < b->getBlockCount(type); ++j) {
                const BlockFile* b2 = b->getNode(type, j);
                if (b2->hasContent()) {
                    cfgOut << *b2->getContent();
                }
                else if (b2->hasNodes()) {
                    // completely empty block is allowed
                    std::cerr << "A shader block (stages/" << type
                              << ") contains nested blocks instead of content." << std::endl;
                    return false;
                }
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

static void include_all(std::ostream& out, const fs::path& root,
                        const std::unordered_map<std::string, IncludeData>& includeData,
                        const std::vector<std::string>& directIncludes) {
    for (const std::string& inc : directIncludes) {
        auto itr = includeData.find(inc);
        if (itr == includeData.end())
            throw std::runtime_error("Could not find include file for: " + inc);
        out << "#include \"" << fs::relative(itr->second.headerFileName, root).string() << "\"\n";
        include_all(out, root, includeData, itr->second.included);
    }
}

bool compile_shader(const Compiler* compiler, ShaderBin& shaderBin, Cache& cache,
                    const std::string& shaderFile, const std::string& outputFolder,
                    std::unordered_map<std::string, IncludeData>& includeData) {
    std::stringstream cu;
    std::stringstream shaderObj;
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
    std::vector<Variants> variants;
    size_t descriptorCount = cuBlocks.getBlockCount("descriptor");
    for (size_t i = 0; i < descriptorCount; ++i) {
        const BlockFile* descriptor = cuBlocks.getNode("descriptor", i);
        if (descriptor->getBlockCount("variants") == 0)
            continue;
        Variants v;
        read_variants(descriptor->getNode("variants"), v);
        variants.push_back(std::move(v));
    }
    ShaderBin::ShaderData shaderData;
    if (!generate_binary(compiler, shaderData, variants, std::move(genInput))) {
        std::cerr << "Could not generate binary: " << shaderFile << std::endl;
        return false;
    }

    const std::string shaderName = fs::path{shaderFile}.stem().string();

    shaderBin.addShader(shaderName, std::move(shaderData));

    const std::string className = "shader_obj_" + shaderName;

    directIncludes.push_back(shaderName);  // include it's own header as well

    shaderObj << "#pragma once\n\n";
    shaderObj << "#include <shaderobject.h>\n";
    shaderObj << "#include <shaderbin.h>\n\n";
    include_all(shaderObj, fs::path{outputFolder}, includeData, directIncludes);
    shaderObj << "\n";
    shaderObj << "class " << className << " final : public ShaderObject {\n";
    shaderObj << "  public:\n";
    shaderObj
      << "    " << className
      << "(drv::LogicalDevicePtr device, const ShaderBin &shaderBin) : ShaderObject(device) {\n";
    shaderObj << "        const ShaderBin::ShaderData *shader = shaderBin.getShader(\""
              << shaderName << "\");\n";
    shaderObj << "        if (shader == nullptr)\n";
    shaderObj << "            throw std::runtime_error(\"Shader not found: " << shaderName
              << "\");\n";
    shaderObj << "        loadShader(*shader);\n";
    shaderObj << "    }\n";
    shaderObj << "    ~" << className << "() override {}\n";
    shaderObj << "  private:\n";
    shaderObj << "};\n";

    const std::string h = hash_string(shaderObj.str());
    const std::string cacheEntry = shaderName + "_obj";
    if (auto itr = cache.headerHashes.find(cacheEntry);
        itr == cache.headerHashes.end() || itr->second != h) {
        fs::path filePath = fs::path(outputFolder) / fs::path("shader_obj_" + shaderName + ".h");
        std::ofstream outFile(filePath.string());
        if (!outFile.is_open()) {
            std::cerr << "Could not open output file: " << filePath.string() << std::endl;
            return false;
        }
        outFile << shaderObj.str();
        cache.headerHashes[cacheEntry] = h;
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

bool generate_header(Cache& cache, const std::string& shaderFile, const std::string& outputFolder,
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
    std::stringstream out;
    Variants variants;
    if (variantsBlockCount == 1) {
        const BlockFile* variantBlock = descBlock->getNode("variants");
        if (!read_variants(variantBlock, variants)) {
            std::cerr << "Could not read variants: " << shaderFile << std::endl;
            return false;
        }
    }
    const std::string className = "shader_" + name + "_descriptor";
    out << "#pragma once\n\n";
    out << "#include <shaderdescriptor.h>\n\n";
    out << "class " << className << " final : public ShaderDescriptor\n";
    out << "{\n";
    out << "  public:\n";
    out << "    ~" << className << "() override {}\n";
    for (const auto& [variantName, values] : variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        std::string enumName = get_variant_enum_name(variantName);
        out << "    enum class " << enumName << " {\n";
        for (size_t i = 0; i < values.size(); ++i) {
            std::string val = get_variant_enum_value(values[i]);
            out << "        " << val << " = " << i << ",\n";
        }
        out << "    };\n";
        std::string valName = get_variant_enum_val_name(variantName);
        out << "    void setVariant_" << variantName << "(" << enumName << " value) {\n";
        out << "        " << valName << " = value;\n";
        out << "    }\n";
    }
    out
      << "    void setVariant(const std::string& variantName, const std::string& value) override {\n";
    std::string ifString = "if";
    for (const auto& [variantName, values] : variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        std::string enumName = get_variant_enum_name(variantName);
        std::string valName = get_variant_enum_val_name(variantName);
        out << "        " << ifString << " (variantName == \"" << variantName << "\") {\n";
        std::string ifString2 = "if";
        for (const std::string& variantVal : values) {
            const std::string val = get_variant_enum_value(variantVal);
            out << "            " << ifString2 << " (value == \"" << variantVal << "\")\n";
            out << "                " << valName << " = " << enumName << "::" << val << ";\n";
            ifString2 = "else if";
        }
        if (ifString2 != "if")
            out << "            else\n    ";
        out
          << "            throw std::runtime_error(\"Unknown value (\" + value + \") for shader variant param: "
          << variantName << "\");\n";
        out << "        }";
        ifString = " else if";
    }
    if (ifString != "if")
        out << " else\n    ";
    else
        out << "\n";
    out << "        throw std::runtime_error(\"Unknown variant param: \" + variantName);\n";
    out << "    }\n";
    out << "    void setVariant(const std::string& variantName, int value) override {\n";
    ifString = "if";
    for (const auto& [variantName, values] : variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        std::string enumName = get_variant_enum_name(variantName);
        std::string valName = get_variant_enum_val_name(variantName);
        out << "        " << ifString << " (variantName == \"" << variantName << "\")\n";
        out << "            " << valName << " = static_cast<" << enumName << ">(value);\n";
        ifString = "else if";
    }
    if (ifString != "if")
        out << "        else\n    ";
    out << "        throw std::runtime_error(\"Unknown variant param: \" + variantName);\n";
    out << "    }\n";
    out << "  private:\n";
    for (const auto& [variantName, values] : variants.values) {
        if (variantName.length() == 0 || values.size() == 0)
            continue;
        std::string enumName = get_variant_enum_name(variantName);
        std::string valName = get_variant_enum_val_name(variantName);
        out << "    " << enumName << " " << valName << ";\n";
    }
    out << "};\n";

    fs::path filePath = fs::path(outputFolder) / fs::path("shader_" + name + ".h");
    incData.headerFileName = fs::path{filePath};
    const std::string h = hash_string(out.str());
    if (auto itr = cache.headerHashes.find(name);
        itr == cache.headerHashes.end() || itr->second != h) {
        std::ofstream outFile(filePath.string());
        if (!outFile.is_open()) {
            std::cerr << "Could not open output file: " << filePath.string() << std::endl;
            return false;
        }
        outFile << out.str();
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

VariantConfig get_variant_config(size_t index, const std::vector<Variants>& variants) {
    VariantConfig ret;
    for (const Variants& v : variants) {
        for (const auto& [name, values] : v.values) {
            size_t valueId = index % values.size();
            ret.variantValues[name] = valueId;
            index /= values.size();
        }
    }
    return ret;
}

static void generate_shader_code(std::ostream& out /* resources */) {
    out << "#version 450\n";
    out << "#extension GL_ARB_separate_shader_objects : enable\n";
}

static std::vector<uint32_t> compile_shader_binary(const Compiler* compiler, ShaderBin::Stage stage,
                                                   size_t len, const char* code) {
    return compiler->GLSLtoSPV(stage, code);
}

static std::string format_variant(size_t variant, const std::vector<Variants>& variants,
                                  const std::stringstream& text) {
    VariantConfig config = get_variant_config(variant, variants);
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
  const Compiler* compiler, ShaderBin::ShaderData& shaderData, size_t variant,
  const std::vector<Variants>& variants, const std::stringstream& shader,
  std::unordered_map<ShaderHash, std::pair<size_t, size_t>>& codeOffsets,
  std::unordered_map<ShaderHash, std::pair<size_t, size_t>>& binaryOffsets,
  ShaderBin::Stage stage) {
    std::stringstream shaderCodeSS;
    generate_shader_code(shaderCodeSS);
    std::string shaderCode = shaderCodeSS.str() + format_variant(variant, variants, shader);

    ShaderHash codeHash = hash_code(shaderCode);
    if (auto itr = codeOffsets.find(codeHash); itr != codeOffsets.end()) {
        shaderData.stages[variant].stageOffsets[stage] = itr->second.first;
        shaderData.stages[variant].stageCodeSizes[stage] = itr->second.second;
        return true;
    }
    std::vector<uint32_t> binary =
      compile_shader_binary(compiler, stage, shaderCode.length(), shaderCode.c_str());
    ShaderHash binHash = hash_binary(binary.size(), binary.data());
    if (auto itr = binaryOffsets.find(binHash); itr != binaryOffsets.end()) {
        shaderData.stages[variant].stageOffsets[stage] = itr->second.first;
        shaderData.stages[variant].stageCodeSizes[stage] = itr->second.second;
        codeOffsets[codeHash] = itr->second;
        return true;
    }
    size_t offset = shaderData.codes.size();
    size_t codeSize = binary.size();
    codeOffsets[codeHash] = std::make_pair(offset, codeSize);
    binaryOffsets[binHash] = std::make_pair(offset, codeSize);
    std::copy(binary.begin(), binary.end(),
              std::inserter(shaderData.codes, shaderData.codes.end()));
    shaderData.stages[variant].stageOffsets[stage] = offset;
    shaderData.stages[variant].stageCodeSizes[stage] = codeSize;
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

static ShaderBin::StageConfig read_stage_configs(size_t variant,
                                                 const std::vector<Variants>& variants,
                                                 const std::stringstream& vs,
                                                 const std::stringstream& ps,
                                                 const std::stringstream& cs) {
    std::string vsInfo = format_variant(variant, variants, vs);
    std::string psInfo = format_variant(variant, variants, ps);
    std::string csInfo = format_variant(variant, variants, cs);
    std::unordered_map<std::string, std::string> vsValues = read_values(vsInfo);
    std::unordered_map<std::string, std::string> psValues = read_values(psInfo);
    std::unordered_map<std::string, std::string> csValues = read_values(csInfo);
    ShaderBin::StageConfig ret;
    if (auto itr = vsValues.find("entry"); itr != vsValues.end())
        ret.vs.entryPoint = itr->second;
    if (auto itr = psValues.find("entry"); itr != psValues.end())
        ret.ps.entryPoint = itr->second;
    if (auto itr = csValues.find("entry"); itr != csValues.end())
        ret.cs.entryPoint = itr->second;
    return ret;
}

bool generate_binary(const Compiler* compiler, ShaderBin::ShaderData& shaderData,
                     const std::vector<Variants>& variants, ShaderGenerationInput&& input) {
    std::set<std::string> variantParams;
    size_t count = 1;
    shaderData.variantParamNum = 0;
    for (const Variants& v : variants) {
        for (const auto& itr : v.values) {
            if (shaderData.variantParamNum >= ShaderBin::MAX_VARIANT_PARAM_COUNT) {
                std::cerr
                  << "A shader has exceeded the current limit for max shader variant parameters ("
                  << ShaderBin::MAX_VARIANT_PARAM_COUNT << ")" << std::endl;
                return false;
            }
            shaderData.variantValues[shaderData.variantParamNum] = itr.second.size();
            shaderData.variantParamNum++;
            if (variantParams.count(itr.first) > 0) {
                std::cerr << "A shader variant param name is used multiple times: " << itr.first
                          << std::endl;
                return false;
            }
            count *= itr.second.size();
            variantParams.insert(itr.first);
        }
    }
    shaderData.totalVariantCount = count;
    shaderData.stages.clear();
    shaderData.stages.resize(shaderData.totalVariantCount);
    std::unordered_map<ShaderHash, std::pair<size_t, size_t>> codeOffsets;
    std::unordered_map<ShaderHash, std::pair<size_t, size_t>> binaryOffsets;
    for (size_t variant = 0; variant < shaderData.totalVariantCount; ++variant) {
        ShaderBin::StageConfig cfg =
          read_stage_configs(variant, variants, input.vsCfg, input.psCfg, input.csCfg);
        shaderData.stages[variant].configs = cfg;
        if (cfg.vs.entryPoint != ""
            && !generate_binary(compiler, shaderData, variant, variants, input.ps, codeOffsets,
                                binaryOffsets, ShaderBin::PS)) {
            std::cerr << "Could not generate PS binary." << std::endl;
            return false;
        }
        if (cfg.ps.entryPoint != ""
            && !generate_binary(compiler, shaderData, variant, variants, input.vs, codeOffsets,
                                binaryOffsets, ShaderBin::VS)) {
            std::cerr << "Could not generate VS binary." << std::endl;
            return false;
        }
        if (cfg.cs.entryPoint != ""
            && !generate_binary(compiler, shaderData, variant, variants, input.cs, codeOffsets,
                                binaryOffsets, ShaderBin::CS)) {
            std::cerr << "Could not generate CS binary." << std::endl;
            return false;
        }
    }
    return true;
}

void Cache::writeJson(json& out) const {
    WRITE_OBJECT(headerHashes, out);
}

void Cache::readJson(const json& in) {
    READ_OBJECT_OPT(headerHashes, in, {});
}
