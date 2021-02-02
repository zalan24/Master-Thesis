#include "compile.h"

#include <cctype>
#include <fstream>
#include <set>

#include <blockfile.h>
#include <uncomment.h>

namespace fs = std::filesystem;

static bool include_headers(const std::string& filename, std::ostream& out,
                            const std::unordered_map<std::string, fs::path>& headerPaths,
                            std::set<std::string>& includes,
                            std::set<std::string>& filesInProgress) {
    if (filesInProgress.count(filename) != 0) {
        std::cerr << "File recursively included: " << filename << std::endl;
        return false;
    }
    if (includes.count(filename) > 0)
        return true;
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
                    auto itr = headerPaths.find(headerId);
                    if (itr == headerPaths.end()) {
                        std::cerr << "Could not find header: " << headerId << std::endl;
                        ret = false;
                        break;
                    }
                    if (!include_headers(itr->second.string(), out, headerPaths, includes,
                                         filesInProgress)) {
                        std::cerr << "Error in header " << headerId << " (" << itr->second.string()
                                  << "), included from " << filename << std::endl;
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

static bool collect_shader(const BlockFile& blockFile, std::ostream& out, const std::string& type) {
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

bool compile_shader(const std::string& shaderFile,
                    const std::unordered_map<std::string, fs::path>& headerPaths) {
    std::stringstream cu;
    std::set<std::string> includes;
    std::set<std::string> progress;
    if (!include_headers(shaderFile, cu, headerPaths, includes, progress)) {
        std::cerr << "Could not collect headers for shader: " << shaderFile << std::endl;
        return false;
    }
    BlockFile cuBlocks(cu, false);
    if (!cuBlocks.hasNodes()) {
        std::cerr << "Compilation unit doesn't have any blocks: " << shaderFile << std::endl;
        return false;
    }
    std::stringstream vs;
    std::stringstream ps;
    std::stringstream cs;

    if (!collect_shader(cuBlocks, vs, "vs")) {
        std::cerr << "Could not collect vs shader content in: " << shaderFile << std::endl;
        return false;
    }
    if (!collect_shader(cuBlocks, ps, "ps")) {
        std::cerr << "Could not collect ps shader content in: " << shaderFile << std::endl;
        return false;
    }
    if (!collect_shader(cuBlocks, cs, "cs")) {
        std::cerr << "Could not collect cs shader content in: " << shaderFile << std::endl;
        return false;
    }

    std::cout << "vs: " << vs.str() << std::endl;
    std::cout << "ps: " << ps.str() << std::endl;
    std::cout << "cs: " << cs.str() << std::endl;

    return true;
}

bool generate_header(const std::string& shaderFile, const std::string& outputFolder) {
    if (!fs::exists(fs::path(outputFolder)) && !fs::create_directories(fs::path(outputFolder))) {
        std::cerr << "Could not create directory for shader headers: " << outputFolder << std::endl;
        return false;
    }
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
    fs::path filePath = fs::path(outputFolder) / fs::path("shader_" + name + ".h");
    std::ofstream out(filePath.string());
    if (!out.is_open()) {
        std::cerr << "Could not open output file: " << filePath.string() << std::endl;
        return false;
    }
    Variants variants;
    if (variantsBlockCount == 1) {
        const BlockFile* variantBlock = descBlock->getNode("variants");
        if (!read_variants(variantBlock, variants)) {
            std::cerr << "Could not read variants: " << shaderFile << std::endl;
            return false;
        }
    }
    out << "#pragma once\n\n";
    out << "namespace shader_" << name << "\n";
    out << "{\n";
    for (const auto& [name, values] : variants.values) {
        if (name.length() == 0 || values.size() == 0)
            continue;
        std::string enumName = name;
        for (char& c : enumName)
            c = static_cast<char>(tolower(c));
        enumName[0] = static_cast<char>(toupper(enumName[0]));
        out << "enum " << enumName << " {\n";
        for (size_t i = 0; i < values.size(); ++i) {
            std::string val = values[i];
            for (char& c : val)
                c = static_cast<char>(toupper(c));
            out << "\t" << val << " = " << i << ",\n";
        }
        out << "};\n";
    }
    out << "} // shader_" << name << "\n";
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
