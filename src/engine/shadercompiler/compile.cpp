#include "compile.h"

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
