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
                for (std::sregex_iterator i = headersBegin; i != headersEnd; ++i) {
                    std::string headerId = (*i)[0];
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

bool compile_shader(const std::string& shaderFile,
                    const std::unordered_map<std::string, fs::path>& headerPaths) {
    std::stringstream cu;
    std::set<std::string> includes;
    std::set<std::string> progress;
    if (!include_headers(shaderFile, cu, headerPaths, includes, progress)) {
        std::cerr << "Could not collect headers for shader: " << shaderFile << std::endl;
        return false;
    }

    std::cout << cu.str() << std::endl;

    return true;
}
