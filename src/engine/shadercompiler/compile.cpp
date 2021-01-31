#include "compile.h"

#include <set>

namespace fs = std::filesystem;

bool uncomment(std::istream& in, std::ostream& out) {
    bool inComment = false;
    bool inString = false;
    bool inChar = false;
    bool lineComment = false;
    bool wasInComment = false;
    char lastC = '\0';
    char c;
    while (in.get(c)) {
        const bool canWrite = !wasInComment;
        wasInComment = inComment;
        switch (c) {
            case '\'':
                if (!inString && !inComment && (!inChar || lastC != '\\'))
                    inChar = !inChar;
                break;
            case '"':
                if (!inChar && !inComment && (!inString || lastC != '\\'))
                    inString = !inString;
                break;
            case '/':
                if (lastC == '/' && !inChar && !inString && !inComment) {
                    inComment = true;
                    lineComment = true;
                }
                else if (lastC == '*' && !inChar && !inString && inComment && !lineComment) {
                    inComment = false;
                }
                break;
            case '*':
                if (lastC == '/' && !inChar && !inString && !inComment) {
                    inComment = true;
                    lineComment = false;
                }
                break;
            case '\n':
                if (inComment && lineComment)
                    inComment = false;
                break;
            default:
                break;
        }
        if (lastC != '\0' && !inComment && canWrite) {
            out.put(lastC);
        }
        lastC = c;
    }
    if (lastC != '\0' && !inComment && !wasInComment)
        out.put(lastC);
    return !inString && !inChar;
}

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
    filesInProgress.insert(filename);
    try {
        // rec
    }
    catch (...) {
        filesInProgress.erase(filename);
        throw;
    }
    filesInProgress.erase(filename);
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

    return true;
}
