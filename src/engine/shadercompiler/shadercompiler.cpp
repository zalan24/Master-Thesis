#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

#include <CLI/CLI.hpp>

#include <shaderbin.h>

#include "compile.h"
#include "spirvcompiler.h"

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    CLI::App app{"Shader compiler"};
    app.set_help_all_flag("--help-all", "Show all help");

    std::vector<std::string> files;
    app.add_option("-s,--sources,sources", files, "Files or folders to open");
    std::string root = "";
    app.add_option("-r,--root", root, "Shaders source root folder");
    std::string headers = "";
    app.add_option("--headers", headers, "Headers output dir");
    std::string output = "";
    app.add_option("-o,--output", output, "Output binary file");
    std::string debugOut = "";
    app.add_option("-d,--debug", debugOut, "Debug output folder");
    std::string cacheF = "";
    app.add_option("-c,--cache", cacheF, "Cache folder");
    std::string generated = "";
    app.add_option("-g,--generated", generated, "Folder to export generated shaders");

    CLI11_PARSE(app, argc, argv)

    std::unordered_map<std::string, fs::path> headerPaths;

    if (root == "" || output == "" || headers == "" || debugOut == "") {
        std::cerr
          << "Please provide a source root folder, a debug output dir, a header output dir and an output bin file"
          << std::endl;
        return 1;
    }
    if (files.size() == 0)
        return 0;
    fs::path rootPath = root;
    fs::path debugPath = debugOut;

    if (!fs::exists(debugPath))
        fs::create_directories(debugPath);

    std::regex headerRegex("(.*)\\.sh");
    std::regex shaderRegex("(.*)\\.sd");

    ShaderBin shaderBin;
    Compiler compiler;
    Cache cache;
    std::unordered_map<std::string, IncludeData> includeData;

    fs::path cacheFolder = fs::path{cacheF};
    fs::path cacheFile = cacheFolder / fs::path{"cache.json"};
    fs::path registryFile = headers / fs::path{"shaderregistry.h"};

    if (generated != "") {
        fs::path genFolder = fs::path{generated};
        if (fs::exists(genFolder))
            fs::remove_all(genFolder);
        fs::create_directories(genFolder);
    }

    {
        std::ifstream cacheIn(cacheFile.string());
        if (cacheIn.is_open()) {
            json cacheJson;
            cacheIn >> cacheJson;
            ISerializable::serialize(cacheJson, cache);
        }
    }

    try {
        ShaderRegistryOutput registryData;
        init_registry(registryData);

        for (const std::string& f : files) {
            std::smatch m;
            if (!std::regex_match(f, m, headerRegex) && !std::regex_match(f, m, shaderRegex)) {
                std::cerr
                  << "An input file doesn't match either the shader header (.sh) or the shader (.sd) file extensions."
                  << std::endl;
                return 1;
            }
            if (!::generate_header(cache, registryData, f, headers, includeData)) {
                std::cerr << "Could not generate header for: " << f << std::endl;
                return 1;
            }
        }
        for (const std::string& f : files) {
            std::smatch m;
            if (!std::regex_match(f, m, shaderRegex))
                continue;
            std::cout << "Compiling: " << f << std::endl;
            if (!compile_shader(debugPath, &compiler, shaderBin, cache, registryData, f, headers,
                                includeData, generated)) {
                std::cerr << "Failed to compile a shader: " << f << std::endl;
                return 1;
            }
        }
        finish_registry(registryData);
        {
            std::ofstream registry(registryFile.c_str());
            registry << registryData.includes.str() << registryData.headersStart.str()
                     << registryData.headersCtor.str() << registryData.headersEnd.str()
                     << registryData.objectsStart.str() << registryData.objectsCtor.str()
                     << registryData.objectsEnd.str();
        }
        std::ofstream binOut(output, std::ios::binary | std::ios::out);
        if (!binOut.is_open()) {
            std::cerr << "Could not open output file: " << output << std::endl;
            return 1;
        }
        shaderBin.write(binOut);
        binOut.close();
    }
    catch (const std::exception& e) {
        std::cerr << "An exception has ocurred: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "An unknown exception has ocurred" << std::endl;
        return 1;
    }
    {
        if (!fs::exists(fs::path(cacheFolder)) && !fs::create_directories(fs::path(cacheFolder))) {
            std::cerr << "Could not create directory for shader headers: " << cacheFolder
                      << std::endl;
            return 1;
        }

        std::ofstream cacheOut(cacheFile.string());
        if (!cacheOut.is_open()) {
            std::cerr << "Could not save cache" << std::endl;
            return 1;
        }
        json cacheJson = ISerializable::serialize(cache);
        cacheOut << cacheJson;
    }
    return 0;
}
