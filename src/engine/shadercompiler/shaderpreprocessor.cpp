#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

#include <CLI/CLI.hpp>

// #include <shaderbin.h>

// #include "compile.h"
// #include "compileconfig.h"
// #include "spirvcompiler.h"

#include "preprocessor.h"

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    try {
        CLI::App app{"Shader preprocessor"};
        app.set_help_all_flag("--help-all", "Show all help");

        std::vector<std::string> sources;
        app.add_option("--sources", sources, "Source files to open");
        std::vector<std::string> headers;
        app.add_option("--headers", headers, "Header files to open");
        std::string target = "";
        app.add_option("-t,--target", target, "Target output file (pass this to the linker)");
        std::string outputDir = "";
        app.add_option("-o,--output", outputDir, "Output dir for generated files");
        // std::string root = "";
        // app.add_option("-r,--root", root, "Shaders source root folder");
        // std::string headers = "";
        // app.add_option("--headers", headers, "Headers output dir");
        // std::string output = "";
        // app.add_option("-o,--output", output, "Output binary file");
        // std::string debugOut = "";
        // app.add_option("-d,--debug", debugOut, "Debug output folder");
        // std::string cacheF = "";
        // app.add_option("-c,--cache", cacheF, "Cache folder");
        // std::string generated = "";
        // app.add_option("-g,--generated", generated, "Folder to export generated shaders");
        std::string hardwareReq = "";
        app.add_option("--hardware", hardwareReq, "Hardware requirements config");
        // std::string compileOptionsFile = "";
        // app.add_option("--options", compileOptionsFile, "Compile options config");
        // // std::string hashFile = "";
        // // app.add_option("--hash", hashFile, "File path to hash output file (.h)");

        CLI11_PARSE(app, argc, argv)

        // std::unordered_map<std::string, fs::path> headerPaths;

        if (target == "" || outputDir == "") {
            std::cerr << "Please provide a source root folder and a target file." << std::endl;
            return 1;
        }
        // if (files.size() == 0)
        //     return 0;
        // fs::path rootPath = root;
        // fs::path debugPath = debugOut;

        // if (!fs::exists(debugPath))
        //     fs::create_directories(debugPath);

        // std::regex headerRegex("(.*)\\.sh");
        // std::regex shaderRegex("(.*)\\.sd");

        drv::DeviceLimits limits;
        if (hardwareReq != "")
            limits.importFromFile(fs::path{hardwareReq});

        if (!fs::exists(fs::path{target}))
            fs::create_directories(fs::path{target}.parent_path());
        if (!fs::exists(fs::path{outputDir}))
            fs::create_directories(fs::path{outputDir});

        Preprocessor preprocessor;

        preprocessor.importFromFile(fs::path{target});

        for (const auto& header : headers)
            preprocessor.processHeader(fs::path{header}, fs::path{outputDir});
        for (const auto& source : sources)  // sources have implicit headers
            preprocessor.processHeader(fs::path{source}, fs::path{outputDir});
        for (const auto& source : sources)
            preprocessor.processSource(fs::path{source}, fs::path{outputDir}, limits);

        preprocessor.generateRegistryFile(fs::path{outputDir} / fs::path{"shaderregistry.h"});
        preprocessor.cleanUp();

        // Compiler compiler;
        // ShaderBin shaderBin(limits);
        // CompilerData compileData;
        // compileData.outputFolder = headers;
        // compileData.debugPath = debugPath;
        // compileData.genFolder = generated;
        // compileData.compiler = &compiler;
        // compileData.shaderBin = &shaderBin;
        // // TODO compilerData.stats

        // fs::path cacheFolder = fs::path{cacheF};
        // fs::path cacheFile = cacheFolder / fs::path{"cache.json"};
        // fs::path registryFile = headers / fs::path{"shaderregistry.h"};

        //     if (generated != "") {
        //         fs::path genFolder = fs::path{generated};
        //         if (fs::exists(genFolder))
        //             fs::remove_all(genFolder);
        //         fs::create_directories(genFolder);
        //     }

        //     {
        //         std::ifstream cacheIn(cacheFile.string());
        //         if (cacheIn.is_open()) {
        //             json cacheJson;
        //             cacheIn >> cacheJson;
        //             ISerializable::serialize(cacheJson, compileData.cache);
        //         }
        //     }
        //     uint64_t shaderHash = compileData.cache.getHeaderHash();  // initial state

        //     CompileOptions compileOptions;
        //     if (compileOptionsFile != "") {
        //         std::ifstream optionsIn(hardwareReq.c_str());
        //         if (!optionsIn.is_open()) {
        //             std::cerr << "Could not open file: " << compileOptionsFile << std::endl;
        //             return 1;
        //         }
        //         compileOptions.read(optionsIn);
        //     }

        //     init_registry(compileData.registry);

        //     for (const std::string& f : files) {
        //         std::smatch m;
        //         if (!std::regex_match(f, m, headerRegex) && !std::regex_match(f, m, shaderRegex)) {
        //             std::cerr
        //               << "An input file doesn't match either the shader header (.sh) or the shader (.sd) file extensions."
        //               << std::endl;
        //             return 1;
        //         }
        //         if (!::generate_header(compileData, f)) {
        //             std::cerr << "Could not generate header for: " << f << std::endl;
        //             return 1;
        //         }
        //     }
        //     TODO;  // analyze incData + stats here
        //     for (const std::string& f : files) {
        //         std::smatch m;
        //         if (!std::regex_match(f, m, shaderRegex))
        //             continue;
        //         std::cout << "Compiling: " << f << std::endl;
        //         if (!compile_shader(compileData, f)) {
        //             std::cerr << "Failed to compile a shader: " << f << std::endl;
        //             return 1;
        //         }
        //     }
        //     finish_registry(compileData.registry);
        //     {
        //         std::ofstream registry(registryFile.c_str());
        //         registry << compileData.registry.includes.str()
        //                  << compileData.registry.headersStart.str()
        //                  << compileData.registry.headersCtor.str()
        //                  << compileData.registry.headersEnd.str()
        //                  << compileData.registry.objectsStart.str()
        //                  << compileData.registry.objectsCtor.str()
        //                  << compileData.registry.objectsEnd.str();
        //     }
        //     std::ofstream binOut(output, std::ios::binary | std::ios::out);
        //     if (!binOut.is_open()) {
        //         std::cerr << "Could not open output file: " << output << std::endl;
        //         return 1;
        //     }
        //     if (shaderHash != compileData.cache.getHeaderHash()) {
        //         // TODO
        //     }
        //     shaderBin.setHash(compileData.cache.getHeaderHash());
        //     shaderBin.write(binOut);
        //     binOut.close();
        if (!preprocessor.exportToFile(fs::path{target}))
            throw std::runtime_error("Could not export to file: " + target);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception has ocurred: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "An unknown exception has ocurred" << std::endl;
        return 1;
    }
    // {
    //     if (!fs::exists(fs::path(cacheFolder)) && !fs::create_directories(fs::path(cacheFolder))) {
    //         std::cerr << "Could not create directory for shader headers: " << cacheFolder
    //                   << std::endl;
    //         return 1;
    //     }

    //     std::ofstream cacheOut(cacheFile.string());
    //     if (!cacheOut.is_open()) {
    //         std::cerr << "Could not save cache" << std::endl;
    //         return 1;
    //     }
    //     json cacheJson = ISerializable::serialize(compileData.cache);
    //     cacheOut << cacheJson;
    // }
    return 0;
}
