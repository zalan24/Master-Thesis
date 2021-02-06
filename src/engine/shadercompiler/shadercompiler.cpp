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

// struct ShaderVarData
// {
//     std::string type;
//     std::string name;
//     size_t texId;
// };

int main(int argc, char* argv[]) {
    CLI::App app{"Shader compiler"};
    app.set_help_all_flag("--help-all", "Show all help");

    std::vector<std::string> files;
    app.add_option("-s,--sources,sources", files, "Files or folders to open");
    std::string root = "";
    app.add_option("-r,--root", root, "Shaders source root folder");
    std::string headers = "";
    app.add_option("-d,--headers", headers, "Headers output dir");
    std::string output = "";
    app.add_option("-o,--output", output, "Output binary file");
    std::string cacheF = "";
    app.add_option("-c,--cache", cacheF, "Cache folder");

    CLI11_PARSE(app, argc, argv)

    std::unordered_map<std::string, fs::path> headerPaths;

    if (root == "" || output == "" || headers == "") {
        std::cerr
          << "Please provide a source root folder, a header output dir and an output bin file"
          << std::endl;
        return 1;
    }
    if (files.size() == 0)
        return 0;
    fs::path rootPath = root;

    std::regex headerRegex("(.*)\\.sh");
    std::regex shaderRegex("(.*)\\.sd");

    ShaderBin shaderBin;
    Compiler compiler;
    Cache cache;
    std::unordered_map<std::string, IncludeData> includeData;

    fs::path cacheFolder = fs::path{cacheF};
    fs::path cacheFile = cacheFolder / fs::path{"cache.json"};

    {
        std::ifstream cacheIn(cacheFile.string());
        if (cacheIn.is_open()) {
            json cacheJson;
            cacheIn >> cacheJson;
            ISerializable::serialize(cacheJson, cache);
        }
    }

    try {
        for (const std::string& f : files) {
            std::smatch m;
            if (!std::regex_match(f, m, headerRegex) && !std::regex_match(f, m, shaderRegex)) {
                std::cerr
                  << "An input file doesn't match either the shader header (.sh) or the shader (.sd) file extensions."
                  << std::endl;
                return 1;
            }
            if (!::generate_header(cache, f, headers, includeData)) {
                std::cerr << "Could not generate header for: " << f << std::endl;
                return 1;
            }
        }
        // for (const std::string& f : files) {
        //     std::smatch m;
        //     if (std::regex_match(f, m, headerRegex)) {
        //         fs::path p = f;
        //         fs::path relPath = fs::relative(p, root);
        //         std::string relPathStr = relPath.string();
        //         if (std::regex_match(relPathStr, m, headerRegex)) {
        //             std::string fileId = m[1];
        //             for (char& c : fileId)
        //                 if (c == '\\')
        //                     c = '/';
        //             headerPaths[fileId] = p;
        //         }
        //         else
        //             throw std::runtime_error("Something is wrong with a relative path");
        //     }
        // }
        for (const std::string& f : files) {
            std::smatch m;
            if (!std::regex_match(f, m, shaderRegex))
                continue;
            std::cout << "Compiling: " << f << std::endl;
            if (!compile_shader(&compiler, shaderBin, cache, f, headers, includeData)) {
                std::cerr << "Failed to compile a shader: " << f << std::endl;
                return 1;
            }
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

    // if (argc <= 2) {
    //     cerr << "Not enough input arguments" << endl;
    //     cerr << "usage:" << endl;
    //     cerr << "<command> <output_file> <input_file1> [<input_files>...] " << endl;
    //     return 1;
    // }
    // ofstream out(argv[1]);
    // if (!out) {
    //     cerr << "Could not open for write: " << argv[1] << endl;
    //     return 1;
    // }
    // out << "#pragma once" << endl;
    // out << "#include <map>" << endl;
    // out << "#include <mapbox/eternal.hpp>" << endl;
    // out << "#include <stdexcept>" << endl;
    // out << "#include <vector>" << endl;
    // out << "#include <string>" << endl << endl;
    // out << "struct ShaderVarData {" << endl;
    // out << "    std::string type;" << endl;
    // out << "    std::string name;" << endl;
    // out << "    size_t texId;" << endl;
    // out << "};" << endl << endl;
    // out << "struct ShaderData {" << endl;
    // out << "    std::string name;" << endl;
    // out << "    std::string stage;" << endl;
    // out << "    std::string content;" << endl;
    // out << "    std::vector<ShaderVarData> uniform;" << endl;
    // out << "    std::vector<ShaderVarData> attribute;" << endl;
    // // out << "    std::vector<ShaderVarData> varying;" << endl;
    // out << "};" << endl << endl;

    // out << "inline const std::map<std::string, ShaderData> getShaderMap() {" << endl;
    // out << "    std::map<std::string, ShaderData> data{" << endl;

    // std::map<std::string, std::vector<std::string>> programs;
    // for (int i = 2; i < argc; ++i) {
    //     std::regex filenameRegex{"(\\w+)\\.([a-zA-Z]+)"};
    //     std::regex uniformRegex{"uniform ([^ ;]+) (\\w+)"};
    //     std::regex attributeRegex{"(attribute|in) (\\w+) (\\w+)"};
    //     std::regex texRegex{".*sampler.*"};
    //     // std::regex varyingRegex{"varying ([^ ;]+) ([^ ;]+)"};
    //     std::vector<ShaderVarData> uniform;
    //     std::vector<ShaderVarData> attribute;
    //     // std::vector<ShaderVarData> varying;
    //     size_t texId = 0;
    //     auto processVar = [&](const std::regex& reg, std::vector<ShaderVarData>& vec,
    //                           const std::string& line, unsigned int typeGroup,
    //                           unsigned int nameGroup) {
    //         std::smatch base_match;
    //         if (std::regex_search(line, base_match, reg)) {
    //             const std::string type = base_match[typeGroup].str();
    //             const std::string name = base_match[nameGroup].str();
    //             bool tex = std::regex_match(type, texRegex);
    //             vec.push_back(ShaderVarData{type, name, tex ? texId++ : 0});
    //         }
    //     };
    //     auto processUniform = [&uniform, &uniformRegex, &processVar](const std::string& line) {
    //         processVar(uniformRegex, uniform, line, 1, 2);
    //     };
    //     auto processAttribute = [&attribute, &attributeRegex,
    //                              &processVar](const std::string& line) {
    //         processVar(attributeRegex, attribute, line, 2, 3);
    //     };
    //     // auto processVarying = [&varying, &varyingRegex, &processVar](const std::string& line) {
    //     //     processVar(varyingRegex, varying, line);
    //     // };
    //     auto processAll = [&processUniform, &processAttribute](const std::string& line) {
    //         processUniform(line);
    //         processAttribute(line);
    //         // processVarying(line);
    //     };
    //     out << "// file: " << argv[i] << endl;
    //     const string filename = argv[i];
    //     ifstream shader(argv[i]);
    //     if (!shader) {
    //         cerr << "Could not open for read: " << argv[i] << endl;
    //         return 1;
    //     }
    //     std::string content = "";
    //     std::string line;
    //     std::smatch base_match;
    //     if (!std::regex_search(filename, base_match, filenameRegex) || base_match.size() != 3) {
    //         cerr << "Input filename was in the wrong format: " << filename << endl;
    //         return 1;
    //     }
    //     const std::string shaderName = base_match[1].str();
    //     const std::string stage = base_match[2].str();
    //     const std::string shaderId = shaderName + "_" + stage;
    //     programs[shaderName].push_back(shaderId);
    //     out << "{\"" << shaderId << "\", {\"" << shaderName << "\", \"" << stage << "\"," << endl;
    //     while (getline(shader, line)) {
    //         processAll(line);
    //         out << "\"" << line << "\\n\"" << endl;
    //     }
    //     out << ", {" << endl;
    //     bool comma = false;
    //     for (const ShaderVarData& data : uniform) {
    //         if (comma)
    //             out << ", ";
    //         out << "{\"" << data.type << "\", \"" << data.name << "\", " << data.texId << "}";
    //         comma = true;
    //     }
    //     out << "}, {" << endl;
    //     comma = false;
    //     for (const ShaderVarData& data : attribute) {
    //         if (comma)
    //             out << ", ";
    //         out << "{\"" << data.type << "\", \"" << data.name << "\", 0}";
    //         comma = true;
    //     }
    //     // out << "}, {" << endl;
    //     // comma = false;
    //     // for (const ShaderVarData& data : varying) {
    //     //     if (comma)
    //     //         out << ", ";
    //     //     out << "{\"" << data.type << "\", \"" << data.name << "\"}";
    //     //     comma = true;
    //     // }
    //     if (i + 1 < argc)
    //         out << "}}}," << endl;
    //     else
    //         out << "}}}" << endl << endl;
    // }
    // out << "    };" << endl;
    // out << "    return data;" << endl;
    // out << "}" << endl << endl;

    // out << "inline const ShaderData getShader(const std::string& shaderName) {" << endl;
    // out << "    const auto& data = getShaderMap();" << endl;
    // out << "    auto itr = data.find(shaderName);" << endl;
    // out << "    if (itr == std::end(data))" << endl;
    // out << "        throw std::runtime_error(\"Could not find shader: \" + shaderName);" << endl;
    // out << "    return itr->second;" << endl;
    // out << "}" << endl << endl;

    // out << "inline const std::map<std::string, std::vector<std::string>> getPrograms() {" << endl;
    // out << "    const std::map<std::string, std::vector<std::string>> programs{" << endl;
    // bool comma = false;
    // for (const auto& [shaderName, shaders] : programs) {
    //     if (comma)
    //         out << "        ,{\"" << shaderName << "\", {";
    //     else
    //         out << "        {\"" << shaderName << "\", {";
    //     comma = true;
    //     bool comma2 = false;
    //     for (const std::string& shaderId : shaders) {
    //         if (comma2)
    //             out << ", \"" << shaderId << "\"";
    //         else
    //             out << "\"" << shaderId << "\"";
    //         comma2 = true;
    //     }
    //     out << "}}" << endl;
    // }
    // out << "    };" << endl;
    // out << "    return programs;" << endl;
    // out << "}" << endl << endl;

    // out
    //   << "inline const std::vector<std::string> getProgramShaders(const std::string& programName) {"
    //   << endl;
    // out << "    const auto data = getPrograms();" << endl;
    // out << "    auto itr = data.find(programName);" << endl;
    // out << "    if (itr == std::end(data))" << endl;
    // out << "        throw std::runtime_error(\"Could not find program: \" + programName);" << endl;
    // out << "    return itr->second;" << endl;
    // out << "}" << endl << endl;

    // out << "inline const std::vector<std::string> getProgramNames() {" << endl;
    // out << "    const std::vector<std::string> names{";
    // comma = false;
    // for (const auto& [shaderName, shaders] : programs) {
    //     if (comma)
    //         out << ", \"" << shaderName << "\"";
    //     else
    //         out << "\"" << shaderName << "\"";
    //     comma = true;
    // }
    // out << "};" << endl;
    // out << "    return names;" << endl;
    // out << "}" << endl << endl;

    return 0;
}
