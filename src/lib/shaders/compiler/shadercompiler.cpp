#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>

using namespace std;

struct ShaderVarData
{
    std::string type;
    std::string name;
    size_t texId;
};

int main(int argc, char* argv[]) {
    if (argc <= 2) {
        cerr << "Not enough input arguments" << endl;
        cerr << "usage:" << endl;
        cerr << "<command> <output_file> <input_file1> [<input_files>...] " << endl;
        return 1;
    }
    ofstream out(argv[1]);
    if (!out) {
        cerr << "Could not open for write: " << argv[1] << endl;
        return 1;
    }
    out << "#pragma once" << endl;
    out << "#include <map>" << endl;
    out << "#include <mapbox/eternal.hpp>" << endl;
    out << "#include <stdexcept>" << endl;
    out << "#include <vector>" << endl;
    out << "#include <string>" << endl << endl;
    out << "struct ShaderVarData {" << endl;
    out << "    std::string type;" << endl;
    out << "    std::string name;" << endl;
    out << "    size_t texId;" << endl;
    out << "};" << endl << endl;
    out << "struct ShaderData {" << endl;
    out << "    std::string name;" << endl;
    out << "    std::string stage;" << endl;
    out << "    std::string content;" << endl;
    out << "    std::vector<ShaderVarData> uniform;" << endl;
    out << "    std::vector<ShaderVarData> attribute;" << endl;
    // out << "    std::vector<ShaderVarData> varying;" << endl;
    out << "};" << endl << endl;

    out << "inline const std::map<std::string, ShaderData> getShaderMap() {" << endl;
    out << "    std::map<std::string, ShaderData> data{" << endl;

    std::map<std::string, std::vector<std::string>> programs;
    for (int i = 2; i < argc; ++i) {
        std::regex filenameRegex{"([a-zA-Z0-9]+)\\.([a-zA-Z]+)"};
        std::regex uniformRegex{"uniform ([^ ;]+) ([^ ;]+)"};
        std::regex attributeRegex{"(attribute|in) ([^ ;]+) ([^ ;]+)"};
        std::regex texRegex{".*sampler.*"};
        // std::regex varyingRegex{"varying ([^ ;]+) ([^ ;]+)"};
        std::vector<ShaderVarData> uniform;
        std::vector<ShaderVarData> attribute;
        // std::vector<ShaderVarData> varying;
        size_t texId = 0;
        auto processVar = [&](const std::regex& reg, std::vector<ShaderVarData>& vec,
                              const std::string& line, unsigned int typeGroup,
                              unsigned int nameGroup) {
            std::smatch base_match;
            if (std::regex_search(line, base_match, reg)) {
                const std::string type = base_match[typeGroup].str();
                const std::string name = base_match[nameGroup].str();
                bool tex = std::regex_match(type, texRegex);
                vec.push_back(ShaderVarData{type, name, tex ? texId++ : 0});
            }
        };
        auto processUniform = [&uniform, &uniformRegex, &processVar](const std::string& line) {
            processVar(uniformRegex, uniform, line, 1, 2);
        };
        auto processAttribute = [&attribute, &attributeRegex,
                                 &processVar](const std::string& line) {
            processVar(attributeRegex, attribute, line, 2, 3);
        };
        // auto processVarying = [&varying, &varyingRegex, &processVar](const std::string& line) {
        //     processVar(varyingRegex, varying, line);
        // };
        auto processAll = [&processUniform, &processAttribute](const std::string& line) {
            processUniform(line);
            processAttribute(line);
            // processVarying(line);
        };
        out << "// file: " << argv[i] << endl;
        const string filename = argv[i];
        ifstream shader(argv[i]);
        if (!shader) {
            cerr << "Could not open for read: " << argv[i] << endl;
            return 1;
        }
        std::string content = "";
        std::string line;
        std::smatch base_match;
        if (!std::regex_search(filename, base_match, filenameRegex) || base_match.size() != 3) {
            cerr << "Input filename was in the wrong format: " << filename << endl;
            return 1;
        }
        const std::string shaderName = base_match[1].str();
        const std::string stage = base_match[2].str();
        const std::string shaderId = shaderName + "_" + stage;
        programs[shaderName].push_back(shaderId);
        out << "{\"" << shaderId << "\", {\"" << shaderName << "\", \"" << stage << "\"," << endl;
        while (getline(shader, line)) {
            processAll(line);
            out << "\"" << line << "\\n\"" << endl;
        }
        out << ", {" << endl;
        bool comma = false;
        for (const ShaderVarData& data : uniform) {
            if (comma)
                out << ", ";
            out << "{\"" << data.type << "\", \"" << data.name << "\", " << data.texId << "}";
            comma = true;
        }
        out << "}, {" << endl;
        comma = false;
        for (const ShaderVarData& data : attribute) {
            if (comma)
                out << ", ";
            out << "{\"" << data.type << "\", \"" << data.name << "\", 0}";
            comma = true;
        }
        // out << "}, {" << endl;
        // comma = false;
        // for (const ShaderVarData& data : varying) {
        //     if (comma)
        //         out << ", ";
        //     out << "{\"" << data.type << "\", \"" << data.name << "\"}";
        //     comma = true;
        // }
        if (i + 1 < argc)
            out << "}}}," << endl;
        else
            out << "}}}" << endl << endl;
    }
    out << "    };" << endl;
    out << "    return data;" << endl;
    out << "}" << endl << endl;

    out << "inline const ShaderData getShader(const std::string& shaderName) {" << endl;
    out << "    const auto& data = getShaderMap();" << endl;
    out << "    auto itr = data.find(shaderName);" << endl;
    out << "    if (itr == std::end(data))" << endl;
    out << "        throw std::runtime_error(\"Could not find shader: \" + shaderName);" << endl;
    out << "    return itr->second;" << endl;
    out << "}" << endl << endl;

    out << "inline const std::map<std::string, std::vector<std::string>> getPrograms() {" << endl;
    out << "    const std::map<std::string, std::vector<std::string>> programs{" << endl;
    bool comma = false;
    for (const auto& [shaderName, shaders] : programs) {
        if (comma)
            out << "        ,{\"" << shaderName << "\", {";
        else
            out << "        {\"" << shaderName << "\", {";
        comma = true;
        bool comma2 = false;
        for (const std::string& shaderId : shaders) {
            if (comma2)
                out << ", \"" << shaderId << "\"";
            else
                out << "\"" << shaderId << "\"";
            comma2 = true;
        }
        out << "}}" << endl;
    }
    out << "    };" << endl;
    out << "    return programs;" << endl;
    out << "}" << endl << endl;

    out
      << "inline const std::vector<std::string> getProgramShaders(const std::string& programName) {"
      << endl;
    out << "    const auto data = getPrograms();" << endl;
    out << "    auto itr = data.find(programName);" << endl;
    out << "    if (itr == std::end(data))" << endl;
    out << "        throw std::runtime_error(\"Could not find program: \" + programName);" << endl;
    out << "    return itr->second;" << endl;
    out << "}" << endl << endl;

    out << "inline const std::vector<std::string> getProgramNames() {" << endl;
    out << "    const std::vector<std::string> names{";
    comma = false;
    for (const auto& [shaderName, shaders] : programs) {
        if (comma)
            out << ", \"" << shaderName << "\"";
        else
            out << "\"" << shaderName << "\"";
        comma = true;
    }
    out << "};" << endl;
    out << "    return names;" << endl;
    out << "}" << endl << endl;

    return 0;
}
