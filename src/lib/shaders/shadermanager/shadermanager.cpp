#include "shadermanager.h"

#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

#include "shadercodes.h"

struct Shader
{
    Shader(GLuint i) noexcept : id(i) {}
    GLuint id;
    bool valid = true;
    void close() {
        if (valid)
            glDeleteShader(id);
    }
    ~Shader() { close(); }
    Shader(const Shader&) = delete;
    Shader(Shader&& other) {
        id = other.id;
        valid = other.valid;
        other.valid = false;
    }
    Shader& operator=(const Shader&) = delete;
    Shader& operator=(Shader&& other) {
        if (&other == this)
            return *this;
        close();
        id = other.id;
        valid = other.valid;
        other.valid = false;
        return *this;
    }
};
struct Program
{
    Program(GLuint i) noexcept : id(i) {}
    GLuint id;
    bool valid = true;
    void close() {
        if (valid)
            glDeleteProgram(id);
    }
    ~Program() { close(); }
    Program(const Program&) = delete;
    Program(Program&& other) {
        id = other.id;
        valid = other.valid;
        other.valid = false;
    }
    Program& operator=(const Program&) = delete;
    Program& operator=(Program&& other) {
        if (&other == this)
            return *this;
        close();
        id = other.id;
        valid = other.valid;
        other.valid = false;
        return *this;
    }
};

struct ProgramData
{
    Program program;
    std::vector<Shader> shaders;
    std::map<std::string, ShaderVar> uniforms;
    std::map<std::string, ShaderVar> attributes;
};

ShaderManager::ShaderManager() {
    static const std::map<std::string, unsigned int> stages{{"frag", GL_FRAGMENT_SHADER},
                                                            {"vert", GL_VERTEX_SHADER}};
    for (const std::string& programName : getProgramNames()) {
        Program program{glCreateProgram()};
        std::map<std::string, std::string> uniforms;
        std::map<std::string, std::string> attributes;
        std::vector<Shader> shaders;
        for (const std::string& shaderName : getProgramShaders(programName)) {
            const ShaderData& data = getShader(shaderName);
            const auto stageItr = stages.find(data.stage);
            if (stageItr == std::end(stages))
                throw std::runtime_error("Unknown shader stage: " + data.stage);
            Shader shader{glCreateShader(stageItr->second)};
            const char* content = data.content.c_str();
            glShaderSource(shader.id, 1, &content, nullptr);
            glCompileShader(shader.id);
            GLint isCompiled = 0;
            glGetShaderiv(shader.id, GL_COMPILE_STATUS, &isCompiled);
            if (isCompiled == GL_FALSE) {
                GLint maxLength = 0;
                glGetShaderiv(shader.id, GL_INFO_LOG_LENGTH, &maxLength);
                std::vector<GLchar> infoLog(maxLength);
                glGetShaderInfoLog(shader.id, maxLength, &maxLength, &infoLog[0]);
                throw std::runtime_error("Could not compile shader (" + shaderName
                                         + "): " + std::string{&infoLog[0], &infoLog.back()});
            }
            glAttachShader(program.id, shader.id);
            shaders.push_back(std::move(shader));
            if (stageItr->second == GL_VERTEX_SHADER) {
                for (const ShaderVarData& varData : data.attribute)
                    attributes[varData.name] = varData.type;
            }
            for (const ShaderVarData& varData : data.uniform)
                uniforms[varData.name] = varData.type;
        }
        glLinkProgram(program.id);
        GLint isLinked = 0;
        glGetProgramiv(program.id, GL_LINK_STATUS, (int*)&isLinked);
        if (isLinked == GL_FALSE) {
            GLint maxLength = 0;
            glGetProgramiv(program.id, GL_INFO_LOG_LENGTH, &maxLength);
            std::vector<GLchar> infoLog(maxLength);
            glGetProgramInfoLog(program.id, maxLength, &maxLength, &infoLog[0]);
            throw std::runtime_error("Could not link program (" + programName
                                     + "): " + std::string{&infoLog[0], &infoLog.back()});
        }
        std::map<std::string, ShaderVar> uniformsMap;
        std::map<std::string, ShaderVar> attributesMap;
        for (const auto& [varName, varType] : uniforms) {
            uniformsMap[varName] = {glGetUniformLocation(program.id, varName.c_str()), varType};
            if (uniformsMap[varName].id < 0)
                throw std::runtime_error("Could not find uniform (" + varName + ") in program ("
                                         + programName + ")");
        }
        for (const auto& [varName, varType] : attributes) {
            attributesMap[varName] = {glGetAttribLocation(program.id, varName.c_str()), varType};
            if (attributesMap[varName].id < 0)
                throw std::runtime_error("Could not find attribute (" + varName + ") in program ("
                                         + programName + ")");
        }
        for (Shader& shader : shaders) {
            glDetachShader(program.id, shader.id);
        }

        programs[programName] = std::unique_ptr<ProgramData>{
          new ProgramData{std::move(program), std::move(shaders), std::move(uniformsMap),
                          std::move(attributesMap)}};
    }
}

ShaderManager::~ShaderManager() {
}

const ProgramData& ShaderManager::findProgram(const std::string& name) const {
    auto itr = programs.find(name);
    if (itr == std::end(programs))
        throw std::runtime_error("Could not find program: " + name);
    return *itr->second;
}

GLuint ShaderManager::getProgram(const std::string& name) const {
    return findProgram(name).program.id;
}

void ShaderManager::useProgram(const std::string& name) const {
    glUseProgram(getProgram(name));
    lastProgram = name;
}

const ShaderVar& ShaderManager::findUniform(const std::string& name,
                                            const std::string& program) const {
    const ProgramData& programData = findProgram(program == "" ? lastProgram : program);
    auto itr = programData.uniforms.find(name);
    if (itr == std::end(programData.uniforms))
        throw std::runtime_error("Shader uniform (" + name
                                 + ") could not be found in program: " + program);
    return itr->second;
}

GLint ShaderManager::getUniform(const std::string& name, const std::string& program) const {
    return findUniform(name, program).id;
}

const ShaderVar& ShaderManager::findAttribute(const std::string& name,
                                              const std::string& program) const {
    const ProgramData& programData = findProgram(program == "" ? lastProgram : program);
    auto itr = programData.attributes.find(name);
    if (itr == std::end(programData.attributes))
        throw std::runtime_error("Shader attribute (" + name
                                 + ") could not be found in program: " + program);
    return itr->second;
}

GLint ShaderManager::getAttribute(const std::string& name, const std::string& program) const {
    return findAttribute(name, program).id;
}

void AttributeBinder::bind(const ShaderManager& shaderManager) const {
    for (const AttributeData& var : data) {
        const ShaderVar& shaderVar = shaderManager.findAttribute(var.varName);
        if (var.type.name != shaderVar.type)
            throw std::runtime_error("Could not bind attribute (" + var.varName
                                     + ") to shader attribute, because their types do not match ("
                                     + var.type.name + " != " + shaderVar.type + ")");
        GLint loc = shaderVar.id;
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, var.type.size, var.type.type, var.normalized, dataSize,
                              var.pointer);
    }
}

std::vector<ShaderManager::ShaderUniformData> ShaderManager::getProgramUniforms(
  const std::string& name) const {
    const ProgramData& programData = findProgram(name);
    std::vector<ShaderManager::ShaderUniformData> ret;
    for (const auto& [name, var] : programData.uniforms)
        ret.push_back({name, var.type});
    return ret;
}

template <>
void ShaderManager::setUniforms<glm::mat4>(GLint varId, GLuint programId, const glm::mat4* value,
                                           std::size_t count) const {
    glUniformMatrix4fv(varId, count, GL_FALSE, glm::value_ptr(*value));
}

template <>
void ShaderManager::setUniforms<glm::vec2>(GLint varId, GLuint programId, const glm::vec2* value,
                                           std::size_t count) const {
    glUniform2fv(varId, count, glm::value_ptr(*value));
}
template <>
void ShaderManager::setUniforms<glm::vec3>(GLint varId, GLuint programId, const glm::vec3* value,
                                           std::size_t count) const {
    glUniform3fv(varId, count, glm::value_ptr(*value));
}
template <>
void ShaderManager::setUniforms<glm::vec4>(GLint varId, GLuint programId, const glm::vec4* value,
                                           std::size_t count) const {
    glUniform4fv(varId, count, glm::value_ptr(*value));
}
template <>
void ShaderManager::setUniforms<float>(GLint varId, GLuint programId, const float* value,
                                       std::size_t count) const {
    glUniform1fv(varId, count, value);
}
template <>
void ShaderManager::setUniforms<int>(GLint varId, GLuint programId, const int* value,
                                     std::size_t count) const {
    glUniform1iv(varId, count, value);
}
template <>
void ShaderManager::setUniforms<unsigned int>(GLint varId, GLuint programId,
                                              const unsigned int* value, std::size_t count) const {
    glUniform1uiv(varId, count, value);
}

template <>
void ShaderManager::getUniforms<float>(GLint varId, GLuint programId, float* value,
                                       std::size_t count) const {
    glGetnUniformfv(programId, varId, count * sizeof(float), value);
}
