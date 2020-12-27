#pragma once

#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

struct ProgramData;

struct Shader;
struct Program;
class ShaderManager;

struct ShaderVar
{
    GLint id;
    std::string type;
};

struct ShaderVarType
{
    GLenum type;
    GLint size;
    std::string name;
};

template <typename T>
inline ShaderVarType getShaderVarType();

template <>
inline ShaderVarType getShaderVarType<unsigned int>() {
    return {GL_UNSIGNED_INT, 1, "uint"};
}

template <>
inline ShaderVarType getShaderVarType<int>() {
    return {GL_INT, 1, "int"};
}

template <>
inline ShaderVarType getShaderVarType<float>() {
    return {GL_FLOAT, 1, "float"};
}

template <>
inline ShaderVarType getShaderVarType<glm::vec2>() {
    return {GL_FLOAT, 2, "vec2"};
}

template <>
inline ShaderVarType getShaderVarType<glm::vec3>() {
    return {GL_FLOAT, 3, "vec3"};
}

template <>
inline ShaderVarType getShaderVarType<glm::vec4>() {
    return {GL_FLOAT, 4, "vec4"};
}

template <>
inline ShaderVarType getShaderVarType<glm::mat2>() {
    return {GL_FLOAT, 0, "mat2"};
}

template <>
inline ShaderVarType getShaderVarType<glm::mat3>() {
    return {GL_FLOAT, 0, "mat3"};
}

template <>
inline ShaderVarType getShaderVarType<glm::mat4>() {
    return {GL_FLOAT, 0, "mat4"};
}

class AttributeBinder
{
 public:
    template <typename T, typename A>
    void addAttribute(const A T::*var, const std::string& varName, bool normalized = false) {
        dataSize = sizeof(T);
        // TODO check if the same type was used before
        // if (typeInited) {
        //     if (type != typeid(T))
        //         throw std::runtime_error("AttributeBinder is already set to a different type");
        // }
        // else {
        //     typeInited = true;
        //     type = typeid(T);
        // }
        const T hackWack{};
        const uint8_t* h = reinterpret_cast<const uint8_t*>(&hackWack);
        const uint8_t* v = reinterpret_cast<const uint8_t*>(&((&hackWack)->*var));
        size_t dist = v - h;
        data.push_back(AttributeData{varName, getShaderVarType<A>(), normalized,
                                     reinterpret_cast<const void*>(dist)});
    }

    void bind(const ShaderManager& shaderManager) const;

 private:
    // std::type_info type;
    // bool typeInited = false;
    std::size_t dataSize = 0;
    struct AttributeData
    {
        std::string varName;
        ShaderVarType type;
        GLboolean normalized;
        const GLvoid* pointer;
    };
    std::vector<AttributeData> data;
};

class ShaderManager
{
 public:
    ShaderManager();
    ~ShaderManager();

    const ShaderVar& findUniform(const std::string& name, const std::string& program = "") const;
    GLint getUniform(const std::string& name, const std::string& program = "") const;
    const ShaderVar& findAttribute(const std::string& name, const std::string& program = "") const;
    GLint getAttribute(const std::string& name, const std::string& program = "") const;
    const ProgramData& findProgram(const std::string& name) const;
    GLuint getProgram(const std::string& name) const;

    struct ShaderUniformData
    {
        std::string name;
        std::string type;
    };

    std::vector<ShaderUniformData> getProgramUniforms(const std::string& name) const;

    void useProgram(const std::string& name) const;

    template <typename T>
    void setUniforms(const std::string& varName, const T* values, std::size_t count,
                     const std::string& program = "") const {
        const ShaderVar var = findUniform(varName, program);
        const ShaderVarType type = getShaderVarType<T>();
        if (var.type != type.name)
            throw std::runtime_error("Could not set uniform (" + varName
                                     + "), because of type mismatch (" + type.name
                                     + " != " + var.type + ")");
        setUniforms(var.id, getProgram(program == "" ? lastProgram : program), values, count);
    }

    template <typename T>
    void setUniforms(const std::string& varName, const std::vector<T>& values,
                     const std::string& program = "") const {
        setUniforms(varName, values.data(), values.size(), program);
    }

    template <typename T>
    void setUniform(const std::string& varName, const T& value,
                    const std::string& program = "") const {
        setUniforms(varName, &value, 1, program);
    }

    template <typename T>
    void getUniforms(const std::string& varName, T* values, std::size_t count,
                     const std::string& program = "") const {
        const ShaderVar var = findUniform(varName, program);
        const ShaderVarType type = getShaderVarType<T>();
        if (var.type != type.name)
            throw std::runtime_error("Could not get uniform (" + varName
                                     + "), because of type mismatch (" + type.name
                                     + " != " + var.type + ")");
        getUniforms(var.id, getProgram(program == "" ? lastProgram : program), values, count);
    }

    template <typename T>
    void getUniforms(const std::string& varName, std::vector<T>& values,
                     const std::string& program = "") const {
        getUniforms(varName, values.data(), values.size(), program);
    }

    template <typename T>
    T getUniform(const std::string& varName, const std::string& program = "") const {
        T ret;
        getUniforms(varName, &ret, 1, program);
        return ret;
    }

 private:
    std::map<std::string, std::unique_ptr<ProgramData>> programs;

    mutable std::string lastProgram = "";

    template <typename T>
    void setUniforms(GLint varId, GLuint programId, const T* values, std::size_t count) const;

    template <>
    void setUniforms<glm::mat4>(GLint varId, GLuint programId, const glm::mat4* value,
                                std::size_t count) const;
    template <>
    void setUniforms<glm::vec2>(GLint varId, GLuint programId, const glm::vec2* value,
                                std::size_t count) const;
    template <>
    void setUniforms<glm::vec3>(GLint varId, GLuint programId, const glm::vec3* value,
                                std::size_t count) const;
    template <>
    void setUniforms<glm::vec4>(GLint varId, GLuint programId, const glm::vec4* value,
                                std::size_t count) const;
    template <>
    void setUniforms<float>(GLint varId, GLuint programId, const float* value,
                            std::size_t count) const;
    template <>
    void setUniforms<int>(GLint varId, GLuint programId, const int* value, std::size_t count) const;

    template <>
    void setUniforms<unsigned int>(GLint varId, GLuint programId, const unsigned int* value,
                                   std::size_t count) const;

    template <typename T>
    void getUniforms(GLint varId, GLuint programId, T* values, std::size_t count) const;

    template <>
    void getUniforms<float>(GLint varId, GLuint programId, float* values, std::size_t count) const;
};
