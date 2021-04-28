#pragma once

#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

using json = nlohmann::json;

std::string hash_string(const std::string& data);
std::string hash_binary(size_t size, const void* data);

class ISerializable
{
 protected:
    // This is required by the vector and map serialization
    static auto serialize(int32_t value) { return value; }
    static auto serialize(uint32_t value) { return value; }
    static auto serialize(float value) { return value; }
    static auto serialize(const std::string& value) { return value; }
    static auto serialize(bool value) { return value; }

    static void serialize(const json& in, int32_t& value) { value = in; }
    static void serialize(const json& in, uint32_t& value) { value = in; }
    static void serialize(const json& in, float& value) { value = in; }
    static void serialize(const json& in, std::string& value) { value = in; }
    static void serialize(const json& in, bool& value) { value = in; }

 public:
    virtual void write(std::ostream& out) const;
    virtual void read(std::istream& in);

    virtual void writeJson(json& out) const = 0;
    virtual void readJson(const json& in) = 0;

    std::string hash() const;

    template <typename T>
    static json serialize(const std::vector<T>& data) {
        json out = json::array();
        for (const T& v : data)
            out.push_back(serialize(v));
        return out;
    }

    template <typename T, size_t N>
    static json serialize(const std::array<T, N>& data) {
        json out = json::array();
        for (const T& v : data)
            out.push_back(serialize(v));
        return out;
    }

    template <typename T>
    static json serialize(const std::set<T>& data) {
        json out = json::array();
        for (const T& v : data)
            out.push_back(serialize(v));
        return out;
    }

    template <typename V>
    static json serialize(const std::map<std::string, V>& data) {
        json out = json::object();
        for (const auto& [k, v] : data)
            out[k] = serialize(v);
        return out;
    }

    template <typename V>
    static json serialize(const std::unordered_map<std::string, V>& data) {
        json out = json::object();
        for (const auto& [k, v] : data)
            out[k] = serialize(v);
        return out;
    }

    static json serialize(const ISerializable* value) {
        json out = json::object();
        value->writeJson(out);
        return out;
    }

    static json serialize(const ISerializable& value) { return serialize(&value); }

    template <typename T>
    static json serialize(size_t count, const T* values) {
        json out = json::array();
        for (size_t i = 0; i < count; ++i)
            out.push_back(serialize(values[i]));
        return out;
    }

    static json serialize(const glm::vec2& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        return out;
    }

    static json serialize(const glm::vec3& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        out.push_back(serialize(value.z));
        return out;
    }

    static json serialize(const glm::vec4& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        out.push_back(serialize(value.z));
        out.push_back(serialize(value.w));
        return out;
    }

    static json serialize(const glm::ivec2& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        return out;
    }

    static json serialize(const glm::ivec3& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        out.push_back(serialize(value.z));
        return out;
    }

    static json serialize(const glm::ivec4& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        out.push_back(serialize(value.z));
        out.push_back(serialize(value.w));
        return out;
    }

    static json serialize(const glm::mat4& value) {
        json out = json::array();
        out.push_back(serialize(value[0]));
        out.push_back(serialize(value[1]));
        out.push_back(serialize(value[2]));
        out.push_back(serialize(value[3]));
        return out;
    }

    template <typename T, size_t C>
    static auto serializeEnum(T value, const std::pair<T, const char*> (&names)[C]) {
        uint32_t ind = 0;
        while (ind < C && names[ind].first != value)
            ind++;
        assert(ind != C);
        return serialize(std::string(names[ind].second));
    }

    template <typename T>
    static void serialize(const json& in, std::vector<T>& data) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        data.resize(in.size());
        for (size_t i = 0; i < in.size(); ++i)
            serialize(in[i], data[i]);
    }

    template <typename T, size_t N>
    static void serialize(const json& in, std::array<T, N>& data) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != N)
            throw std::runtime_error("Wrong input size: " + in.dump());
        for (size_t i = 0; i < N; ++i)
            serialize(in[i], data[i]);
    }

    template <typename T>
    static void serialize(const json& in, std::set<T>& data) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        data.clear();
        for (size_t i = 0; i < in.size(); ++i) {
            T value;
            serialize(in[i], value);
            data.insert(std::move(value));
        }
    }

    template <typename V>
    static void serialize(const json& in, std::map<std::string, V>& data) {
        if (!in.is_object())
            throw std::runtime_error("Input json is not an object: " + in.dump());
        data.clear();
        for (auto& [key, value] : in.items())
            serialize(value, data[key]);
    }

    template <typename V>
    static void serialize(const json& in, std::unordered_map<std::string, V>& data) {
        if (!in.is_object())
            throw std::runtime_error("Input json is not an object: " + in.dump());
        data.clear();
        for (auto& [key, value] : in.items())
            serialize(value, data[key]);
    }

    static void serialize(const json& in, ISerializable* value) {
        if (!in.is_object())
            throw std::runtime_error("Input json is not an object: " + in.dump());
        value->readJson(in);
    }

    static void serialize(const json& in, ISerializable& value) { return serialize(in, &value); }

    template <typename T>
    static void serialize(const json& in, size_t count, T* values) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != count)
            throw std::runtime_error("Wrong json array size, expecting size of "
                                     + std::to_string(count) + ": " + in.dump());
        for (size_t i = 0; i < count; ++i)
            serialize(in[i], values[i]);
    }

    static void serialize(const json& in, glm::vec2& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 2)
            throw std::runtime_error("Wrong json array size, expecting size of 2: " + in.dump());
        serialize(in[0], value.x);
        serialize(in[1], value.y);
    }

    static void serialize(const json& in, glm::vec3& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 3)
            throw std::runtime_error("Wrong json array size, expecting size of 3: " + in.dump());
        serialize(in[0], value.x);
        serialize(in[1], value.y);
        serialize(in[2], value.z);
    }

    static void serialize(const json& in, glm::vec4& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 4)
            throw std::runtime_error("Wrong json array size, expecting size of 4: " + in.dump());
        serialize(in[0], value.x);
        serialize(in[1], value.y);
        serialize(in[2], value.z);
        serialize(in[3], value.w);
    }

    static void serialize(const json& in, glm::ivec2& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 2)
            throw std::runtime_error("Wrong json array size, expecting size of 2: " + in.dump());
        serialize(in[0], value.x);
        serialize(in[1], value.y);
    }

    static void serialize(const json& in, glm::ivec3& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 3)
            throw std::runtime_error("Wrong json array size, expecting size of 3: " + in.dump());
        serialize(in[0], value.x);
        serialize(in[1], value.y);
        serialize(in[2], value.z);
    }

    static void serialize(const json& in, glm::ivec4& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 4)
            throw std::runtime_error("Wrong json array size, expecting size of 4: " + in.dump());
        serialize(in[0], value.x);
        serialize(in[1], value.y);
        serialize(in[2], value.z);
        serialize(in[3], value.w);
    }

    static void serialize(const json& in, glm::mat4& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 4)
            throw std::runtime_error("Wrong json array size, expecting size of 4: " + in.dump());
        serialize(in[0], value[0]);
        serialize(in[1], value[1]);
        serialize(in[2], value[2]);
        serialize(in[3], value[3]);
    }

    template <typename T, size_t C>
    static void serializeEnum(const json& in, T& value,
                              const std::pair<T, const char*> (&names)[C]) {
        std::string name;
        serialize(in, name);
        uint32_t ind = 0;
        while (ind < C && strcmp(names[ind].second, name.c_str()) != 0)
            ind++;
        if (ind == C)
            throw std::runtime_error("Invalid enum value <" + name
                                     + "> for type: " + std::string(typeid(T).name()));
        value = names[ind].first;
    }

 protected:
    ~ISerializable() = default;
};

class IVirtualSerializable : public ISerializable
{
 public:
    virtual ISerializable* init(const std::string& type) = 0;
    virtual const ISerializable* getCurrent() const = 0;
    virtual std::string getCurrentType() const = 0;
    virtual void reset() = 0;

    void writeJson(json& out) const override final;
    void readJson(const json& in) override final;

 protected:
    ~IVirtualSerializable() = default;
};

#define WRITE_OBJECT(name, json) json[#name] = serialize(name)
#define WRITE_OBJECTS(name, count, json) json[#name] = serialize(count, name)
#define WRITE_ENUM(name, json, names) json[#name] = serializeEnum(name, names)
#define READ_OBJECT(name, json) serialize(json[#name], name)
#define READ_OBJECTS(name, count, json) serialize(json[#name], count, name)
#define READ_ENUM(name, json, names) serialize(json[#name], name, names)
#define READ_OBJECT_OPT(name, json, def) \
    if (json.count(#name) > 0)           \
        serialize(json[#name], name);    \
    else                                 \
        name = def
#define READ_ENUM_OPT(name, json, def, names)    \
    if (json.count(#name) > 0)                   \
        serializeEnum(json[#name], name, names); \
    else                                         \
        name = def
#define WRITE_TIMESTAMP(json) json["timeStamp"] = serialize(std::string{__TIMESTAMP__})
#define CHECK_TIMESTAMP(json) (json["timeStamp"] == std::string{__TIMESTAMP__})
