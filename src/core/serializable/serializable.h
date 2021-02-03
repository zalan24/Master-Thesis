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

class ISerializable
{
 protected:
    // This is required by the vector and map serialization
    static auto serialize(int value) { return value; }
    static auto serialize(float value) { return value; }
    static auto serialize(const std::string& value) { return value; }
    static auto serialize(bool value) { return value; }

    static void serialize(const json& in, int& value) { value = in; }
    static void serialize(const json& in, float& value) { value = in; }
    static void serialize(const json& in, std::string& value) { value = in; }
    static void serialize(const json& in, bool& value) { value = in; }

 public:
    virtual void write(std::ostream& out) const;
    virtual void read(std::istream& in);

    virtual void writeJson(json& out) const = 0;
    virtual void readJson(const json& in) = 0;

    template <typename T>
    static json serialize(const std::vector<T>& data) {
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

    template <typename T>
    static void serialize(const json& in, std::vector<T>& data) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        data.resize(in.size());
        for (size_t i = 0; i < in.size(); ++i)
            data[i] = in[i];
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
#define READ_OBJECT(name, json) serialize(json[#name], name)
#define READ_OBJECTS(name, count, json) serialize(json[#name], count, name)
#define READ_OBJECT_OPT(name, json, def) \
    if (json.count(#name) > 0)           \
        serialize(json[#name], name);    \
    else                                 \
        name = def