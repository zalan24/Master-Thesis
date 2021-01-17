#pragma once

#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

class ISerializable
{
 public:
    struct Entry
    {
        enum Type
        {
            INT,
            FLOAT,
            STRING,
            BOOL,
            OBJECT
        } type;
        union
        {
            const void* readPtr;
            void* writePtr;
        };
        std::string name;
        int count;
        Entry(Type t, const void* rPtr, const std::string& n, int c = -1)
          : type(t), name(n), count(c) {
            readPtr = rPtr;
        }
        Entry(Type t, void* wPtr, const std::string& n, int c = -1) : type(t), name(n), count(c) {
            writePtr = wPtr;
        }
    };

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
    static void serialize(const json& in, std::vector<T>& data) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        data.resize(in.size());
        for (size_t i = 0; i < in.size(); ++i)
            data[i] = in[i];
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

 protected:
    ~ISerializable() = default;

    // This is required by the vector and map serialization
    static auto serialize(int value) { return value; }
    static auto serialize(float value) { return value; }
    static auto serialize(const std::string& value) { return value; }
    static auto serialize(bool value) { return value; }

    static void serialize(const json& in, int& value) { value = in; }
    static void serialize(const json& in, float& value) { value = in; }
    static void serialize(const json& in, std::string& value) { value = in; }
    static void serialize(const json& in, bool& value) { value = in; }
};

template <typename T>
ISerializable::Entry::Type getType();
template <>
ISerializable::Entry::Type getType<int>();
template <>
ISerializable::Entry::Type getType<float>();
template <>
ISerializable::Entry::Type getType<std::string>();
template <>
ISerializable::Entry::Type getType<bool>();
template <>
ISerializable::Entry::Type getType<ISerializable*>();

#define WRITE_OBJECT(name, json) json[#name] = serialize(name)
#define READ_OBJECT(name, json) serialize(json[#name], name)
