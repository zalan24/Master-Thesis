#pragma once

#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <reflectable.hpp>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <binary_io.h>
#include <fixedarray.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

std::string hash_string(const std::string& data);
std::string hash_binary(size_t size, const void* data);

class ISerializable
{
 protected:
    using Marker = uint16_t;
    static constexpr Marker MARKER_STR = 0x53fe;
    static constexpr Marker MARKER_STR_END = 0xef35;
    static constexpr Marker MARKER_VEC = 0x542e;
    static constexpr Marker MARKER_VEC_END = 0xe245;
    static constexpr Marker MARKER_MAP = 0xfea7;
    static constexpr Marker MARKER_MAP_END = 0x7aef;
    static constexpr Marker MARKER_SET = 0x5e10;
    static constexpr Marker MARKER_SET_END = 0x01e5;
    static constexpr Marker MARKER_OBJ = 0x1253;
    static constexpr Marker MARKER_OBJ_END = 0x3521;
    static constexpr Marker MARKER_PTR = 0x12a3;
    static constexpr Marker MARKER_PTR_END = 0x35a1;

    template <typename Integer, typename = std::enable_if_t<std::is_integral<Integer>::value>>
    static auto serialize(Integer value) {
        return value;
    }

    static auto serialize(float value) { return value; }
    static auto serialize(double value) { return value; }
    static auto serialize(const std::string& value) { return value; }
    static auto serialize(bool value) { return value; }

    template <typename Integer, typename = std::enable_if_t<std::is_integral<Integer>::value>>
    static void serialize(const json& in, Integer& value) {
        value = in;
    }
    static void serialize(const json& in, float& value) { value = in; }
    static void serialize(const json& in, double& value) { value = in; }
    static void serialize(const json& in, std::string& value) { value = in; }
    static void serialize(const json& in, bool& value) { value = in; }

    template <typename Integer, typename = std::enable_if_t<std::is_integral<Integer>::value>>
    static bool serializeBin(std::ostream& out, Integer value) {
        return write_data(out, value);
    }
    static bool serializeBin(std::ostream& out, float value) { return write_data(out, value); }
    static bool serializeBin(std::ostream& out, double value) { return write_data(out, value); }
    static bool serializeBin(std::ostream& out, const std::string& value) {
        if (!write_data(out, MARKER_STR))
            return false;
        return write_string(out, value);
    }
    static bool serializeBin(std::ostream& out, bool value) { return write_data(out, value); }

    template <typename Integer, typename = std::enable_if_t<std::is_integral<Integer>::value>>
    static bool serializeBin(std::istream& in, Integer& value) {
        return read_data(in, value);
    }
    static bool serializeBin(std::istream& in, float& value) { return read_data(in, value); }
    static bool serializeBin(std::istream& in, double& value) { return read_data(in, value); }
    static bool serializeBin(std::istream& in, std::string& value) {
        Marker m;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_STR)
            throw std::runtime_error("Invalid binary file (str)");
        return read_string(in, value);
    }
    static bool serializeBin(std::istream& in, bool& value) { return read_data(in, value); }

 public:
    virtual bool writeBin(std::ostream& out) const = 0;
    virtual bool readBin(std::istream& in) = 0;

    virtual void writeJson(json& out) const = 0;
    virtual void readJson(const json& in) = 0;

    bool exportToFile(const fs::path& p) const;
    bool importFromFile(const fs::path& p);

    std::string hash() const;

    template <typename T>
    static json serialize(const std::unique_ptr<T>& data) {
        return serialize(*data);
    }

    template <typename T>
    static bool serializeBin(std::ostream& out, const std::unique_ptr<T>& data) {
        if (!write_data(out, MARKER_PTR))
            return false;
        if (!serializeBin(out, *data))
            return false;
        return write_data(out, MARKER_PTR_END);
    }

    template <typename T>
    static json serialize(const std::vector<T>& data) {
        json out = json::array();
        for (const T& v : data)
            out.push_back(serialize(v));
        return out;
    }

    template <typename T>
    static bool serializeBin(std::ostream& out, const std::vector<T>& data) {
        if (!write_data(out, MARKER_VEC))
            return false;
        uint64_t size = data.size();
        if (!write_data(out, size))
            return false;
        for (const T& v : data)
            if (!serializeBin(out, v))
                return false;
        return write_data(out, MARKER_VEC_END);
    }

    template <typename T, size_t N>
    static json serialize(const FixedArray<T, N>& data) {
        json out = json::array();
        for (uint32_t i = 0; i < data.size(); ++i)
            out.push_back(serialize(data[i]));
        return out;
    }

    template <typename T, size_t N>
    static bool serializeBin(std::ostream& out, const FixedArray<T, N>& data) {
        if (!write_data(out, MARKER_VEC))
            return false;
        uint64_t size = data.size();
        if (!write_data(out, size))
            return false;
        for (uint32_t i = 0; i < data.size(); ++i)
            if (!serializeBin(out, data[i]))
                return false;
        return write_data(out, MARKER_VEC_END);
    }

    template <typename T, size_t N>
    static json serialize(const std::array<T, N>& data) {
        json out = json::array();
        for (const T& v : data)
            out.push_back(serialize(v));
        return out;
    }

    template <typename T, size_t N>
    static bool serializeBin(std::ostream& out, const std::array<T, N>& data) {
        if (!write_data(out, MARKER_VEC))
            return false;
        uint64_t size = data.size();
        if (!write_data(out, size))
            return false;
        for (const T& v : data)
            if (!serializeBin(out, v))
                return false;
        return write_data(out, MARKER_VEC_END);
    }

    template <typename T>
    static json serialize(const std::set<T>& data) {
        json out = json::array();
        for (const T& v : data)
            out.push_back(serialize(v));
        return out;
    }

    template <typename T>
    static bool serializeBin(std::ostream& out, const std::set<T>& data) {
        if (!write_data(out, MARKER_SET))
            return false;
        uint64_t size = data.size();
        if (!write_data(out, size))
            return false;
        for (const T& v : data)
            if (!serializeBin(out, v))
                return false;
        return write_data(out, MARKER_SET_END);
    }

    template <typename V>
    static json serialize(const std::map<std::string, V>& data) {
        json out = json::object();
        for (const auto& [k, v] : data)
            out[k] = serialize(v);
        return out;
    }

    template <typename K, typename V>
    static json serialize(const std::map<K, V>& data) {
        json out = json::array();
        for (const auto& [k, v] : data) {
            json pair = json::array();
            pair.push_back(serialize(k));
            pair.push_back(serialize(v));
            out.push_back(pair);
        }
        return out;
    }

    template <typename K, typename V>
    static bool serializeBin(std::ostream& out, const std::map<K, V>& data) {
        if (!write_data(out, MARKER_MAP))
            return false;
        uint64_t size = data.size();
        if (!write_data(out, size))
            return false;
        uint64_t count = 0;
        for (const auto& [key, value] : data) {
            count++;
            if (!serializeBin(out, key))
                return false;
            if (!serializeBin(out, value))
                return false;
        }
        assert(size == count);
        return write_data(out, MARKER_MAP_END);
    }

    template <typename V>
    static json serialize(const std::unordered_map<std::string, V>& data) {
        json out = json::object();
        for (const auto& [k, v] : data)
            out[k] = serialize(v);
        return out;
    }

    template <typename K, typename V>
    static json serialize(const std::unordered_map<K, V>& data) {
        json out = json::array();
        for (const auto& [k, v] : data) {
            json pair = json::array();
            pair.push_back(serialize(k));
            pair.push_back(serialize(v));
            out.push_bock(pair);
        }
        return out;
    }

    template <typename K, typename V>
    static bool serializeBin(std::ostream& out, const std::unordered_map<K, V>& data) {
        if (!write_data(out, MARKER_MAP))
            return false;
        uint64_t size = data.size();
        if (!write_data(out, size))
            return false;
        for (const auto& [key, value] : data) {
            if (!serializeBin(out, key))
                return false;
            if (!serializeBin(out, value))
                return false;
        }
        return write_data(out, MARKER_MAP_END);
    }

    static json serialize(const ISerializable* value) {
        json out = json::object();
        value->writeJson(out);
        return out;
    }

    static bool serializeBin(std::ostream& out, const ISerializable* value) {
        if (!write_data(out, MARKER_OBJ))
            return false;
        if (!value->writeBin(out))
            return false;
        return write_data(out, MARKER_OBJ_END);
    }

    static json serialize(const ISerializable& value) { return serialize(&value); }
    static bool serializeBin(std::ostream& out, const ISerializable& value) {
        return serializeBin(out, &value);
    }

    template <typename T>
    static json serialize(size_t count, const T* values) {
        json out = json::array();
        for (size_t i = 0; i < count; ++i)
            out.push_back(serialize(values[i]));
        return out;
    }

    template <typename T>
    static bool serializeBin(std::ostream& out, size_t count, const T* values) {
        if (!write_data(out, MARKER_VEC))
            return false;
        uint64_t size = uint64_t(count);
        if (!write_data(out, size))
            return false;
        for (size_t i = 0; i < count; ++i)
            if (!serializeBin(out, v))
                return false;
        return write_data(out, MARKER_VEC_END);
    }

    static json serialize(const glm::vec2& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        return out;
    }

    static bool serializeBin(std::ostream& out, const glm::vec2& value) {
        return write_data(out, value);
    }

    static json serialize(const glm::vec3& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        out.push_back(serialize(value.z));
        return out;
    }

    static bool serializeBin(std::ostream& out, const glm::vec3& value) {
        return write_data(out, value);
    }

    static json serialize(const glm::quat& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        out.push_back(serialize(value.z));
        out.push_back(serialize(value.w));
        return out;
    }

    static bool serializeBin(std::ostream& out, const glm::quat& value) {
        return write_data(out, value);
    }

    static json serialize(const glm::vec4& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        out.push_back(serialize(value.z));
        out.push_back(serialize(value.w));
        return out;
    }

    static bool serializeBin(std::ostream& out, const glm::vec4& value) {
        return write_data(out, value);
    }

    static json serialize(const glm::ivec2& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        return out;
    }

    static bool serializeBin(std::ostream& out, const glm::ivec2& value) {
        return write_data(out, value);
    }

    static json serialize(const glm::ivec3& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        out.push_back(serialize(value.z));
        return out;
    }

    static bool serializeBin(std::ostream& out, const glm::ivec3& value) {
        return write_data(out, value);
    }

    static json serialize(const glm::ivec4& value) {
        json out = json::array();
        out.push_back(serialize(value.x));
        out.push_back(serialize(value.y));
        out.push_back(serialize(value.z));
        out.push_back(serialize(value.w));
        return out;
    }

    static bool serializeBin(std::ostream& out, const glm::ivec4& value) {
        return write_data(out, value);
    }

    static json serialize(const glm::mat4& value) {
        json out = json::array();
        out.push_back(serialize(value[0]));
        out.push_back(serialize(value[1]));
        out.push_back(serialize(value[2]));
        out.push_back(serialize(value[3]));
        return out;
    }

    static bool serializeBin(std::ostream& out, const glm::mat4& value) {
        return write_data(out, value);
    }

    template <typename T, size_t C>
    static auto serializeEnum(T value, const std::pair<T, const char*> (&names)[C]) {
        uint32_t ind = 0;
        while (ind < C && names[ind].first != value)
            ind++;
        assert(ind != C);
        return serialize(std::string(names[ind].second));
    }

    template <typename T, size_t C>
    static auto serializeEnumBin(std::ostream& out, T value,
                                 const std::pair<T, const char*> (&names)[C]) {
        uint32_t ind = 0;
        while (ind < C && names[ind].first != value)
            ind++;
        assert(ind != C);
        return serializeBin(out, std::string(names[ind].second));
    }

    template <typename T>
    static void serialize(const json& in, std::vector<T>& data) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        data.resize(in.size());
        for (size_t i = 0; i < in.size(); ++i)
            serialize(in[i], data[i]);
    }

    template <typename T>
    static void serialize(const json& in, std::unique_ptr<T>& data) {
        data = std::make_unique<T>();
        return serialize(in, *data);
    }

    template <typename T>
    static bool serializeBin(std::istream& in, std::unique_ptr<T>& data) {
        Marker m;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_PTR)
            throw std::runtime_error("Invalid binary file (ptr)");

        data = std::make_unique<T>();
        if (!serializeBin(in, *data))
            return false;

        if (!read_data(in, m))
            return false;
        if (m != MARKER_PTR_END)
            throw std::runtime_error("Invalid binary file (ptr)");
        return true;
    }

    template <typename T>
    static bool serializeBin(std::istream& in, std::vector<T>& data) {
        Marker m;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_VEC)
            throw std::runtime_error("Invalid binary file (vec)");
        uint64_t size;
        if (!read_data(in, size))
            return false;
        data.clear();
        data.reserve(size);
        for (uint64_t i = 0; i < size; ++i) {
            T value;
            if (!serializeBin(in, value))
                return false;
            data.push_back(std::move(value));
        }
        if (!read_data(in, m))
            return false;
        if (m != MARKER_VEC_END)
            throw std::runtime_error("Invalid binary file (vecEnd)");
        return true;
    }

    template <typename T, size_t N>
    static void serialize(const json& in, FixedArray<T, N>& data) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        data = FixedArray<T, N>(in.size());
        for (size_t i = 0; i < in.size(); ++i)
            serialize(in[i], data[i]);
    }

    template <typename T, size_t N>
    static bool serializeBin(std::istream& in, FixedArray<T, N>& data) {
        Marker m;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_VEC)
            throw std::runtime_error("Invalid binary file (vec)");
        uint64_t size;
        if (!read_data(in, size))
            return false;
        data = FixedArray<T, N>(size);
        for (uint64_t i = 0; i < size; ++i)
            if (!serializeBin(in, data[i]))
                return false;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_VEC_END)
            throw std::runtime_error("Invalid binary file (vecEnd)");
        return true;
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

    template <typename T, size_t N>
    static bool serializeBin(std::istream& in, std::array<T, N>& data) {
        Marker m;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_VEC)
            throw std::runtime_error("Invalid binary file (vec)");
        uint64_t size;
        if (!read_data(in, size))
            return false;
        if (data.size() != size)
            throw std::runtime_error("Wrong input size");
        for (uint64_t i = 0; i < size; ++i) {
            T value;
            if (!serializeBin(in, value))
                return false;
            data[i] = std::move(value);
        }
        if (!read_data(in, m))
            return false;
        if (m != MARKER_VEC_END)
            throw std::runtime_error("Invalid binary file (arrayEnd)");
        return true;
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

    template <typename T>
    static bool serializeBin(std::istream& in, std::set<T>& data) {
        Marker m;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_SET)
            throw std::runtime_error("Invalid binary file (set)");
        uint64_t size;
        if (!read_data(in, size))
            return false;
        data.clear();
        for (uint64_t i = 0; i < size; ++i) {
            T value;
            if (!serializeBin(in, value))
                return false;
            data.insert(std::move(value));
        }
        if (!read_data(in, m))
            return false;
        if (m != MARKER_SET_END)
            throw std::runtime_error("Invalid binary file (setEnd)");
        return true;
    }

    template <typename V>
    static void serialize(const json& in, std::map<std::string, V>& data) {
        if (!in.is_object())
            throw std::runtime_error("Input json is not an object: " + in.dump());
        data.clear();
        for (auto& [key, value] : in.items())
            serialize(value, data[key]);
    }

    template <typename K, typename V>
    static void serialize(const json& in, std::map<K, V>& data) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        data.clear();
        for (auto& pair : in) {
            if (!pair.is_array() || pair.size() != 2)
                throw std::runtime_error("Input json is not a 2-sized array: " + pair.dump());
            K key;
            serialize(pair[0], key);
            serialize(pair[1], data[key]);
        }
    }

    template <typename K, typename V>
    static bool serializeBin(std::istream& in, std::map<K, V>& data) {
        Marker m;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_MAP)
            throw std::runtime_error("Invalid binary file (map)");
        uint64_t size;
        if (!read_data(in, size))
            return false;
        data.clear();
        for (uint64_t i = 0; i < size; ++i) {
            K key;
            if (!serializeBin(in, key))
                return false;
            V value;
            if (!serializeBin(in, value))
                return false;
            data[std::move(key)] = std::move(value);
        }
        if (!read_data(in, m))
            return false;
        if (m != MARKER_MAP_END)
            throw std::runtime_error("Invalid binary file (mapEnd)");
        return true;
    }

    template <typename V>
    static void serialize(const json& in, std::unordered_map<std::string, V>& data) {
        if (!in.is_object())
            throw std::runtime_error("Input json is not an object: " + in.dump());
        data.clear();
        for (auto& [key, value] : in.items())
            serialize(value, data[key]);
    }

    template <typename K, typename V>
    static void serialize(const json& in, std::unordered_map<K, V>& data) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        data.clear();
        for (auto& pair : in) {
            if (!pair.is_array() || pair.size() != 2)
                throw std::runtime_error("Input json is not a 2-sized array: " + pair.dump());
            K key;
            V value;
            serialize(pair[0], key);
            serialize(pair[1], data[key]);
        }
    }

    template <typename K, typename V>
    static bool serializeBin(std::istream& in, std::unordered_map<K, V>& data) {
        Marker m;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_MAP)
            throw std::runtime_error("Invalid binary file (map)");
        uint64_t size;
        if (!read_data(in, size))
            return false;
        data.clear();
        for (uint64_t i = 0; i < size; ++i) {
            K key;
            if (!serializeBin(in, key))
                return false;
            V value;
            if (!serializeBin(in, value))
                return false;
            data[std::move(key)] = std::move(value);
        }
        if (!read_data(in, m))
            return false;
        if (m != MARKER_MAP_END)
            throw std::runtime_error("Invalid binary file (unordered_mapEnd)");
        return true;
    }

    static void serialize(const json& in, ISerializable* value) {
        if (!in.is_object())
            throw std::runtime_error("Input json is not an object: " + in.dump());
        value->readJson(in);
    }

    static bool serializeBin(std::istream& in, ISerializable* value) {
        Marker m;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_OBJ)
            throw std::runtime_error("Invalid binary file (obj)");
        if (!value->readBin(in))
            return false;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_OBJ_END)
            throw std::runtime_error("Invalid binary file (objEnd)");
        return true;
    }

    static void serialize(const json& in, ISerializable& value) { return serialize(in, &value); }
    static bool serializeBin(std::istream& in, ISerializable& value) {
        return serializeBin(in, &value);
    }

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

    template <typename T>
    static bool serializeBin(std::istream& in, size_t count, T* values) {
        Marker m;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_VEC)
            throw std::runtime_error("Invalid binary file (vec)");
        uint64_t size;
        if (!read_data(in, size))
            return false;
        if (count != size)
            throw std::runtime_error("Invalid count in binary file");
        for (uint64_t i = 0; i < size; ++i)
            if (!serializeBin(in, values[i]))
                return false;
        if (!read_data(in, m))
            return false;
        if (m != MARKER_VEC_END)
            throw std::runtime_error("Invalid binary file (static_arrayEnd)");
        return true;
    }

    static void serialize(const json& in, glm::vec2& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 2)
            throw std::runtime_error("Wrong json array size, expecting size of 2: " + in.dump());
        serialize(in[0], value.x);
        serialize(in[1], value.y);
    }

    static bool serializeBin(std::istream& in, glm::vec2& value) { return read_data(in, value); }

    static void serialize(const json& in, glm::vec3& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 3)
            throw std::runtime_error("Wrong json array size, expecting size of 3: " + in.dump());
        serialize(in[0], value.x);
        serialize(in[1], value.y);
        serialize(in[2], value.z);
    }

    static bool serializeBin(std::istream& in, glm::vec3& value) { return read_data(in, value); }

    static void serialize(const json& in, glm::quat& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 4)
            throw std::runtime_error("Wrong json array size, expecting size of 4: " + in.dump());
        serialize(in[0], value.x);
        serialize(in[1], value.y);
        serialize(in[2], value.z);
        serialize(in[3], value.w);
    }

    static bool serializeBin(std::istream& in, glm::quat& value) { return read_data(in, value); }

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

    static bool serializeBin(std::istream& in, glm::vec4& value) { return read_data(in, value); }

    static void serialize(const json& in, glm::ivec2& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 2)
            throw std::runtime_error("Wrong json array size, expecting size of 2: " + in.dump());
        serialize(in[0], value.x);
        serialize(in[1], value.y);
    }

    static bool serializeBin(std::istream& in, glm::ivec2& value) { return read_data(in, value); }

    static void serialize(const json& in, glm::ivec3& value) {
        if (!in.is_array())
            throw std::runtime_error("Input json is not an array: " + in.dump());
        if (in.size() != 3)
            throw std::runtime_error("Wrong json array size, expecting size of 3: " + in.dump());
        serialize(in[0], value.x);
        serialize(in[1], value.y);
        serialize(in[2], value.z);
    }

    static bool serializeBin(std::istream& in, glm::ivec3& value) { return read_data(in, value); }

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

    static bool serializeBin(std::istream& in, glm::ivec4& value) { return read_data(in, value); }

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

    static bool serializeBin(std::istream& in, glm::mat4& value) { return read_data(in, value); }

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

    template <typename T, size_t C>
    static void serializeEnumBin(std::istream& in, T& value,
                                 const std::pair<T, const char*> (&names)[C]) {
        std::string name;
        if (!serializeBin(in, name))
            return false;
        uint32_t ind = 0;
        while (ind < C && strcmp(names[ind].second, name.c_str()) != 0)
            ind++;
        if (ind == C)
            throw std::runtime_error("Invalid enum value <" + name
                                     + "> for type: " + std::string(typeid(T).name()));
        value = names[ind].first;
        return true;
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

template <typename Derived>
class IAutoSerializable : public ISerializable
{
 protected:
    virtual bool needTimeStamp() const { return false; }

    struct serialize_json_out_visitor
    {
        json& out;
        template <class FieldData>
        void operator()(FieldData f) {
            using Type = std::decay_t<decltype(f.get())>;
            if constexpr (std::is_enum_v<Type>) {
                out[f.name()] = ISerializable::serialize(Derived::get_enum_name(f.get()));
            }
            else {
                out[f.name()] = ISerializable::serialize(f.get());
            }
        }
    };

    struct serialize_json_in_visitor
    {
        const json& in;
        template <class FieldData>
        void operator()(FieldData f) {
            using Type = std::decay_t<decltype(f.get())>;
            if (!in.contains(std::string{f.name()}))
                throw std::runtime_error("Json doesn't contain a required object: "
                                         + std::string{f.name()});
            if (in.count(std::string{f.name()}) > 1)
                throw std::runtime_error("Json contains too many of an object: "
                                         + std::string{f.name()});
            if constexpr (std::is_enum_v<Type>) {
                std::string name;
                ISerializable::serialize(in[f.name()], name);
                f.get() = static_cast<Type>(Derived::get_enum(name));
            }
            else {
                ISerializable::serialize(in[f.name()], f.get());
            }
        }
    };

    struct serialize_bin_out_visitor
    {
        bool& ok;
        std::ostream& out;
        template <class FieldData>
        void operator()(FieldData f) {
            using Type = std::decay_t<decltype(f.get())>;
            if constexpr (std::is_enum_v<Type>) {
                int32_t enumVal = static_cast<int32_t>(f.get());
                ok = ISerializable::serializeBin(out, enumVal) && ok;
            }
            else {
                ok = ISerializable::serializeBin(out, f.get()) && ok;
            }
        }
    };

    struct serialize_bin_in_visitor
    {
        bool& ok;
        std::istream& in;
        template <class FieldData>
        void operator()(FieldData f) {
            using Type = std::decay_t<decltype(f.get())>;
            if constexpr (std::is_enum_v<Type>) {
                int32_t enumVal;
                ok = ISerializable::serializeBin(in, enumVal) && ok;
                f.get() = static_cast<Type>(enumVal);
            }
            else {
                ok = ISerializable::serializeBin(in, f.get()) && ok;
            }
        }
    };

    void checkDerived() const {
#ifdef DEBUG
        if (dynamic_cast<const Derived*>(this) == nullptr)
            throw std::runtime_error(
              "This struct is configured incorrectly. Use itself as template parameter for parent");
#endif
    }

 public:
    bool writeBin(std::ostream& out) const override {
        checkDerived();
        if (needTimeStamp())
            write_string(out, __TIMESTAMP__);
        bool ok = true;
        reflect::visit_each(*static_cast<const Derived*>(this), serialize_bin_out_visitor{ok, out});
        return ok;
    }
    bool readBin(std::istream& in) override {
        checkDerived();
        if (needTimeStamp()) {
            std::string stamp;
            read_string(in, stamp);
            if (stamp != __TIMESTAMP__)
                return false;
        }
        bool ok = true;
        reflect::visit_each(*static_cast<Derived*>(this), serialize_bin_in_visitor{ok, in});
        return ok;
    }

    void writeJson(json& out) const override {
        checkDerived();
        if (needTimeStamp())
            WRITE_TIMESTAMP(out);
        reflect::visit_each(*static_cast<const Derived*>(this), serialize_json_out_visitor{out});
    }
    void readJson(const json& in) override {
        checkDerived();
        if (needTimeStamp())
            CHECK_TIMESTAMP(in);
        reflect::visit_each(*static_cast<Derived*>(this), serialize_json_in_visitor{in});
    }

 protected:
    ~IAutoSerializable() = default;
};
