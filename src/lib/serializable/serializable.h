#pragma once

#include <iostream>
#include <string>
#include <type_traits>
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
            BOOL
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
    virtual void gatherEntries(std::vector<Entry>& entries) const = 0;

    virtual ~ISerializable() {}

    virtual void write(std::ostream& out) const;
    virtual void read(std::istream& in);

    void write(json& out) const;
    void read(const json& in);

 protected:
    template <typename T>
    static Entry::Type getType();
    template <>
    static Entry::Type getType<int>();
    template <>
    static Entry::Type getType<float>();
    template <>
    static Entry::Type getType<std::string>();
    template <>
    static Entry::Type getType<bool>();
};

#define REGISTER_ENTRY(name, entries) \
    entries.push_back(                \
      ISerializable::Entry(ISerializable::getType<std::decay_t<decltype(name)>>(), &name, #name))
#define REGISTER_ENTRY_ARRAY(name, entries, count) \
    entries.push_back(ISerializable::Entry(getType<name>(), &name, #name, count))
