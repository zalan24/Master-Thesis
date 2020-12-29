#include "serializable.h"

void ISerializable::write(std::ostream& out) const {
    json j;
    write(j);
    out << j;
}

void ISerializable::read(std::istream& in) {
    json j;
    in >> j;
    read(j);
}

template <>
ISerializable::Entry::Type ISerializable::getType<int>() {
    return Entry::INT;
}
template <>
ISerializable::Entry::Type ISerializable::getType<float>() {
    return Entry::FLOAT;
}
template <>
ISerializable::Entry::Type ISerializable::getType<std::string>() {
    return Entry::STRING;
}
template <>
ISerializable::Entry::Type ISerializable::getType<bool>() {
    return Entry::BOOL;
}
template <>
ISerializable::Entry::Type ISerializable::getType<ISerializable*>() {
    return Entry::OBJECT;
}

static void write(json& to, const ISerializable::Entry& e) {
    switch (e.type) {
        case ISerializable::Entry::INT:
            to[e.name] = *static_cast<const int*>(e.readPtr);
            break;
        case ISerializable::Entry::FLOAT:
            to[e.name] = *static_cast<const float*>(e.readPtr);
            break;
        case ISerializable::Entry::STRING:
            to[e.name] = *static_cast<const std::string*>(e.readPtr);
            break;
        case ISerializable::Entry::BOOL:
            to[e.name] = *static_cast<const bool*>(e.readPtr);
            break;
        case ISerializable::Entry::OBJECT:
            json obj;
            (*static_cast<const ISerializable* const*>(e.readPtr))->write(obj);
            to[e.name] = obj;
            break;
    }
}

static void push(json& to, const ISerializable::Entry& e) {
    switch (e.type) {
        case ISerializable::Entry::INT:
            to.push_back(*static_cast<const int*>(e.readPtr));
            break;
        case ISerializable::Entry::FLOAT:
            to.push_back(*static_cast<const float*>(e.readPtr));
            break;
        case ISerializable::Entry::STRING:
            to.push_back(*static_cast<const std::string*>(e.readPtr));
            break;
        case ISerializable::Entry::BOOL:
            to.push_back(*static_cast<const bool*>(e.readPtr));
            break;
        case ISerializable::Entry::OBJECT:
            json obj;
            (*static_cast<const ISerializable* const*>(e.readPtr))->write(obj);
            to.push_back(obj);
            break;
    }
}

template <typename I>
static void read(const json& from, ISerializable::Entry::Type type, void* wPtr, const I& index,
                 size_t arrIndex = 0) {
    switch (type) {
        case ISerializable::Entry::INT:
            static_cast<int*>(wPtr)[arrIndex] = from[index];
            break;
        case ISerializable::Entry::FLOAT:
            static_cast<float*>(wPtr)[arrIndex] = from[index];
            break;
        case ISerializable::Entry::STRING:
            static_cast<std::string*>(wPtr)[arrIndex] = from[index];
            break;
        case ISerializable::Entry::BOOL:
            static_cast<bool*>(wPtr)[arrIndex] = from[index];
            break;
        case ISerializable::Entry::OBJECT:
            static_cast<ISerializable**>(wPtr)[arrIndex]->read(from[index]);
            break;
    }
}

void ISerializable::write(json& out) const {
    std::vector<Entry> entries;
    gatherEntries(entries);
    for (const Entry& e : entries) {
        if (e.count >= 0) {
            out[e.name] = json::array();
            for (size_t i = 0; i < e.count; ++i)
                ::push(out[e.name], e);
        }
        else
            ::write(out, e);
    }
}

void ISerializable::read(const json& in) {
    std::vector<Entry> entries;
    gatherEntries(entries);
    for (const Entry& e : entries) {
        if (!in.is_object())
            throw std::runtime_error("Input json is not an object: " + in.dump());
        if (!in.count(e.name))
            throw std::runtime_error("Input json is missing a property (" + e.name
                                     + "):" + in.dump());
        if (e.count >= 0) {
            const json& array = in[e.name];
            if (!array.is_array())
                throw std::runtime_error(
                  "'" + e.name + "' property should be an array in input json: " + in.dump());
            for (size_t i = 0; i < e.count; ++i)
                ::read(in[e.name], e.type, e.writePtr, i, i);
        }
        else
            ::read(in, e.type, e.writePtr, e.name);
    }
}