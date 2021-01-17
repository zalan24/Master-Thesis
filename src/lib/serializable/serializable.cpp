#include "serializable.h"

void ISerializable::write(std::ostream& out) const {
    json j;
    writeJson(j);
    out << j;
}

void ISerializable::read(std::istream& in) {
    json j;
    in >> j;
    readJson(j);
}

template <>
ISerializable::Entry::Type getType<int>() {
    return ISerializable::Entry::INT;
}
template <>
ISerializable::Entry::Type getType<float>() {
    return ISerializable::Entry::FLOAT;
}
template <>
ISerializable::Entry::Type getType<std::string>() {
    return ISerializable::Entry::STRING;
}
template <>
ISerializable::Entry::Type getType<bool>() {
    return ISerializable::Entry::BOOL;
}
template <>
ISerializable::Entry::Type getType<ISerializable*>() {
    return ISerializable::Entry::OBJECT;
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
            (*static_cast<const ISerializable* const*>(e.readPtr))->writeJson(obj);
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
            (*static_cast<const ISerializable* const*>(e.readPtr))->writeJson(obj);
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
            static_cast<ISerializable**>(wPtr)[arrIndex]->readJson(from[index]);
            break;
    }
}
