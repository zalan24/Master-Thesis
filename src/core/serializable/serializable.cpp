#include "serializable.h"

#include <HashLib4CPP.h>

std::string hash_string(const std::string& data) {
    if (data == "")
        return "-";
    IHash hash = HashLib4CPP::Hash128::CreateMurmurHash3_x64_128();
    IHashResult res = hash->ComputeString(data);
    return res->ToString();
}

std::string hash_binary(size_t size, const void* data) {
    IHash hash = HashLib4CPP::Hash128::CreateMurmurHash3_x64_128();
    // int64_t... what a good choice for memory size...
    IHashResult res = hash->ComputeUntyped(data, static_cast<int64_t>(size));
    return res->ToString();
}

std::string ISerializable::hash() const {
    std::stringstream ss;
    write(ss);
    return hash_string(ss.str());
}

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

void IVirtualSerializable::writeJson(json& out) const {
    std::string type = getCurrentType();
    const ISerializable* content = getCurrent();
    if (content == nullptr) {
        type = "";
        WRITE_OBJECT(type, out);
    }
    else {
        WRITE_OBJECT(type, out);
        WRITE_OBJECT(content, out);
    }
}

void IVirtualSerializable::readJson(const json& in) {
    reset();
    std::string type;
    READ_OBJECT_OPT(type, in, "");
    if (type != "") {
        ISerializable* content = init(type);
        READ_OBJECT(content, in);
    }
}
