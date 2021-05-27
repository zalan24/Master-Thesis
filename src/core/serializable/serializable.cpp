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
    writeBin(ss);
    std::vector<char> binary(size_t(ss.gcount()));
    ss.read(binary.data(), ss.gcount());
    return hash_binary(binary.size(), reinterpret_cast<const void*>(binary.data()));
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

bool ISerializable::exportToFile(const fs::path& p) const {
    if (p.extension() == ".json") {
        std::ofstream out(p.c_str());
        if (!out.is_open())
            return false;
        json j;
        writeJson(j);
        out << j;
        return !out.fail();
    }
    else if (p.extension() == ".bin") {
        std::ofstream out(p.c_str(), std::ios::binary);
        if (!out.is_open())
            return false;
        return writeBin(out);
    }
    else
        throw std::runtime_error("Unknown file extension: " + p.string());
}

bool ISerializable::importFromFile(const fs::path& p) {
    if (p.extension() == ".json") {
        std::ifstream in(p.c_str());
        if (!in.is_open())
            return false;
        json j;
        in >> j;
        readJson(j);
        return !in.fail();
    }
    else if (p.extension() == ".bin") {
        std::ifstream in(p.c_str(), std::ios::binary);
        if (!in.is_open())
            return false;
        return readBin(in);
    }
    else
        throw std::runtime_error("Unknown file extension: " + p.string());
}
