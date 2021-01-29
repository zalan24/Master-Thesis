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
