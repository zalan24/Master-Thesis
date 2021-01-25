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
