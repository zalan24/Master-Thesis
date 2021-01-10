#include "meshprovider.h"

MeshProvider::~MeshProvider() {
}

MeshProvider::ResourceDescriptor::ResourceDescriptor(Type _type, bool _flipYZ)
  : type(_type), filename(""), flipYZ(_flipYZ) {
    assert(type != FILE);
}

MeshProvider::ResourceDescriptor::ResourceDescriptor(const std::string& _filename, bool _flipYZ)
  : type(FILE), filename(_filename), flipYZ(_flipYZ) {
}

void MeshProvider::ResourceDescriptor::gatherEntries(std::vector<Entry>& entries) const {
    // TODO register enum
    REGISTER_ENTRY(filename, entries);
    REGISTER_ENTRY(flipYZ, entries);
}
