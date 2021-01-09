#include "meshprovider.h"

MeshProvider::~MeshProvider() {
}

MeshProvider::ResourceDescriptor::ResourceDescriptor(Type _type) : type(_type), filename("") {
    assert(type != FILE);
}

MeshProvider::ResourceDescriptor::ResourceDescriptor(const std::string& _filename)
  : type(FILE), filename(_filename) {
}

void MeshProvider::ResourceDescriptor::gatherEntries(std::vector<Entry>& entries) const {
    // TODO register enum
    REGISTER_ENTRY(filename, entries);
}
