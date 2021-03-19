#include "binary_io.h"

#include <util.hpp>

void write_string(std::ostream& out, const std::string& value) {
    uint64_t size = value.length();
    write_data(out, size);
    out.write(value.data(), safe_cast<std::streamsize>(size));
}

void read_string(std::istream& in, std::string& value) {
    uint64_t size = 0;
    read_data(in, size);
    value.resize(size);
    in.read(value.data(), safe_cast<std::streamsize>(size));
}
