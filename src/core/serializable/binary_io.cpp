#include "binary_io.h"

void write_string(std::ostream& out, const std::string& value) {
    uint64_t size = value.length();
    write_data(out, size);
    out.write(value.c_str(), size);
}
