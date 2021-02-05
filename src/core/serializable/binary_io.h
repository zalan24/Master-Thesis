#pragma once

#include <iostream>
#include <vector>

template <typename T>
void write_data(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_string(std::ostream& out, const std::string& value);

template <typename T>
void write_vector(std::ostream& out, const std::vector<T>& data) {
    uint64_t size = data.size();
    write_data(out, size);
    out.write(reinterpret_cast<const char*>(data.data()), sizeof(data[0]) * size);
}
