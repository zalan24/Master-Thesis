#pragma once

#include <iostream>
#include <vector>

template <typename T>
bool write_data(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return out.good();
}

bool write_string(std::ostream& out, const std::string& value);

template <typename T>
bool write_vector(std::ostream& out, const std::vector<T>& data) {
    uint64_t size = data.size();
    write_data(out, size);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(sizeof(data[0]) * size));
    return out.good();
}

template <typename T>
bool read_data(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return in.good();
}

bool read_string(std::istream& in, std::string& value);

template <typename T>
bool read_vector(std::istream& in, std::vector<T>& data) {
    uint64_t size = 0;
    read_data(in, size);
    data.resize(size);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(sizeof(data[0]) * size));
    return in.good();
}
