#pragma once

#include <cassert>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <vector>

#define M_E 2.71828182845904523536
#define M_LOG2E 1.44269504088896340736
#define M_LOG10E 0.434294481903251827651
#define M_LN2 0.693147180559945309417
#define M_LN10 2.30258509299404568402
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#define M_PI_4 0.785398163397448309616
#define M_1_PI 0.318309886183790671538
#define M_2_PI 0.636619772367581343076
#define M_1_SQRTPI 0.564189583547756286948
#define M_2_SQRTPI 1.12837916709551257390
#define M_SQRT2 1.41421356237309504880
#define M_SQRT_2 0.707106781186547524401

template <typename T, typename P>
T assertReturn(T&& value, P&& predicate) {
    assert(predicate(value));
    return std::forward<T>(value);
}

template <typename Target, typename T>
Target safeCast(T&& value) {
    return static_cast<Target>(assertReturn(std::forward<T>(value), [](T v) {
        if constexpr (std::is_signed<T>::value == std::is_signed<Target>::value) {
            return std::numeric_limits<Target>::min() <= v
                   && v <= std::numeric_limits<Target>::max();
        }
        else {
            if constexpr (std::is_signed<T>::value)
                return v >= static_cast<T>(0)
                       && static_cast<typename std::make_unsigned<std::decay_t<T>>::type>(v)
                            <= std::numeric_limits<Target>::max();
            else
                return v <= static_cast<typename std::make_unsigned<std::decay_t<Target>>::type>(
                              std::numeric_limits<Target>::max());
        }
    }));
}

using CombinationFlag = unsigned int;
enum CombinationBits : CombinationFlag
{
    INCLUDE_BOTH_BIT = 1,
    INCLUDE_ONLY_A_BIT = 2,
    INCLUDE_ONLY_B_BIT = 4,
};

template <typename T>
std::set<T> combination(const std::set<T>& s1, const std::set<T>& s2, CombinationFlag flags) {
    auto itr1 = std::begin(s1);
    auto itr2 = std::begin(s2);
    std::set<T> ret;
    while (itr1 != std::end(s1) || itr2 != std::end(s2)) {
        if (itr1 != std::end(s1) && itr2 != std::end(s2) && *itr1 == *itr2) {
            if (flags & CombinationBits::INCLUDE_BOTH_BIT)
                ret.insert(*itr1);
            itr1++;
            itr2++;
        }
        else if (itr1 == std::end(s1) || itr2 != std::end(s2) && *itr2 < *itr1) {
            if (flags & CombinationBits::INCLUDE_ONLY_B_BIT)
                ret.insert(*itr2);
            itr2++;
        }
        else {
            if (flags & CombinationBits::INCLUDE_ONLY_A_BIT)
                ret.insert(*itr1);
            itr1++;
        }
    }
    return ret;
}

template <typename T>
inline void writeData(std::ostream& out, const T& data);
template <typename T>
inline void readData(std::istream& in, T& data);
template <typename T>
inline void writeData(std::ostream& out, const std::vector<T>& data);
template <typename T>
inline void readData(std::istream& in, std::vector<T>& data);
template <typename T>
inline void writeData(std::ostream& out, const std::unique_ptr<T>& data);
template <typename T>
inline void readData(std::istream& in, std::unique_ptr<T>& data);
template <typename K, typename V>
inline void writeData(std::ostream& out, const std::map<K, V>& data);
template <typename K, typename V>
inline void readData(std::istream& in, std::map<K, V>& data);

template <typename T>
inline void writeData(std::ostream& out, const T& data) {
    out.write(reinterpret_cast<const char*>(&data), sizeof(data));
}

template <typename T>
inline void readData(std::istream& in, T& data) {
    in.read(reinterpret_cast<char*>(&data), sizeof(data));
}

template <typename T>
inline void writeData(std::ostream& out, const std::vector<T>& data) {
    writeData(out, data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        writeData(out, data[i]);
    }
}

template <typename T>
inline void readData(std::istream& in, std::vector<T>& data) {
    data.clear();
    decltype(data.size()) len;
    readData(in, len);
    data.resize(len);
    for (std::size_t i = 0; i < len; ++i) {
        readData(in, data[i]);
    }
}

template <typename T>
inline void writeData(std::ostream& out, const std::unique_ptr<T>& data) {
    bool empty = data == nullptr;
    writeData(out, empty);
    if (!empty)
        writeData(out, *data);
}

template <typename T>
inline void readData(std::istream& in, std::unique_ptr<T>& data) {
    bool empty;
    readData(in, empty);
    if (empty) {
        data = nullptr;
    }
    else {
        if (data == nullptr)
            data.reset(new T{});
        readData(in, *data);
    }
}

template <typename K, typename V>
inline void writeData(std::ostream& out, const std::map<K, V>& data) {
    writeData(out, data.size());
    for (const auto& [k, v] : data) {
        writeData(out, k);
        writeData(out, v);
    }
}

template <typename K, typename V>
inline void readData(std::istream& in, std::map<K, V>& data) {
    data.clear();
    decltype(data.size()) len;
    readData(in, len);
    for (std::size_t i = 0; i < len; ++i) {
        K k;
        V v;
        readData(in, k);
        readData(in, v);
        data[k] = v;
    }
}

template <typename T>
inline void writeData(std::ostream& out, const std::set<T>& data) {
    writeData(out, data.size());
    for (const auto& v : data) {
        writeData(out, v);
    }
}

template <typename T>
inline void readData(std::istream& in, std::set<T>& data) {
    data.clear();
    decltype(data.size()) len;
    readData(in, len);
    for (std::size_t i = 0; i < len; ++i) {
        T v;
        readData(in, v);
        data.insert(v);
    }
}
