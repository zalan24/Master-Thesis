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

#ifdef PRINT_STACK
#    include <backwardcppconfig.h>
inline void printStack() {
    using namespace backward;
    StackTrace st;
    st.load_here(32);
    TraceResolver tr;
    tr.load_stacktrace(st);
    for (size_t i = 0; i < st.size(); ++i) {
        ResolvedTrace trace = tr.resolve(st[i]);
        std::cout << "#" << i << " ";
        if (trace.source.filename == "")
            std::cout << trace.object_filename << ": " << trace.object_function << "[" << trace.addr
                      << "]" << std::endl;
        else
            std::cout << trace.source.filename << ":" << trace.source.line << ":"
                      << trace.source.col << ": " << trace.source.function << std::endl;
    }
}
#else
inline void printStack() {
}
#endif

#ifdef DEBUG
#    define ASSERT(x)         \
        do {                  \
            if (!(x))         \
                printStack(); \
            assert(x);        \
        } while (false)
#else
#    define ASSERT(x) (static_cast<void>(sizeof(x)))
#endif

// template <typename T, typename T2>
// T safe_cast(const T2& value) {
// #ifdef DEBUG
//     ASSERT(static_cast<T2>(static_cast<T>(value)) == value);
// #endif
//     return static_cast<T>(value);
// }

template <typename T, unsigned int D>
class Range
{
 public:
    constexpr Range(const T& start = 0, const T& end = 0) {
        for (unsigned int i = 0; i < D; ++i) {
            a[i] = start;
            b[i] = end;
        }
    }

    void set(const T& start, const T& end, unsigned int d = 0) {
        a[d] = start;
        b[d] = end;
    }

    const T& start(unsigned int d = 0) const { return a[d]; }

    const T& end(unsigned int d = 0) const { return b[d]; }

    T range(unsigned int d = 0) const { return b[d] - a[d]; }

    bool contains(T& value, unsigned d = 0) const { return a[d] <= value && value < b[d]; }
    T center(unsigned int d = 0) const { return (a[d] + b[d]) / static_cast<T>(2); }

    template <typename F>
    T lerp(F&& f, unsigned int d = 0) const {
        return ::lerp(a[d], b[d], std::forward<F>(f));
    }

    T area() const {
        T ret{1};
        for (unsigned int d = 0; d < D; ++d) {
            if (b[d] < a[d])
                return T{0};
            ret *= range(d);
        }
        return ret;
    }

    friend Range intersection(const Range& _a, const Range& _b) {
        Range ret;
        for (unsigned int d = 0; d < D; ++d) {
            ret.set(std::max(_a.start(d), _b.start(d)), std::min(_a.end(d), _b.end(d)), d);
            if (ret.start(d) > ret.end(d))
                ret.set(ret.start(d), ret.start(d), d);
        }
        return ret;
    }

 private:
    T a[D] = {T{0}};
    T b[D] = {T{0}};
};

template <typename T, typename P>
T assert_return(T&& value, P&& predicate) {
    ASSERT(predicate(value));
    return std::forward<T>(value);
}

template <typename Target, typename T>
Target safe_cast(T&& value) {
    return static_cast<Target>(assert_return(std::forward<T>(value), [](T v) {
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

