#pragma once

#include <algorithm>
#include <array>
#include <iostream>

#include "polynomial.hpp"

template <typename T, typename V = T>
class Spline
{
    using Poly = Polynomial<T, V>;
    using Sample = typename Poly::Sample;

 public:
    void start(const V& start, const std::vector<T>& startValues, const V& _end,
               const std::vector<T>& endValues) {
        ASSERT(startValues.size() + endValues.size() > 0);
        deg = static_cast<unsigned int>(startValues.size() + endValues.size() - 1);
        segments.clear();
        std::vector<Sample> samples(deg + 1);
        for (unsigned int i = 0; i < startValues.size(); ++i)
            samples[i] = Sample{start, startValues[i], i};
        for (unsigned int i = 0; i < endValues.size(); ++i)
            samples[i + startValues.size()] = Sample{_end, endValues[i], i};
        segments.push_back({start, Poly::samples(samples, deg)});
        end = _end;
    }
    void add(const V& _end, const std::vector<T>& values) {
        ASSERT(segments.size() > 0);
        ASSERT(end < _end);
        ASSERT(deg + 1 > values.size());
        std::vector<Sample> samples(deg + 1);
        for (unsigned int i = 0; i < values.size(); ++i)
            samples[i] = Sample{_end, values[i], i};
        for (unsigned int i = 0; i < deg + 1 - values.size(); ++i)
            samples[i + values.size()] =
              Sample{end, segments.back().polynomial.derivative(i)(end), i};
        segments.push_back({end, Poly::samples(samples, deg)});
        end = _end;
    }

    T operator()(const V& x) const {
        ASSERT(segments.size() > 0);
        auto itr =
          std::lower_bound(std::begin(segments), std::end(segments), x,
                           [](const Segment& lhs, const V& rhs) { return lhs.start < rhs; });
        if (itr == std::end(segments))
            return segments.back().polynomial(x);
        if (itr != std::begin(segments))
            itr--;
        return itr->polynomial(x);
    }

    Spline derivative(unsigned int d = 1) const {
        if (d == 0)
            return *this;
        Spline ret;
        ret.segments.reserve(segments.size());
        for (const auto& s : segments)
            ret.segments.push_back({s.start, s.polynomial.derivative(d)});
        ret.end = end;
        return ret;
    }

    void draw(std::ostream& out, unsigned int width = 120, unsigned int height = 20) const {
        ASSERT(segments.size() > 0);
        std::vector<std::string> lines(height, std::string(width, ' '));
        T min = std::numeric_limits<T>::infinity();
        T max = -std::numeric_limits<T>::infinity();
        for (unsigned int i = 0; i < width; ++i) {
            const V x =
              ::lerp(segments.front().start, end, static_cast<V>(i) / static_cast<V>(width));
            const T y = (*this)(x);
            min = std::min(min, y);
            max = std::max(max, y);
        }
        out << '(' << segments.front().start << ':' << end << ")x(" << min << ':' << max << ")\n";
        for (unsigned int i = 0; i < width; ++i) {
            const V x =
              ::lerp(segments.front().start, end, static_cast<V>(i) / static_cast<V>(width));
            const T y = (*this)(x);
            unsigned int l =
              static_cast<unsigned int>((y - min) / (max - min) * static_cast<T>(height - 1));
            lines[height - l - 1][i] = '.';
        }
        std::string bottomLine = std::string(width + 2, '-');
        for (const auto& segment : segments) {
            const V x = (segment.start - segments.front().start) / (end - segments.front().start)
                        * static_cast<V>(width - 1);
            bottomLine[static_cast<unsigned int>(x) + 1] = '+';
        }
        out << bottomLine << '\n';
        for (const auto& line : lines)
            out << '|' << line << "|\n";
        out << bottomLine << std::endl;
    }

    unsigned int degree() const { return deg; }

    Range<V, 1> range() const {
        ASSERT(segments.size() > 0);
        return Range<V, 1>{segments.front().start, end};
    }

    unsigned int segmentCount() const { return safe_cast<unsigned int>(segments.size()); }
    const V& segmentStart(unsigned int index) const {
        ASSERT(index < segments.size());
        return segments[index].start;
    }
    const V& segmentEnd(unsigned int index) const {
        ASSERT(index < segments.size());
        if (index + 1 < segmentCount())
            return segments[index + 1].start;
        else
            return end;
    }

 private:
    unsigned int deg;
    struct Segment
    {
        V start;
        Poly polynomial;
    };
    std::vector<Segment> segments;
    V end;
};
