/**
 * CITE:
 * The read & write float library is used from:
 * https://github.com/MalcolmMcLean/ieee754
 * From the author Malcolm McLean
 */

#ifndef __TLIB__UTIL_HPP__
#define __TLIB__UTIL_HPP__

#include "tlib_base.hpp"
#include <bit>
#include <cassert>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

namespace tensorlib {

namespace util {
namespace functional {

/**
 * Apply a function on iterators between [it, end) according to its
 * Notice we expect users to ensure the size of its must >= size of [it, end), otherwise it is undefined behaviour
 * @param it iterator starts
 * @param end end
 * @param f a function
 * @param its all iterators applying on f
 */
template <typename It, typename F, typename... Its>
void transformOnIterator(It &&it, It &&end, F &&f, Its &&...its) noexcept(
    (noexcept(*it) && noexcept(++it) && noexcept(it != it)) &&                              // no except for it
    ((noexcept(*it = f((*its)...))) && (noexcept(*its) && ...) && (noexcept(++its) && ...)) // no except for f and its
) {
    for (; it != end; ++it, ((++its), ...)) {
        (*it) = f((*its)...);
    }
}

/**
 * A wrapper function of apply_function_on_iterator according to f
 */
template <typename T, typename F, typename... Iterables>
void transformOnIterable(T &&iterable, F &&f, Iterables &&...iterables) {
    transformOnIterator(iterable.begin(), iterable.end(), std::forward<F>(f), (iterables.begin())...);
}

/**
 * Apply a function on iterators between [it, end) according to its
 * Notice we expect users to ensure the size of its must >= size of [it, end), otherwise it is undefined behaviour
 * @param it iterator starts
 * @param end end
 * @param f a function
 * @param its all iterators applying on f
 */
template <typename It, typename F, typename... Its>
void applyOnIterator(It &&it, It &&end, F &&f, Its &&...its) noexcept(
    (noexcept(*it) && noexcept(++it) && noexcept(it != it)) &&                             // no except for it
    ((noexcept(f(*it, (*its)...))) && (noexcept(*its) && ...) && (noexcept(++its) && ...)) // no except for f and its
) {
    for (; it != end; ++it, ((++its), ...)) {
        f((*it), (*its)...);
    }
}

/**
 * A wrapper function of apply_function_on_iterator according to f
 */
template <typename T, typename F, typename... Iterables>
void applyOnIterable(T &&iterable, F &&f, Iterables &&...iterables) {
    applyOnIterator(iterable.begin(), iterable.end(), std::forward<F>(f), (iterables.begin())...);
}

std::string messageOf(auto &&...args) {
    return ((std::stringstream()) << ... << std::forward<decltype(args)>(args)).str();
}

}; // namespace functional

template <typename T>
struct DataIO {
    static T read(std::istream &in) {
        T t;
        in >> t;
        return t;
    }

    static void write(std::ostream &out, const T &t) {
        out << t << ' ';
    }
};

template <>
struct DataIO<float> {
    /**
     * Read float in istream
     * @param in istream
     * @return float read
     */
    static float read(std::istream &in) {
        auto readChar = [&in]() {
            char c;
            in.read(&c, 1);
            return c;
        };
        if constexpr (std::numeric_limits<float>::is_iec559) {
            float f;
            auto *cs = reinterpret_cast<unsigned char *>(&f);
            if constexpr (std::endian::native == std::endian::big) {
                for (int i = 0; i < sizeof(float); i++)
                    cs[i] = readChar();
            } else {
                for (int i = sizeof(float); i > 0; i--)
                    cs[i - 1] = readChar();
            }
            return f;
        } else {
            float f;
            in >> f;
            return f;
        }
    }

    /**
     * Write float into ostream
     * @param out ostream
     * @param x float
     */
    static void write(std::ostream &out, float x) {
        auto writeChar = [&out](char &c) { out.write(&c, 1); };
        if constexpr (std::numeric_limits<double>::is_iec559) {
            char *cs = reinterpret_cast<char *>(&x);
            if constexpr (std::endian::native == std::endian::big) {
                for (int i = 0; i < sizeof(float); i++)
                    writeChar(cs[i]);
            } else {
                for (int i = sizeof(float); i > 0; i--)
                    writeChar(cs[i - 1]);
            }
        } else {
            out << x;
        }
    }
};

/**
 * Similar to DataIO<float>
 */
template <>
struct DataIO<double> {
    static double read(std::istream &in, int bigendian = 0) {
        auto readChar = [&in]() {
            char c;
            in.read(&c, 1);
            return c;
        };
        if constexpr (std::numeric_limits<double>::is_iec559) {
            double f;
            auto *cs = reinterpret_cast<unsigned char *>(&f);
            if constexpr (std::endian::native == std::endian::big) {
                for (int i = 0; i < sizeof(double); i++)
                    cs[i] = readChar();
            } else {
                for (int i = sizeof(double); i > 0; i--)
                    cs[i - 1] = readChar();
            }
            return f;
        } else {
            double f;
            in >> f;
            return f;
        }
    }

    static void write(std::ostream &out, double x) {
        auto writeChar = [&out](char &c) { out.write(&c, 1); };
        if constexpr (std::numeric_limits<double>::is_iec559) {
            char *cs = reinterpret_cast<char *>(&x);
            if constexpr (std::endian::native == std::endian::big) {
                for (int i = 0; i < sizeof(double); i++)
                    writeChar(cs[i]);
            } else {
                for (int i = sizeof(double); i > 0; i--)
                    writeChar(cs[i - 1]);
            }
        } else {
            out << x;
        }
    }
};

}; // namespace util
}; // namespace tensorlib

#endif