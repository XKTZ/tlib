#ifndef __TLIB__BASIC_HPP__
#define __TLIB__BASIC_HPP__

#include <cstdlib>
#include <iostream>
#include <random>
#include <utility>

#include "tlib/tlib_config.hpp"
#include "tlib/tlib_util.hpp"

namespace tensorlib {

namespace base {
template <typename T>
concept IsArrayNoExceptDefaultConstruct = NEW_NOEXCEPT && std::is_nothrow_default_constructible<T>::value;

template <typename T>
concept IsArrayNoExceptCopyConstruct = NEW_NOEXCEPT && std::is_nothrow_copy_constructible<T>::value;

template <typename T>
concept IsArrayNoExceptMoveConstruct = NEW_NOEXCEPT && std::is_nothrow_move_constructible<T>::value;

template <typename T>
concept IsArrayNoExceptCopyAssign = NEW_NOEXCEPT && std::is_nothrow_copy_assignable<T>::value;

template <typename T>
concept IsArrayNoExceptMoveAssign = NEW_NOEXCEPT && std::is_nothrow_move_assignable<T>::value;

/**
 * A fixed length array. The length of the array is fixed, while the items in it is mutable.
 */
template <typename T, bool RangeCheck = OUT_OF_RANGE_CHECK>
class FixArray final {
private:
    static constexpr bool DEFAULT_CONSTRUCTOR_NOEXCEPT = IsArrayNoExceptDefaultConstruct<T>;
    static constexpr bool COPY_CONSTRUCTOR_NOEXCEPT = IsArrayNoExceptCopyConstruct<T>;
    static constexpr bool MOVE_CONSTRUCTOR_NOEXCEPT = IsArrayNoExceptMoveConstruct<T>;
    static constexpr bool COPY_ASSIGNMENT_NOEXCEPT = IsArrayNoExceptCopyAssign<T>;
    static constexpr bool MOVE_ASSIGNMENT_NOEXCEPT = IsArrayNoExceptMoveAssign<T>;

    using Iterator = T *;
    using ConstIterator = const T *;

    SizeType len;

    T *data;

    /**
     * Release its resource occupying
     */
    void releaseResource() noexcept {
        len = 0;
        delete[] data;
    }

    /**
     * Swap with another array
     * @param a another array a
     */
    void swapWith(FixArray<T> &a) noexcept {
        std::swap(len, a.len);
        std::swap(data, a.data);
    }

public:
    /**
     * Create by size
     */
    explicit FixArray(const SizeType len) noexcept(DEFAULT_CONSTRUCTOR_NOEXCEPT)
        : len{len}, data{static_cast<T *>(operator new[](sizeof(T) * len))} {
        if constexpr (COPY_CONSTRUCTOR_NOEXCEPT) {
            for (SizeType i = 0; i < len; i++) {
                new (&data[i]) T();
            }
        } else {
            SizeType on = 0;
            try {
                for (SizeType i = 0; i < len; i++) {
                    new (&data[on++]) T();
                }
            } catch (...) {
                on--;
                for (; on > 0; on--) {
                    data[on - 1].~T();
                }
                operator delete[](data);
                throw;
            }
        }
    }
    /**
     * Create by size
     */
    explicit FixArray(const SizeType len, const T &t) noexcept(DEFAULT_CONSTRUCTOR_NOEXCEPT)
        : len{len}, data{static_cast<T *>(operator new[](sizeof(T) * len))} {
        if constexpr (COPY_CONSTRUCTOR_NOEXCEPT) {
            for (SizeType i = 0; i < len; i++) {
                new (&data[i]) T(t);
            }
        } else {
            SizeType on = 0;
            try {
                for (SizeType i = 0; i < len; i++) {
                    new (&data[on++]) T(t);
                }
            } catch (...) {
                on--;
                for (; on > 0; on--) {
                    data[on - 1].~T();
                }
                operator delete[](data);
                throw;
            }
        }
    }

    /**
     * Create by using initializer list, strong guarentee
     */
    template <typename S>
        requires std::is_constructible_v<T, S>
    FixArray(const std::initializer_list<S> &lst) noexcept(COPY_CONSTRUCTOR_NOEXCEPT)
        : len{lst.size()}, data{static_cast<T *>(operator new[](sizeof(T) * lst.size()))} {
        if constexpr (COPY_CONSTRUCTOR_NOEXCEPT) {
            SizeType on = 0;
            for (auto x : lst) {
                new (&data[on]) T(x);
                on++;
            }
        } else {
            SizeType on = 0;
            try {
                for (auto x : lst) {
                    new (data + (on++)) T(x);
                }
            } catch (...) {
                on--;
                for (; on > 0; on--) {
                    data[on - 1].~T();
                }
                operator delete[](data);
                throw;
            }
        }
    }

    /**
     * Create using copy constructor, strong guarentee
     * @param dt data
     */
    FixArray(const FixArray &dt) noexcept(COPY_CONSTRUCTOR_NOEXCEPT)
        : len{dt.len}, data{static_cast<T *>(operator new[](sizeof(T) * dt.len))} {
        if constexpr (COPY_CONSTRUCTOR_NOEXCEPT) {
            for (SizeType i = 0; i < len; i++) {
                new (data + i) T(dt.data[i]);
            }
        } else {
            SizeType on = 0;
            try {
                while (on < len) {
                    new (&data[on]) T(dt.data[on]);
                    on++;
                }
            } catch (...) {
                on--;
                for (; on > 0; on--) {
                    data[on - 1].~T();
                }
                operator delete[](data);
                throw;
            }
        }
    }

    /**
     * Create using a move constructor
     * @param dt data
     */
    FixArray(FixArray &&dt) noexcept : len{dt.len}, data{dt.data} {
        dt.len = 0;
        dt.data = nullptr;
    }

    /**
     * Get the size
     * @return size
     */
    SizeType size() const noexcept {
        return len;
    }

    T &back() noexcept {
        return data[len - 1];
    }

    const T &back() const noexcept {
        return data[len - 1];
    }

    T &front() noexcept {
        return data[0];
    }

    const T &front() const noexcept {
        return data[0];
    }

    /**
     * Non const indexer
     * @param idx index
     * @return item at idx
     */
    T &operator[](SizeType idx) noexcept(!RangeCheck) {
        if constexpr (RangeCheck) {
            if (idx >= len) {
                throw std::out_of_range(util::functional::messageOf("Out of range ", data, ": ", idx, ">=", len));
            }
            return data[idx];
        } else {
            return data[idx];
        }
    }

    /**
     * Const indexer
     * @param idx index
     * @return item at idx
     */
    const T &operator[](SizeType idx) const noexcept(!RangeCheck) {
        if constexpr (RangeCheck) {
            if (idx >= len) {
                throw std::out_of_range(util::functional::messageOf("Out of range ", data, ": ", idx, ">=", len));
            }
            return data[idx];
        } else {
            return data[idx];
        }
    }

    /**
     * Const copier
     * @param arr array
     * @return this
     */
    FixArray<T> &operator=(const FixArray<T> &arr) noexcept(COPY_ASSIGNMENT_NOEXCEPT) {
        if (this == &arr) {
            return *this;
        }
        FixArray<T> tmp(arr);

        swapWith(tmp);

        return *this;
    }

    /**
     * Move
     * @param arr array
     * @return this
     */
    FixArray<T> &operator=(FixArray<T> &&arr) noexcept {
        if (this == &arr) {
            return *this;
        }

        releaseResource();
        this->len = arr.len;
        this->data = arr.data;
        arr.len = 0;
        arr.data = nullptr;

        return *this;
    }

    template <typename S>
    bool operator==(const FixArray<S> &b) const noexcept {
        if (this->size() != b.size()) {
            return false;
        }
        for (SizeType i = 0, n = this->size(); i < n; i++) {
            if (!(data[i] == b[i])) {
                return false;
            }
        }
        return true;
    }

    Iterator begin() noexcept {
        return data;
    }

    Iterator end() noexcept {
        return data + len;
    }

    ConstIterator begin() const noexcept {
        return data;
    }

    ConstIterator end() const noexcept {
        return data + len;
    }

    ~FixArray() {
        releaseResource();
    }

public:
    static FixArray<T, RangeCheck> generateIncreasingSequence(SizeType n) {
        FixArray<T, RangeCheck> arr(n);
        for (SizeType i = 0; i < n; i++) {
            arr[i] = i;
        }
        return arr;
    }

    template <SizeType n>
    static FixArray<T, RangeCheck> generateIncreasingSequence() {
        FixArray<T, RangeCheck> arr(n);
        for (SizeType i = 0; i < n; i++) {
            arr[i] = i;
        }
        return arr;
    }

    static FixArray<T, RangeCheck> generateSequenceInRange(SizeType l, SizeType r) {
        FixArray<T, RangeCheck> arr(r - l);
        for (SizeType i = 0; i < r - l; i++) {
            arr[i] = l + i;
        }
        return arr;
    }
};

}; // namespace base

namespace random {

std::mt19937 randomGen(std::random_device{}());

void manualSeed(SizeType seed) {
    randomGen = std::mt19937(seed);
}

template <typename T>
T uniform(T from, T to) {
    return std::uniform_real_distribution{from, to}(randomGen);
}

}; // namespace random

}; // namespace tensorlib

#endif