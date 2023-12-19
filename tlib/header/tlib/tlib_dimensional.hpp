#ifndef __TLIB__TENSOR_BASE_HPP__
#define __TLIB__TENSOR_BASE_HPP__

#include "tlib_base.hpp"
#include "tlib_util.hpp"
#include <sstream>
#include <stdexcept>
#include <string>

namespace tensorlib {
namespace base {
using tensorlib::base::FixArray;

using Shape = FixArray<SizeType>;

template <typename T>
concept IsShape = std::is_same<Shape, std::decay_t<T>>::value ||
                  std::is_same<std::initializer_list<SizeType>, std::decay_t<T>>::value;

using Index = FixArray<SizeType>;

template <typename T>
concept IsIndex = std::is_same<Index, std::decay_t<T>>::value ||
                  std::is_same<std::initializer_list<SizeType>, std::decay_t<T>>::value;

/**
 * Something has a size
 */
class Sized {
public:
    virtual const Shape &size() const = 0;

    ~Sized() {
    }
};

/**
 * Something with a specific dimension.
 */
class Dimensional : public Sized {
protected:
    // basic dimension of object
    Shape dim;

    // dimension of a specific index
    Shape idxToDim;

    // total shape
    SizeType totalSize;

    // basic size
    Shape basicShape;

    Dimensional(Dimensional &&d) noexcept(std::is_nothrow_move_constructible_v<Shape>)
        : dim(std::move(d.dim)), idxToDim{std::move(d.idxToDim)}, totalSize{d.totalSize},
          basicShape{std::move(d.basicShape)} {
    }

    Dimensional(const Dimensional &d) noexcept(std::is_nothrow_copy_constructible_v<Shape> &&
                                               std::is_nothrow_constructible_v<Shape, SizeType>)
        : dim(d.dim), idxToDim{d.idxToDim}, totalSize{d.totalSize}, basicShape{d.basicShape} {
    }

    Dimensional(Shape dim, Shape idxToDim) noexcept(std::is_nothrow_move_constructible_v<Shape> &&
                                                    std::is_nothrow_constructible_v<Shape, SizeType>)
        : dim(std::move(dim)), idxToDim{std::move(idxToDim)}, totalSize{1}, basicShape(this->idxToDim.size()) {
        for (auto d : this->dim) {
            totalSize *= d;
        }
        for (SizeType i = 0; i < this->idxToDim.size(); i++) {
            basicShape[i] = this->dim[this->idxToDim[i]];
        }
    }

public:
    const Shape &size() const override {
        return basicShape;
    }

    virtual SizeType dimensionSize() const {
        return this->idxToDim.size();
    }

    Dimensional &operator=(Dimensional &&dimensional) noexcept(std::is_nothrow_swappable_v<Shape>) {
        if (this == &dimensional) {
            return *this;
        }
        std::swap(dim, dimensional.dim);
        std::swap(idxToDim, dimensional.idxToDim);
        std::swap(totalSize, dimensional.totalSize);

        return *this;
    }

    Dimensional &operator=(const Dimensional &dimensional) noexcept(std::is_nothrow_copy_assignable_v<Shape>) {
        if (this == &dimensional) {
            return *this;
        }
        this->dim = dimensional.dim;
        this->idxToDim = dimensional.idxToDim;
        this->totalSize = dimensional.totalSize;
        return *this;
    }

    virtual ~Dimensional() {
    }
};

/**
 * Offset dimensional is simply a dimensional that stores offset of dim
 */
class OffsetDimensional : public Dimensional {
protected:
    template <typename Index, typename Dimension, typename Arrange, typename Offset>
        requires IsIndex<Index> && IsShape<Dimension> && IsShape<Arrange> && IsShape<Offset>
    constexpr static SizeType getIndexByOffsetDimension(Index &&idx, Dimension &&dim, Arrange &&idxToDim,
                                                        Offset &&offset,
                                                        SizeType off = 0) noexcept(!OUT_OF_RANGE_CHECK) {

        if constexpr (OUT_OF_RANGE_CHECK) {
            if (idxToDim.size() > dim.size()) {
                throw std::out_of_range(std::string{"Arrangement size is greater than Dimension size"});
            }
            if (idx.size() > dim.size()) {
                throw std::out_of_range(std::string{"Too much index"});
            }
        }

        for (SizeType i = 0, n = idx.size(); i < n; i++) {
            off += idx[i] * offset[idxToDim[i]];
        }

        return off;
    }

protected:
    // offset is the offset of index of dim
    Shape offset;

    OffsetDimensional(const OffsetDimensional &d) noexcept(std::is_nothrow_copy_constructible<Dimensional>::value &&
                                                           std::is_nothrow_copy_constructible<Shape>::value) = default;

public:
    OffsetDimensional() = delete;

    OffsetDimensional(Shape dimension,
                      Index arrangement) noexcept(std::is_nothrow_constructible<Dimensional, Shape, Index>::value &&
                                                  std::is_nothrow_move_constructible<Shape>::value &&
                                                  std::is_nothrow_move_constructible<Index>::value &&
                                                  std::is_nothrow_constructible<Shape, SizeType>::value)
        : Dimensional(std::move(dimension), std::move(arrangement)), offset(dim.size()) {
        auto &dim = this->dim;
        if (dim.size() > 0) {
            offset[dim.size() - 1] = 1;
            for (SizeType j = dim.size() - 1; j > 0; j--) {
                offset[j - 1] = offset[j - 1 + 1] * dim[j];
            }
        }
    }

    OffsetDimensional &operator=(const OffsetDimensional &d) noexcept(std::is_nothrow_copy_assignable_v<Shape>) {
        if (this == &d) {
            return *this;
        }
        this->offset = d.offset;
        Dimensional::operator=(d);
        return *this;
    }

    OffsetDimensional &operator=(OffsetDimensional &&d) noexcept(std::is_nothrow_swappable_v<Shape>) {
        if (this == &d) {
            return *this;
        }
        std::swap(offset, d.offset);
        Dimensional::operator=(std::move(d));
        return *this;
    }

    virtual ~OffsetDimensional() {
    }
};

/**
 * Something supports index
 */
template <typename T>
class Indexable {
public:
    virtual T operator[](const Index &idx) = 0;

    virtual ~Indexable() {
    }
};

template <typename T>
class Gettable {
public:
    virtual T &get(const Index &idx) noexcept(!OUT_OF_RANGE_CHECK) = 0;

    virtual const T &get(const Index &idx) const noexcept(!OUT_OF_RANGE_CHECK) = 0;

    virtual ~Gettable() {
    }
};

inline std::string shapeToString(const Shape &s) {
    std::stringstream ss;
    ss << '(';
    for (SizeType i = 0, n = s.size(); i < n; i++) {
        if (i > 0) {
            ss << ',';
        }
        ss << s[i];
    }
    ss << ')';
    return ss.str();
}

}; // namespace base

using base::Index;
using base::Shape;

}; // namespace tensorlib

inline std::ostream &operator<<(std::ostream &out, const tensorlib::base::Shape &s) {
    out << tensorlib::base::shapeToString(s);
    return out;
}

#endif