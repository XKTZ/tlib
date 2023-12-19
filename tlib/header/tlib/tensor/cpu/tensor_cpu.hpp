#ifndef __TENSOR__CPU_HPP__
#define __TENSOR__CPU_HPP__

#include "tlib/tensor/tensor.hpp"
#include "tlib/tlib_base.hpp"
#include <cassert>
#include <cmath>
#include <memory>
#include <optional>
#include <type_traits>

namespace tensorlib {
namespace tensor {

namespace device {

/**
 * Using CPU
 */
struct CPU {};
}; // namespace device

namespace cpu {
/**
 * Basic data in a tensor
 */
template <typename T>
using TensorData = tensorlib::base::FixArray<T>;

/**
 * CPU tensor base
 * @tparam T T
 */
template <typename T>
class TensorBase : public tensorlib::base::OffsetDimensional,
                   public tensorlib::base::Indexable<TensorBase<T>>,
                   public tensorlib::base::Gettable<T> {
    using Index = tensorlib::base::Index;
    using Shape = tensorlib::base::Shape;
    using OffsetDimensional = tensorlib::base::OffsetDimensional;
    using Indexer = std::function<Index(const Index &)>;

private:
    std::shared_ptr<TensorData<T>> data;

    /**
     * The offset of index
     */
    SizeType off;

    /**
     * Indexer
     */
    std::optional<Indexer> indexer;

    /**
     * Real shape
     */
    Shape shape;

    /**
     * Index a specific index
     * @param idx idx
     * @return offset of the index in data
     */
    SizeType indexOn(const Index &idx) const noexcept(!OUT_OF_RANGE_CHECK) {
        if (indexer) {
            return OffsetDimensional::getIndexByOffsetDimension(indexer.value()(idx), this->dim, this->idxToDim,
                                                                this->offset, this->off);
        } else {
            return OffsetDimensional::getIndexByOffsetDimension(idx, this->dim, this->idxToDim, this->offset,
                                                                this->off);
        }
    }

public:

    /**
     * Iterator of the base, has provided contiguous optimization
     */
    struct TensorBaseIterator final {
    private:
        union {
            TensorBase<T> *tb;
            T *ptr;
        };

        Index idx;
        Shape dim;

        bool ends{};

        bool isCont;

        /**
         * Is contiguous or not
         */
        bool isContiguous() const {
            return isCont;
        }

    public:
        /**
         * Create a contiguous tensor base iterator by using pointer
         * @param p
         */
        explicit TensorBaseIterator(T *p) : ptr(p), idx(0), dim(0), ends{false}, isCont(true) {
        }

        /**
         * Provide a non contiguous tensor base iterator by providing pointer as well as index on, and ends or not
         * @param tb tensor base
         * @param idx idx
         * @param ends ends or not
         */
        TensorBaseIterator(TensorBase<T> *tb, Index idx, bool ends = false)
            : tb(tb), idx(std::move(idx)), dim(tb->size()), ends(ends), isCont(false) {
        }

        TensorBaseIterator &operator++() {
            if (isContiguous()) {
                ++ptr;
            } else {
                if (idx.size() == 0) {
                    ends = true;
                    return *this;
                }
                auto on = idx.size();
                idx[on - 1]++;
                while (on > 0 && idx[on - 1] == dim[on - 1]) {
                    if (on > 1) {
                        idx[on - 1] -= dim[on - 1];
                        idx[on - 2]++;
                    }
                    on--;
                }
                if (on == 0) {
                    ends = true;
                }
            }
            return *this;
        }

        bool operator!=(const TensorBaseIterator &b) const {
            if (isContiguous() != b.isContiguous()) {
                return false;
            }
            if (isContiguous()) {
                return ptr != b.ptr;
            } else {
                return !((tb == b.tb) && ((ends == b.ends) && ((ends && b.ends) || (idx == b.idx))));
            }
        }

        T &operator*() const {
            if (isContiguous()) {
                return *ptr;
            } else {
                return tb->get(idx);
            }
        }

        ~TensorBaseIterator() {
        }
    };

    /**
     * Create empty tensor base with size 1
     */
    TensorBase()
        : OffsetDimensional(Shape(0), Shape(0)), data(std::make_shared<TensorData<T>>(1)), off{0}, indexer{nullptr},
          shape(Shape(0)) {
    }

    /**
     * Create tensor base through data, size is (data.size())
     * @param data data
     * @param off offset
     * @param indexer indexer
     */
    explicit TensorBase(TensorData<T> data, SizeType off = 0, std::optional<Indexer> indexer = std::nullopt)
        : OffsetDimensional(Shape{data.size()}, Shape{0}), data(std::make_shared<TensorData<T>>(std::move(data))),
          off{off}, indexer{std::move(indexer)}, shape(OffsetDimensional::size()) {
    }

    /**
     * Create tensor base through data pointer, shape, index of DIMENSION, offset, and indexer
     * @param data data
     * @param shp shp
     * @param idx idx
     * @param off offset
     * @param indexer indexer
     */
    TensorBase(std::shared_ptr<TensorData<T>> data, Shape shp, Index idx, SizeType off = 0,
               std::optional<Indexer> indexer = std::nullopt)
        : OffsetDimensional(std::move(shp), std::move(idx)), data(std::move(data)), off{off},
          indexer{std::move(indexer)}, shape(OffsetDimensional::size()) {
    }

    TensorBase(const TensorBase<T> &b)
        : OffsetDimensional(b), data(b.data), off{b.off}, indexer{b.indexer}, shape{b.shape} {
    }

    TensorBase(TensorBase<T> &&b) noexcept
        : OffsetDimensional(std::move(b)), data(std::move(b.data)), off{b.off}, indexer{std::move(b.indexer)},
          shape{std::move(b.shape)} {
        b.off = 0;
    }

    /**
     * Create tensor base by providing tensor base, a new shape, and an indexer
     * @param b base
     * @param shp shape
     * @param idx indxer
     */
    TensorBase(const TensorBase<T> &b, Shape shp, Indexer idx) : TensorBase(b) {
        shape = std::move(shp);
        this->indexer.emplace([idx = std::move(idx), idxOld = std::move(this->indexer)](const Index &i) {
            if (idxOld.has_value()) {
                return idxOld.value()(idx(i));
            } else {
                return idx(i);
            }
        });
    }

    /**
     * Get shape of tensor base
     * @return shape
     */
    const Shape &size() const override {
        return shape;
    }

    /**
     * Get total size of tensor base
     * @return total size
     */
    SizeType getTotalSize() const {
        SizeType n = 1;
        for (auto x : shape) {
            n *= x;
        }
        return n;
    }

    /**
     * Get the data
     * @return data
     */
    std::shared_ptr<TensorData<T>> getData() {
        return this->data;
    }

    // ============== INDEX ==============

    /**
     * Index the tensor base into
     * @param idx index
     * @return new tensor base
     */
    TensorBase<T> operator[](const Index &idx) override {
        SizeType loc = indexOn(idx);
        Shape s(this->idxToDim.size() - idx.size());
        for (SizeType i = 0, n = s.size(), m = idx.size(); i < n; i++) {
            s[i] = this->idxToDim[i + m];
        }
        return TensorBase<T>(this->data, this->dim, std::move(s), loc);
    }

    /**
     * Get the value in the tensor base
     * @param idx index
     * @return value
     */
    T &get(const Index &idx) noexcept(!OUT_OF_RANGE_CHECK) override {
        SizeType loc = indexOn(idx);
        return (*this->data)[loc];
    }

    /**
     * Get the value in the tensor base
     * @param idx index
     * @return value
     */
    const T &get(const Index &idx) const noexcept(!OUT_OF_RANGE_CHECK) override {
        SizeType loc = indexOn(idx);
        return (*this->data)[loc];
    }

    /**
     * Check if this tensor base is already contiguous or not
     * @return contiguous or not
     */
    bool isContiguous() {
        if (this->indexer.has_value()) {
            return false;
        }
        SizeType dimSize = this->dim.size();
        SizeType idxSize = this->idxToDim.size();
        if (idxSize > dimSize) {
            return false;
        }
        for (SizeType i = 0; i < idxSize; i++) {
            if (dimSize - idxSize + i != this->idxToDim[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Contiguous the tensor base
     * @return itself
     */
    TensorBase &contiguous() {
        if (isContiguous()) {
            return *this;
        }
        const auto &shp = shape;

        SizeType len = this->getTotalSize();
        SizeType on = 0;

        auto dat = std::make_shared<TensorData<T>>(len);

        for (auto x : (*this)) {
            (*dat)[on++] = x;
        }

        OffsetDimensional::operator=(OffsetDimensional(shp, Index::generateIncreasingSequence(shp.size())));

        this->data = dat; // Data
        this->off = 0;
        this->indexer = std::nullopt;

        return *this;
    }

    /**
     * Get the base with a specific offset
     * @param offby offset
     * @return raw pointer off by offset
     */
    T *offsetBy(SizeType offby = 0) {
        this->contiguous();
        return &((*(this->data))[off + offby]);
    }

    /**
     * Get the base with a specific index
     * @param idx index
     * @return raw pointer off by index
     */
    T *offsetBy(const Index &idx) {
        this->contiguous();

        if constexpr (OUT_OF_RANGE_CHECK) {
            if (idx.size() > this->idxToDim.size()) {
                throw std::out_of_range(std::string("Not able to offset in ") +
                                        tensorlib::base::shapeToString(this->size()));
            }
        }

        SizeType overallOff = off;

        for (SizeType i = 0; i < idx.size(); i++) {
            overallOff += offset[idxToDim[i]] * idx[i];
        }

        return &(*(this->data))[overallOff];
    }

    /**
     * Get raw pointer
     * @return raw pointer
     */
    T *raw() {
        return this->offsetBy();
    }

    // ============== BIG 5 ==============

    TensorBase &operator=(const TensorBase &base) {
        if (this == &base)
            return *this;
        OffsetDimensional::operator=(base);
        this->data = base.data;
        this->off = base.off;
        this->indexer = base.indexer;
        this->shape = base.shape;
        return *this;
    }

    TensorBase &operator=(TensorBase &&base) noexcept {
        if (this == &base)
            return *this;
        std::swap(data, base.data);
        std::swap(off, base.off);
        std::swap(indexer, base.indexer);
        std::swap(shape, base.shape);
        OffsetDimensional::operator=(std::move(base));
        return *this;
    }

    ~TensorBase() override = default;

    // ============== DIM OPERATION ==============

    /**
     * Swap two dimension d1, d2 of base
     * @param d1 d1
     * @param d2 d2
     */
    void swapDimension(SizeType d1, SizeType d2) {
        if constexpr (OUT_OF_RANGE_CHECK) {
            if (d1 >= this->dimensionSize() || d2 >= this->dimensionSize()) {
                throw std::out_of_range((std::stringstream() << "not able to swap dimension " << d1 << " " << d2
                                                             << " in " << tensorlib::base::shapeToString(this->size()))
                                            .str());
            }
        }
        this->contiguous();
        std::swap(this->idxToDim[d1], this->idxToDim[d2]);
        std::swap(this->shape[d1], this->shape[d2]);
    }

    /**
     * Get the number of dimensions of this base
     * @return number of dimensions
     */
    SizeType dimensionSize() const noexcept override {
        return shape.size();
    }

    /**
     * Reshape by shp
     * @param shp shp
     * @return reshaped base
     */
    TensorBase<T> reshape(const Shape &shp) {
        this->contiguous();
        return TensorBase<T>(this->data, shp, tensorlib::base::Index::generateIncreasingSequence(shp.size()));
    }

    /**
     * Permute the base
     * @param to permutation of dimension
     * @return permutation
     */
    TensorBase<T> permute(const Index &to) {
        this->contiguous();
        SizeType n = to.size();
        Index idx(n);
        for (SizeType i = 0; i < n; i++) {
            idx[i] = this->idxToDim[to[i]];
        }
        return TensorBase<T>(this->data, OffsetDimensional::dim, idx, off);
    }

    // ============== ITERATOR ==============

    TensorBaseIterator begin() {
        if (this->isContiguous()) {
            return TensorBaseIterator(&((*this->data)[off]));
        }
        return TensorBaseIterator(this, Index(this->shape.size(), 0));
    }

    TensorBaseIterator end() {
        if (this->isContiguous()) {
            SizeType tot = 1;
            for (auto x : this->shape) {
                tot *= x;
            }
            return TensorBaseIterator(&((*this->data)[0]) + tot);
        }
        return TensorBaseIterator(this, Index(this->shape.size(), 0), true);
    }

    // ============== STATIC FUNCTIONS ===============
public:

    /**
     * Create the base by providing shape
     * @tparam Args shape types
     * @param args shape
     * @return base
     */
    template <typename... Args>
        requires(std::is_constructible_v<SizeType, Args> && ...) && (sizeof...(Args) > 0)
    static TensorBase<T> ofShape(Args &&...args) {
        return TensorBase<T>(std::make_shared<TensorData<T>>((args * ...), ComputationConstant<T>::Zero),
                             Shape{args...}, Index::generateIncreasingSequence<sizeof...(Args)>());
    }

    /**
     * Create the single item base
     * @return single item base
     */
    static TensorBase<T> ofShape() {
        return TensorBase<T>(std::make_shared<TensorData<T>>(1, ComputationConstant<T>::Zero), Shape(0), Index(0));
    }

    /**
     * Create the base by providing shape
     * @param shp shape
     * @param t initialize value
     * @return base
     */
    static TensorBase<T> ofShape(const Shape &shp, const T &t = ComputationConstant<T>::Zero) {
        SizeType size = 1;
        for (auto x : shp) {
            size *= x;
        }
        return TensorBase<T>(std::make_shared<TensorData<T>>(size, t), shp,
                             Shape::generateIncreasingSequence(shp.size()));
    }

    /**
     * Check if a base is indexered or not
     * @param base
     * @return indexered or not
     */
    static bool isIndexered(const TensorBase<T> &base) {
        return base.indexer.has_value();
    }
};

}; // namespace cpu

}; // namespace tensor
}; // namespace tensorlib

#include "tensor_cpu_computation.hpp"

#endif