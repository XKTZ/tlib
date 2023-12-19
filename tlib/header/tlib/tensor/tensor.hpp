#ifndef __TLIB__TENSOR_HPP__
#define __TLIB__TENSOR_HPP__

#include "tlib/tlib_base.hpp"
#include "tlib/tlib_dimensional.hpp"
#include "tlib/tlib_util.hpp"
#include "tensor_computational.hpp"
#include "tensor_grad.hpp"

#include <memory>
#include <type_traits>
#include <vector>

namespace tensorlib {

namespace tensor {

namespace _ {

/**
 * Type that can call raw pointer
 * @tparam Type type
 */
template <typename Type>
concept RawPointer = requires(Type t) {
    { t.raw() };
};
}; // namespace _

/**
 * Gradienter
 */
template <typename T, typename Device>
using Gradienter = grad::Gradienter<T, Device>;

/**
 * The basic tensor class. This is the class all differentiable computations are perform on
 */
template <typename T, typename Device>
    requires computational::IsValidComputationalInBase<T, Device>
class Tensor final : public tensorlib::base::Sized,
                     public tensorlib::base::Indexable<Tensor<T, Device>>,
                     public tensorlib::base::Gettable<T> {
    using Shape = tensorlib::base::Shape;
    using Index = tensorlib::base::Index;
    using Sized = tensorlib::base::Sized;
    using Base = BaseTensor<T, Device>;
    using Gradient = Gradienter<T, Device>;
    using Self = Tensor<T, Device>;

private:
    std::shared_ptr<Base> base;
    std::shared_ptr<Gradient> grad;

public:
    /**
     * Create tensor by providing base and gradienter (orig = null)
     * @param base base
     * @param grad gradient
     */
    Tensor(const Base &base, std::shared_ptr<Gradient> grad = nullptr)
        : base{std::make_shared<Base>(base)}, grad{std::move(grad)} {
    }

    /**
     * Create tensor by providing base and gradienter (orig = null)
     * @param base base
     * @param grad gradient
     */
    Tensor(Base &&base, std::shared_ptr<Gradient> grad = nullptr)
        : base{std::make_shared<Base>(std::move(base))}, grad{std::move(grad)} {
    }

    /**
     * Create tensor by providing pointer to base and gradienter (orig = null)
     * @param base base
     * @param grad gradient
     */
    Tensor(std::shared_ptr<Base> base, std::shared_ptr<Gradient> grad = nullptr)
        : base{std::move(base)}, grad{std::move(grad)} {
    }

    /**
     * Create tensor by providing shape and original value (=0)
     * @param shp shape
     * @param t value
     */
    Tensor(const Shape &shp, const T &t = ComputationConstant<T>::Zero)
        : base{std::make_shared<Base>(Base::ofShape(shp, t))}, grad{nullptr} {
    }

    /**
     * Create empty single item tensor
     */
    Tensor() : base{std::make_shared<Base>(Base::ofShape(Shape(0)))}, grad(nullptr) {
    }

    /**
     * Create tensor by providing bashape
     */
    template <typename... Args>
        requires(sizeof...(Args) > 0) && (std::is_constructible_v<SizeType, Args> && ...)
    Tensor(Args &&...args) : base{std::make_shared<Base>(Base::ofShape(Shape{args...}))}, grad(nullptr) {
    }

    /**
     * Copy constructor
     * @param t tensor
     */
    Tensor(const Self &t) : base{t.base}, grad(t.grad) {
    }

    /**
     * Move constructor
     * @param t tensor
     */
    Tensor(Self &&t) : base{std::move(t.base)}, grad(std::move(t.grad)) {
    }

    /**
     * Get the size of tensor
     * @return size
     */
    const Shape &size() const noexcept(std::is_nothrow_copy_constructible<Shape>::value) override {
        return base->size();
    }

    /**
     * Get the dimension size of tensor
     * @return dimension size
     */
    SizeType dimensionSize() const {
        return base->dimensionSize();
    }

    // ====================== INDEX ======================

    /**
     * [] operator, return a tensor that goes into by idx
     * @param idx idx
     * @return tensor by []
     */
    Tensor<T, Device> operator[](const Index &idx) override {
        return Tensor<T, Device>(std::make_shared<Base>((*base)[idx]), nullptr);
    }

    /**
     * Get item at specific location of idx
     * @param idx idx
     * @return item
     */
    T &get(const Index &idx) noexcept(!OUT_OF_RANGE_CHECK) override {
        return this->base->get(idx);
    }

    /**
     * Get item at specific location of idx
     * @param idx idx
     * @return item
     */
    const T &get(const Index &idx) const noexcept(!OUT_OF_RANGE_CHECK) override {
        return this->base->get(idx);
    }

    /**
     * Get single item of tensor
     */
    T &item() noexcept(!OUT_OF_RANGE_CHECK) {
        return this->get(std::initializer_list<SizeType>{});
    }

    /**
     * Get single item of tensor
     */
    const T &item() const noexcept(!OUT_OF_RANGE_CHECK) {
        return this->get(std::initializer_list<SizeType>{});
    }

    // ====================== BASE ======================

    /**
     * Acquire an shared pointer to base
     * @return pointer to base
     */
    std::shared_ptr<Base> acquireBase() const {
        return this->base;
    }

    /**
     * Get the reference of base
     * @return reference of base
     */
    Base &getBase() const {
        return *(this->base);
    }

    /**
     * Get the raw T* from base
     * @return T * raw
     */
    template <int ignored = 0>
        requires _::RawPointer<Base>
    auto raw() const {
        return this->base->raw();
    }

    /**
     * Return if the tensor is saved in contiguous memory or not
     */
    bool isContiguous() {
        return this->base->isContiguous();
    }

    /**
     * Let the base be contiguous
     */
    void contiguous() {
        this->base->contiguous();
    }

    // ====================== GRADIENT ======================

    /**
     * Get gradienter
     * @return gradienter
     */
    std::shared_ptr<Gradient> getGradienter() const {
        return this->grad;
    }

    /**
     * Get the base in gradient
     * @return base in gradient
     */
    Base &getGradientBase() const {
        return grad->getGradient();
    }

    /**
     * Get the gradient tensor
     * @return gradient tensor
     */
    Tensor<T, Device> getGradient() const {
        return Tensor<T, Device>(std::make_shared<Base>(grad->getGradient()));
    }

    /**
     * Requires gradient or not
     */
    bool isRequireGrad() const {
        return this->grad != nullptr;
    }

    /**
     * Let the tensor requires gradient
     */
    void requireGrad() {
        if (this->isRequireGrad())
            return;
        this->grad =
            std::make_shared<Gradient>(this->base->size(), [](auto &) {}, std::vector<std::shared_ptr<Gradient>>{});
    }

    /**
     * Stop the gradient
     */
    void noGrad() {
        this->grad = nullptr;
    }

    /**
     * Zero out the gradient
     */
    void zeroGrad() {
        this->grad->clear();
    }

    /**
     * Perform back propagation
     */
    void backward() {
        if (!this->isRequireGrad()) {
            throw std::runtime_error("Not backwardable, no gradient");
        }
        tensorlib::tensor::grad::performBackProp(this->grad, Base::ofShape(this->size(), ComputationConstant<T>::One));
    }

    Tensor &operator=(Tensor &&t) noexcept(std::is_nothrow_swappable_v<std::shared_ptr<Base>> &&
                                           std::is_nothrow_swappable_v<std::shared_ptr<Gradient>> &&
                                           noexcept(Sized::operator=(std::move(t)))) {
        if (this == &t) {
            return *this;
        }
        std::swap(this->base, t.base);
        std::swap(this->grad, t.grad);

        return *this;
    }

    Tensor &operator=(const Tensor &t) noexcept(std::is_nothrow_copy_assignable_v<std::shared_ptr<Base>> &&
                                                std::is_nothrow_copy_assignable_v<std::shared_ptr<Gradient>>) {
        if (this == &t) {
            return *this;
        }
        this->base = t.base;
        this->grad = t.grad;

        return *this;
    }

    ~Tensor() = default;
};

}; // namespace tensor
}; // namespace tensorlib

namespace tensorlib {
namespace printer {

/**
 * Default tensor printer
 */
struct DefaultTensorPrinter {
private:
    /**
     * Print recursively into ostream
     * @tparam T Type
     * @tparam Device Device
     * @param out ostream
     * @param t tensor
     * @param shp shp
     * @param on dimension on
     * @param idx idx
     */
    template <typename T, typename Device>
    static void print_recursive(std::ostream &out, const tensorlib::tensor::Tensor<T, Device> &t,
                                const tensorlib::base::Shape &shp, SizeType on, tensorlib::base::Index &idx) {
        if (on == shp.size()) {
            out << t.get(idx);
        } else {
            bool notLastIndex = on != shp.size() - 1;
            out << '[';
            for (SizeType i = 0, n = shp[on]; i < n; i++) {
                if (i > 0) {
                    if (notLastIndex) {
                        out << ",\n";
                    } else {
                        out << ",";
                    }
                }
                idx[on] = i;
                print_recursive(out, t, shp, on + 1, idx);
            }
            out << ']';
        }
    }

public:
    /**
     * Print a tensor into ostream
     * @tparam T T
     * @tparam Device Device
     * @param out ostream
     * @param t tensor
     */
    template <typename T, typename Device>
    static void print(std::ostream &out, const tensorlib::tensor::Tensor<T, Device> &t) {
        tensorlib::base::Shape shp = t.size();
        tensorlib::base::Index idx(shp.size());
        out << "tensor{";
        print_recursive(out, t, shp, 0, idx);
        out << ", size=(";
        bool printComma = false;
        for (auto sz : shp) {
            if (printComma) {
                out << ",";
            }
            out << sz;
            printComma = true;
        }
        out << ")";
        auto g = t.getGradienter();
        if (g != nullptr) {
            out << ", grad=" << (&(*g));
        }
        out << "}";
    }
};

using TensorPrinter = DefaultTensorPrinter;
}; // namespace printer
}; // namespace tensorlib

template <typename T, typename Device>
std::ostream &operator<<(std::ostream &out, const tensorlib::tensor::Tensor<T, Device> &t) {
    tensorlib::printer::TensorPrinter::print(out, t);
    return out;
}

#include "tensor_operations.hpp"

#endif