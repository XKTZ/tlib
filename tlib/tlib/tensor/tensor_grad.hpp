#ifndef __TLIB__TENSOR_GRAD__
#define __TLIB__TENSOR_GRAD__

#include "tlib/tensor/tensor_computational.hpp"
#include "tlib/tlib_base.hpp"
#include "tlib/tlib_dimensional.hpp"
#include "tlib/tlib_util.hpp"

#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>

namespace tensorlib {
namespace tensor {
namespace grad {

/**
 * Gradienter class, used for calculating radient
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class Gradienter final {
    using Shape = tensorlib::base::Shape;
    using Compute = Computation<T, Device>;
    using Constant = ComputationConstant<T>;
    using Base = BaseTensor<T, Device>;

private:
    std::function<void(Base &)> grad;
    std::vector<std::shared_ptr<Gradienter<T, Device>>> children;

    Base gradient;

public:
    Gradienter(const Gradienter<T, Device> &g) : grad{g.grad}, children{g.children}, gradient(g.gradient) {
    }

    Gradienter(Gradienter<T, Device> &&g)
        : grad{std::move(g.grad)}, children{std::move(g.children)}, gradient(std::move(g.gradient)) {
    }

    /**
     * @param gradientShape shape of gradient
     * @param grad gradient function
     * @param children children of the gradienter
     */
    Gradienter(Shape gradientShape, std::function<void(Base &)> grad,
               std::vector<std::shared_ptr<Gradienter<T, Device>>> children)
        : grad(std::move(grad)), children(std::move(children)),
          gradient(Compute::Base::ofShape(std::move(gradientShape))) {
    }

    Gradienter &operator=(Gradienter g) {
        grad = std::move(g.grad);
        children = std::move(g.children);
        gradient = std::move(g.gradient);
        return *this;
    }

    /**
     * Get the gradient
     * @return gradient
     */
    Base &getGradient() {
        return gradient;
    }

    /**
     * Get the gradient
     * @return gradient
     */
    const Base &getGradient() const {
        return gradient;
    }

    /**
     * Zero out gradient
     */
    void clear() {
        Compute::setEqual(gradient, Constant::Zero);
    }

    /**
     * Set the gradient to a specific value
     * @return gradient
     */
    template <typename S>
        requires std::is_same_v<std::decay_t<S>, Base>
    void setGradient(S &&g) {
        Compute::setEqual(gradient, g);
    }

    /**
     * perform back prop on this specific gradient
     */
    void backward() {
        grad(gradient);
    }

    /**
     * Get children
     * @return children of gradienter
     */
    std::vector<std::shared_ptr<Gradienter<T, Device>>> &getChildren() {
        return children;
    }

    /**
     * Get children
     * @return children of gradienter
     */
    const std::vector<std::shared_ptr<Gradienter<T, Device>>> &getChildren() const {
        return children;
    }

    ~Gradienter() = default;

    /**
     * Add gradient to a gradienter
     * @tparam S type of gradient
     * @param grad gradienter
     * @param g grad
     */
    template <typename S>
        requires std::is_same_v<std::decay_t<S>, Base>
    static void addGradient(Gradienter<T, Device> *grad, S &&g) {
        if (grad == nullptr)
            return;
        Compute::addEqual(grad->gradient, g);
    }

    /**
     * Add gradient to a gradienter pointer
     * @tparam S type of gradient
     * @param grad gradienter pointer
     * @param g grad
     */
    template <typename S>
        requires std::is_same_v<std::decay_t<S>, Base>
    static void addGradient(std::shared_ptr<Gradienter<T, Device>> &grad, S &&g) {
        addGradient(grad.get(), g);
    }
};

/**
 * Perform the back propagation to a gradienter using a initial value
 * @tparam T Type
 * @tparam Device Device
 * @tparam S initial value type
 * @param g gradient
 * @param initial initial value
 */
template <typename T, typename Device, typename S>
    requires std::is_same_v<std::decay_t<BaseTensor<T, Device>>, S>
static void performBackProp(const std::shared_ptr<Gradienter<T, Device>> &g, S &&initial) {
    using std::queue;
    using std::unordered_map;

    unordered_map<Gradienter<T, Device> *, SizeType> inner;
    queue<Gradienter<T, Device> *> q;

    // initialize the inner map
    q.push(g.get());
    inner[q.front()] = 0;

    while (!q.empty()) {
        Gradienter<T, Device> *u = q.front();
        q.pop();
        for (std::shared_ptr<Gradienter<T, Device>> &vp : u->getChildren()) {
            auto v = vp.get();
            if (v == nullptr)
                continue;
            if (!inner.count(v)) {
                q.push(v);
            }
            inner[v]++;
        }
    }

    // perform back prop
    q.push(g.get());
    g->setGradient(initial);

    while (!q.empty()) {
        Gradienter<T, Device> *u = q.front();
        q.pop();

        u->backward();

        for (std::shared_ptr<Gradienter<T, Device>> &vp : u->getChildren()) {
            auto v = vp.get();
            if (v == nullptr)
                continue;

            if ((--inner[v]) == 0) {
                q.push(v);
            }
        }
    }
}

}; // namespace grad
}; // namespace tensor
}; // namespace tensorlib

#endif