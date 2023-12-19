#ifndef __TLIB__OPTIMIZER_HPP__
#define __TLIB__OPTIMIZER_HPP__

#include "tlib/tlib_nn.hpp"
#include "tlib/tlib_tensor.hpp"
#include <cmath>
#include <utility>

namespace tensorlib {
namespace optim {

/**
 * Optimizer basic class
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class Optimizer {
protected:
    using Parameters = typename nn::ParameterGroup<T, Device>;

private:

    /**
     * Recursively add parameters into vector
     * @param p parameters
     * @param v vector
     */
    static void recursivelyAdd(const Parameters &p, std::vector<Tensor<T, Device> *> &v) {
        for (auto x : p.params) {
            v.push_back(x);
        }

        for (auto &lay : p.layers) {
            recursivelyAdd(*lay, v);
        }
    }

protected:

    /**
     * helper function generate a BaseTensor like all tensor pointers in vector
     * @param v vector
     * @param initial initial value
     * @return generated base
     */
    static std::vector<tensor::BaseTensor<T, Device>> baseLike(const std::vector<Tensor<T, Device> *> &v,
                                                               T initial = 1e-6) {
        std::vector<tensor::BaseTensor<T, Device>> result;
        for (auto &x : v) {
            result.push_back(tensor::BaseTensor<T, Device>::ofShape(x->size(), initial));
        }
        return result;
    }

    /**
     * get vector of parameters from parameter group
     * @param params parameter group
     * @return vector
     */
    static std::vector<Tensor<T, Device> *> from(const Parameters &params) {
        std::vector<Tensor<T, Device> *> result;
        recursivelyAdd(params, result);
        return result;
    }

    T lr;
    std::vector<Tensor<T, Device> *> parameters;

public:

    /**
     * @param parameters parameters
     * @param lr learning rate
     */
    Optimizer(const Parameters &parameters, T lr) : lr(lr), parameters(Optimizer<T, Device>::from(parameters)) {
    }

    /**
     * @return learning rate
     */
    T getLearningRate() noexcept(std::is_nothrow_copy_constructible_v<T>) {
        return lr;
    }

    /**
     * Step is used for step optimizer by one step by providing gradient in parameters
     */
    virtual void step() = 0;

    /**
     * clear gradient
     */
    void zeroGrad() {
        for (auto &t : this->parameters) {
            t->zeroGrad();
        }
    }

    virtual ~Optimizer() = default;
};
}; // namespace optim
}; // namespace tensorlib

#endif