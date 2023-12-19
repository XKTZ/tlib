#ifndef __TLIB__OPTIMIZER_RMSPROP_HPP__
#define __TLIB__OPTIMIZER_RMSPROP_HPP__

#include "tlib/tlib_tensor.hpp"
#include "optimizer.hpp"

namespace tensorlib {
namespace optim {

/**
 * RMSProp optimizer
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class RMSProp : public Optimizer<T, Device> {
    std::vector<tensor::BaseTensor<T, Device>> norm;
    T alpha;
    T epsilon;

public:
    /**
     * @param parameters parameters
     * @param lr learning rate
     * @param alpha momentum on norm
     * @param epsilon epsilon for preventing overflow
     */
    RMSProp(const typename Optimizer<T, Device>::Parameters &parameters, T lr, T alpha = 0.99, T epsilon = 1e-9f)
        : Optimizer<T, Device>(parameters, std::move(lr)), norm(Optimizer<T, Device>::baseLike(this->parameters)),
          alpha(std::move(alpha)), epsilon(std::move(epsilon)) {
    }

    void step() override {
        SizeType n = norm.size();
        for (SizeType i = 0; i < n; i++) {
            Tensor<T, Device> &t = *(this->parameters[i]);
            if (t.isRequireGrad()) {
                auto &m = norm[i];

                // m *= alpha
                tensor::Computation<T, Device>::mulEqual(m, alpha);

                // m += (nabla t)^2 * (1 - alpha)
                tensor::Computation<T, Device>::addEqual(
                    m, tensor::Computation<T, Device>::mul(tensor::Computation<T, Device>::pow(t.getGradientBase(), 2),
                                                           (tensor::ComputationConstant<T>::One - alpha)));

                // t -= lr * ((nabla t) / sqrt(m + epsilon))
                tensor::Computation<T, Device>::minusEqual(
                    t.getBase(), tensor::Computation<T, Device>::mul(
                                     this->lr, tensor::Computation<T, Device>::div(
                                                   t.getGradientBase(),
                                                   tensor::Computation<T, Device>::apply(
                                                       m, [eps = this->epsilon](auto &&x) { return sqrt(x + eps); }))));
            }
        }
    }
};

}; // namespace optim
}; // namespace tensorlib

#endif
