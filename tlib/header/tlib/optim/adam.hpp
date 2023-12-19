#ifndef __TLIB__OPTIMIZER_ADAM_HPP__
#define __TLIB__OPTIMIZER_ADAM_HPP__

#include "tlib/tlib_tensor.hpp"
#include "optimizer.hpp"

namespace tensorlib {
namespace optim {

/**
 * Adam optimizer
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class Adam : public Optimizer<T, Device> {
private:
    std::vector<tensor::BaseTensor<T, Device>> momentum;
    std::vector<tensor::BaseTensor<T, Device>> variance;

    T beta1;
    T beta2;
    T epsilon;

public:
    /**
     * @param parameters parameters
     * @param lr learning rate
     * @param betas {momentum for gradient, momentum for norm}
     * @param epsilon epsilon preventing overflow
     */
    Adam(const typename Optimizer<T, Device>::Parameters &parameters, T lr, std::pair<T, T> betas = {0.9f, 0.999f},
         T epsilon = 1e-9)
        : Optimizer<T, Device>(parameters, std::move(lr)), momentum(Optimizer<T, Device>::baseLike(this->parameters)),
          variance(Optimizer<T, Device>::baseLike(this->parameters)), beta1(std::move(betas.first)),
          beta2(std::move(betas.second)), epsilon(std::move(epsilon)) {
    }

    void step() override {
        SizeType n = this->parameters.size();
        for (SizeType i = 0; i < n; i++) {
            Tensor<T, Device> &t = *(this->parameters[i]);
            if (t.isRequireGrad()) {
                auto &m = momentum[i];
                auto &v = variance[i];

                // m <- beta1 * m + (1-beta1) * (nabla t)
                tensor::Computation<T, Device>::mulEqual(m, this->beta1);
                tensor::Computation<T, Device>::addEqual(
                    m, tensor::Computation<T, Device>::mul(t.getGradientBase(),
                                                           tensor::ComputationConstant<T>::One - this->beta1));

                // v <- beta2 * v + (1-beta2) * (nabla t)^2
                tensor::Computation<T, Device>::mulEqual(v, this->beta2);
                tensor::Computation<T, Device>::addEqual(
                    v, tensor::Computation<T, Device>::mul(
                           tensor::ComputationConstant<T>::One - this->beta2,
                           tensor::Computation<T, Device>::apply(t.getGradientBase(), [](auto &&x) { return x * x; })));

                // t <- t - lr * (m / sqrt(v + epsilon)) * (nabla g)
                tensor::Computation<T, Device>::minusEqual(
                    t.getBase(),
                    tensor::Computation<T, Device>::mul(
                        this->lr, tensor::Computation<T, Device>::div(
                                      m, tensor::Computation<T, Device>::apply(
                                             v, [eps = epsilon](auto &&x) { return std::sqrt(x + eps); }))));
            }
        }
    }
};

}; // namespace optim
}; // namespace tensorlib

#endif
