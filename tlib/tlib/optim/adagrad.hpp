#ifndef __TLIB__OPTIMIZER_ADAGRAD_HPP__
#define __TLIB__OPTIMIZER_ADAGRAD_HPP__

#include "tlib/optim/optimizer.hpp"
#include "tlib/tlib_tensor.hpp"
#include <cmath>

namespace tensorlib {
namespace optim {

/**
 * Adagrad optimizer
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class Adagrad final : public Optimizer<T, Device> {
private:
    std::vector<tensor::BaseTensor<T, Device>> pastNorm;

public:
    /**
     * @param parameters parameter
     * @param lr learning rate
     * @param epsilon epsilon preventing overflow
     */
    Adagrad(const typename Optimizer<T, Device>::Parameters &parameters, T lr, T epsilon = 1e-9f)
        : Optimizer<T, Device>(parameters, std::move(lr)),
          pastNorm(Optimizer<T, Device>::baseLike(this->parameters, epsilon)) {
    }

    void step() override {
        SizeType n = this->parameters.size();
        for (SizeType i = 0; i < n; i++) {
            auto &t = *(this->parameters[i]);

            if (t.isRequireGrad()) {
                auto &pn = pastNorm[i];
                tensor::Computation<T, Device>::addEqual(pn,
                                                         tensor::Computation<T, Device>::pow(t.getGradientBase(), 2));
                tensor::Computation<T, Device>::minusEqual(
                    t.getBase(), tensor::Computation<T, Device>::div(
                                     tensor::Computation<T, Device>::mul(t.getGradientBase(), this->lr),
                                     tensor::Computation<T, Device>::apply(pn, [](auto &&x) { return sqrt(x); })));
            }
        }
    }
};

}; // namespace optim
}; // namespace tensorlib

#endif