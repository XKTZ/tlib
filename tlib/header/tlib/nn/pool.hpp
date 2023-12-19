#ifndef __TLIB__NN_POOL_HPP__
#define __TLIB__NN_POOL_HPP__

#include "tlib/tlib_tensor.hpp"
#include "module.hpp"
#include <utility>

namespace tensorlib {
namespace nn {

namespace functional {

/**
 * Pooling
 * @tparam T Type
 * @tparam Device Device
 * @tparam Dim Pooling dimension size
 * @param x tensor
 * @param dim dimension to pooling, in form of [{poolDimension, scaleSize}] sorted by poolDimension
 * @param f pooling function
 * @return pooled tensor
 */
template <typename T, typename Device, SizeType Dim>
Tensor<T, Device> pooling(const Tensor<T, Device> &x, const std::array<std::pair<SizeType, SizeType>, Dim> &dim,
                          auto &&f) {
    Shape shp = x.size();
    SizeType n = shp.size();
    Shape newShape(n + Dim);
    Index newPermute(n + Dim);
    SizeType idx = 0, dimIdx = 0;
    for (SizeType i = 0; i < n; i++) {
        if (dimIdx < Dim && dim[dimIdx].first == i) {
            newShape[idx] = shp[i] / dim[dimIdx].second;
            newShape[idx + 1] = dim[dimIdx].second;
            newPermute[i] = idx;
            newPermute[n + dimIdx] = idx + 1;
            idx += 2;
            dimIdx++;
        } else {
            newShape[idx] = shp[i];
            newPermute[i] = idx;
            idx++;
        }
    }
    Tensor<T, Device> nxt = tensorlib::functional::permute(tensorlib::functional::reshape(x, newShape), newPermute);
    return std::forward<decltype(f)>(f)(nxt);
}

/**
 * Average pooling
 * @tparam T Type
 * @tparam Device Device
 * @tparam Dim Dimension
 * @param x tensor
 * @param dim dimension
 * @return avgpool(x) on dim
 */
template <typename T, typename Device, SizeType Dim>
Tensor<T, Device> avgPooling(const Tensor<T, Device> &x, const std::array<std::pair<SizeType, SizeType>, Dim> &dim) {
    SizeType total = 1;
    for (auto &[_, d] : dim) {
        total *= d;
    }
    return pooling(x, dim, [total, n = x.dimensionSize()](auto &&y) {
        return tensorlib::functional::sum(y, Index::generateSequenceInRange(n, n + Dim)) / T(total);
    });
}
}; // namespace functional

/**
 * 2D average pool. [..., H, W] -> [..., H/sH, W/sW]
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class AvgPool2d : public Module<T, Device, Tensor<T, Device>, Tensor<T, Device>> {
    std::pair<SizeType, SizeType> kernel;
    SizeType totalSize;

public:
    explicit AvgPool2d(std::pair<SizeType, SizeType> kernel)
        : Module<T, Device, Tensor<T, Device>, Tensor<T, Device>>(), kernel(std::move(kernel)),
          totalSize(this->kernel.first * this->kernel.second) {
    }

    std::string_view name() const override {
        return "Average Pooling 2D";
    }

private:
    Tensor<T, Device> forward(const Tensor<T, Device> &t) override {
        return functional::avgPooling(t, std::array<std::pair<SizeType, SizeType>, 2>{
                                             std::pair<SizeType, SizeType>{t.dimensionSize() - 2, kernel.first},
                                             std::pair<SizeType, SizeType>{t.dimensionSize() - 1, kernel.second}});
    }
};

}; // namespace nn
}; // namespace tensorlib

#endif