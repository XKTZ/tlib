#include <bits/stdc++.h>

using namespace std;

#include "tlib/tlib.hpp"

using namespace tensorlib::tensor::device;
using namespace tensorlib::nn;

int main() {
    using namespace tensorlib::tensor::device;
    using Tensor = tensorlib::tensor::Tensor<float, CPU>;
    using namespace tensorlib::nn;
    using namespace tensorlib::data;
    using namespace tensorlib::optim;

    Tensor ca(1, 1, 3, 3);
    for (int i = 0; i <9; i ++) {
        ca.raw()[i] = i;
    }

    Tensor cv(2, 1, 2, 2);
    for (int i = 0; i < 8; i ++) {
        cv.raw()[i] = float(i) / 2;
    }

    std::cout << ca << '\n';
    std::cout << cv << '\n';

    std::cout << tensorlib::functional::conv2d(ca, cv) << '\n';

    return 0;

    Tensor a = Tensor(5);

    Tensor b = Tensor(5);

    Tensor c = Tensor(5);

    for (int i = 0; i < 5; i++) {
        a.get({i}) = i;
        b.get({i}) = float(i) / 3.f;
        c.get({i}) = -float(i + 1) / 2.5f;
    }

    std::cout << "a = " << a << '\n' << '\n';
    std::cout << "b = " << b << '\n' << '\n';
    std::cout << "c = " << c << '\n' << '\n';

    auto d = a + b;
    std::cout << "a + b =" << d << '\n' << '\n';
    std::cout << "(a + b) / c = " << d / c << '\n' << '\n';
    std::cout << "log(a + b + 1.f) / exp(c) = "
              << tensorlib::functional::log(a + b + 1.f) / tensorlib::functional::exp(c) << '\n'
              << '\n';

    std::cout << "sum(a) * c - sum(b) = " << tensorlib::functional::sum(a) * c - tensorlib::functional::sum(b) << '\n';
}