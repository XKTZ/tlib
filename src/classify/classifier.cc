#include "tlib/tlib.hpp"
#include <bits/stdc++.h>

using tensorlib::tensor::device::CPU;
using namespace tensorlib::vision;
using namespace tensorlib::data;
using namespace tensorlib::nn;

int cnt = 0;

template <typename T, typename Device>
class Discriminator : public Module<T, Device, tensorlib::Tensor<T, Device>, tensorlib::Tensor<T, Device>> {
    using Tensor = tensorlib::Tensor<T, Device>;

private:
    struct ConvBlock : public Module<T, Device, Tensor, Tensor> {

        Conv2d<T, Device> conv1;
        ReLU<T, Device> activ1;

        ConvBlock(int in, int out) : conv1(in, out, {3, 3}, {1, 1}), activ1() {
            this->registerAll(conv1, activ1);
        }

        Tensor forward(const Tensor &x) override {
            cnt ++;
            return activ1(conv1(x));
        }
    };

    ConvBlock conv1;
    AvgPool2d<T, Device> pool1;
    ConvBlock conv2;
    AvgPool2d<T, Device> pool2;
    ConvBlock conv3;
    AvgPool2d<T, Device> pool3;

    size_t reshapeSize;

    Linear<T, Device> lin1;
    ReLU<T, Device> activ;
    Linear<T, Device> lin2;

public:
    Discriminator(int dimIn, int totalSize, int out)
        : conv1(dimIn, dimIn * 4), pool1({2, 2}), conv2(dimIn * 4, dimIn * 16), pool2({2, 2}),
          conv3(dimIn * 16, dimIn * 16), pool3({2, 2}), reshapeSize(dimIn * 16 * totalSize / ((1 << 3) * (1 << 3))),
          lin1(reshapeSize, 64), activ(), lin2(64, out) {
        this->registerAll(conv1, pool1, conv2, pool2, conv3, pool3, lin1, activ, lin2);
    }

private:
    Tensor forward(const Tensor &x) override {
        auto c1 = pool1(conv1(x));
        auto c2 = pool2(conv2(c1));
        auto c3 = pool3(conv3(c2));
        auto result = tensorlib::functional::reshape(c3, {x.size()[0], reshapeSize});
        return lin2(activ(lin1(result)));
    }
};

auto loadDatas(auto &&in, const std::string &base, size_t maxTarget, const std::string &suffix = ".png") {
    using Tensor = tensorlib::Tensor<float, CPU>;
    using namespace std;
    vector<std::pair<Tensor, Tensor>> result;

    std::cout << "Loading dataset..." << '\n';

    int n;
    in >> n;
    string name;
    int target;

    for (int i = 1; i <= n; i++) {
        std::cout << "\r" << i << "/" << n << "                        " << flush;
        in >> name >> target;
        Tensor targ(maxTarget);
        targ.get({target}) = 1.;

        result.push_back(
            {PNGReader::loadImage<PNGPixelGray, float, CPU>(std::fstream(base + "/" + name + suffix)), targ});
    }

    cout << '\n';
    return result;
}

template <typename T, typename C>
void testData(auto &&classifier, Dataset<T, C> &dataset) {

    size_t batchSize = dataset.getBatchSize();

    size_t totcorr = 0;
    size_t tot = 0;
    size_t on = 0;

    for (auto &[x, y] : dataset.iter()) {
        auto pred = classifier((x - 0.5f) * 2.f);
        int corr = 0;
        for (int i = 0; i < batchSize; i++) {
            auto pi = pred[{i}];
            float mx = pi.get({0});
            int mxloc = 0;
            for (int j = 1; j < 10; j++) {
                if (mx < pi.get({j})) {
                    mx = pi.get({j});
                    mxloc = j;
                }
            }
            if (y.get({i, mxloc}) == 1) {
                corr++;
            }
        }

        tot += batchSize;
        totcorr += corr;

        ++on;
        if (on % 10 == 0) {
            std::cout << "\r" << on << ' ' << totcorr << '/' << tot << ' ' << double(totcorr) / double(tot)
                      << "                   " << std::flush;
        }
    }

    std::cout << '\n';

    std::cout << totcorr << '/' << tot << ' ' << double(totcorr) / tot << '\n';
}

template <typename T, typename C, typename O>
void trainData(auto &classifier, tensorlib::optim::Optimizer<float, CPU> &optim, Dataset<T, C> &dataset, int totalEpoch,
               std::optional<O> out = std::nullopt) {
    using namespace std;
    using namespace std::chrono;
    using namespace tensorlib::optim;

    CrossEntropyLoss<float, CPU> loss;

    for (int epoch = 1; epoch <= totalEpoch; epoch++) {
        size_t i = 0;

        size_t tot = 0;

        std::cout << epoch << '\n';

        auto now = high_resolution_clock ::now();

        double eta = -1.;

        for (auto &[x, y] : dataset.iter()) {
            ++i;

            auto result = classifier((x - 0.5f) * 2.f);

            auto pred = tensorlib::nn::functional::softmax(result, {1});

            auto ls = loss(pred, y);

            if (eta < 0) {
                eta = ls.item();
            } else {
                eta = eta * 0.95 + ls.item() * 0.05;
            }

            optim.zeroGrad();
            ls.backward();
            optim.step();

            if (i % 10 == 0) {
                tot = duration_cast<milliseconds>(high_resolution_clock ::now() - now).count();
                std::cout << "\r" << i << ' ' << eta << ' ' << tot << ' ' << double(tot) / i << ' '
                          << double(tot) / (i * dataset.getBatchSize()) << "               " << flush;
            }
        }

        cout << '\n';
    }

    if (out.has_value()) {
        tensorlib::save(classifier.parameters(), out.value());
    }
}

void classify(const std::string &model, const std::vector<std::string> &imgPath) {
    using Tensor = tensorlib::Tensor<float, CPU>;
    Discriminator<float, CPU> classifier(1, 32 * 32, 10);

    if (model == "default") {
        classifier.parameters().load(std::fstream{"./mnist_pretrained", std::ios::in | std::ios::binary});
    } else {
        classifier.parameters().load(std::fstream{model, std::ios::in | std::ios::binary});
    }
    classifier.setStatus(ModeStatus::EVAL);

    int n = imgPath.size();
    std::vector<Tensor> img;
    std::vector<tensorlib::tensor::BaseTensor<float, CPU> *> imgp;
    for (auto &p : imgPath) {
        img.push_back(PNGReader::loadImage<PNGPixelGray, float, CPU>(std::fstream{p}));
        imgp.push_back(&(img.back().getBase()));
    }
    Tensor all = Tensor(tensorlib::tensor::Computation<float, CPU>::concat(imgp));

    auto result = classifier((all - 0.5f) * 2.f);
    for (int i = 0; i < n; i++) {
        auto p = result[{i}];
        int loc = 0;
        std::cout << 0 << ':' << p.get({0}) << ' ';
        for (int j = 1; j < 10; j++) {
            if (p.get({j}) > p.get({loc})) {
                loc = j;
            }
            std::cout << j << ':' << p.get({j}) << ' ';
        }
        std::cout << '\n';
        std::cout << "Predicted most possible is " << loc << '\n';
    }
}

int main() {
    using Tensor = tensorlib::Tensor<float, CPU>;
    using namespace std;
    using namespace std::chrono;
    using namespace tensorlib::optim;

    tensorlib::nn::init::manualSeed(42);

    string inp;

    while (true) {
        std::getline(cin, inp);

        stringstream ss;
        ss << inp;

        string command;
        ss >> command;

        if (command.empty()) {
            continue;
        } else if (command == "classify") {
            string model;
            ss >> model;
            std::vector<string> imgs;
            while (!ss.eof()) {
                imgs.emplace_back();
                ss >> imgs.back();
            }
            classify(model, imgs);
        } else if (command == "train") {
            std::string savePath, base, dataset, optimizer;
            float lr;
            int epoch;
            int batchSize;
            ss >> savePath >> base >> dataset >> optimizer >> lr >> epoch >> batchSize;

            Discriminator<float, CPU> classifier(1, 32 * 32, 10);

            std::unique_ptr<Optimizer<float, CPU>> opt([&classifier, &optimizer, lr]() -> Optimizer<float, CPU> * {
                if (optimizer == "sgd") {
                    return new SGD<float, CPU>(classifier.parameters(), lr, 0.9);
                } else if (optimizer == "rmsprop") {
                    return new RMSProp<float, CPU>(classifier.parameters(), lr);
                } else if (optimizer == "adam") {
                    return new Adam<float, CPU>(classifier.parameters(), lr);
                } else {
                    optimizer = "Unrecognized, Defaultly set Adam";
                    return new Adam<float, CPU>(classifier.parameters(), lr);
                }
            }());

            std::cout << "Training model in following info:" << '\n';
            std::cout << "Dataset: "
                      << "(file:" << dataset << ", base:" << base << ")\n";
            std::cout << "Optimizer: " << optimizer << '\n';
            std::cout << "Learning Rate: " << lr << '\n';
            std::cout << "Epoch: " << epoch << '\n';
            std::cout << "Save at: " << savePath << '\n';

            PredictionDataset<float, CPU> dat(loadDatas(std::fstream{dataset}, base, 10), batchSize, true, false);

            trainData(classifier, *opt, dat, epoch,
                      std::make_optional(std::fstream{savePath, std::ios::trunc | std::ios::out | std::ios::binary}));

        } else if (command == "exit") {
            break;
        } else if (command == "test") {
            std::cout << "<Notice this is a development command, it doesn't ensure correctness>" << '\n';
            std::string model, base, dataset;
            ss >> model >> base >> dataset;

            Discriminator<float, CPU> classifier(1, 32 * 32, 10);

            classifier.setStatus(ModeStatus::EVAL);

            PredictionDataset<float, CPU> dat(loadDatas(std::fstream{dataset}, base, 10), 64, true, false);

            classifier.parameters().load(fstream(model, std::ios::in | std::ios::binary));

            testData(classifier, dat);
        } else {
            std::cout << "classify default <image>...:" << '\n' << "      where <image> is a 32x32 sized image" << '\n';
            std::cout << "classify <model-path> <image>...:" << '\n'
                      << "      using <model-path> to classify an image stored in <image>" << '\n';
            std::cout << "train <model-save-path> <dataset> <optimizer> <learning-rate> <epoch> <batchSize>:\n"
                      << "      use the data saved in "
                         "dataset to train"
                         "a model (in which has a predesigned structure) using optimizer opitimizer and learning rate "
                         "learning-rate for <epoch> epochs, using <batchSize> minibatch"
                      << '\n';
            std::cout << "help:\n"
                      << "      output the help info of digit classification terminal" << '\n';
            std::cout << "exit" << '\n' << "      exit\n";
        }
    }
}