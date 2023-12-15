#include "tlib/tlib.hpp"
#include <bits/stdc++.h>

/**
 * Notice this is not a general DCGAN. It is a WGAN so we need to clip the model data range
 * [1] Martin Arjovsky, Soumith Chintala, Léon Bottou: “Wasserstein GAN”, 2017; [http://arxiv.org/abs/1701.07875
 * arXiv:1701.07875].
 */

using tensorlib::tensor::device::CPU;
using namespace tensorlib::vision;
using namespace tensorlib::data;
using namespace tensorlib::nn;

constexpr int LATENT_DIM = 128;

/**
 * Discriminator of GAN
 */
template <typename T, typename Device>
class Discriminator : public Module<T, Device, tensorlib::Tensor<T, Device>, tensorlib::Tensor<T, Device>> {
    using Tensor = tensorlib::Tensor<T, Device>;

private:
public:
    struct ConvBlock : public Module<T, Device, Tensor, Tensor> {
        Conv2d<T, Device> conv1;
        ReLU<T, Device> activ1;
        AvgPool2d<T, Device> pool1;

        ConvBlock(int in, int out) : conv1(in, out, {3, 3}, {1, 1}), activ1(), pool1({2, 2}) {
            this->registerAll(conv1, activ1, pool1);
        }

        Tensor forward(const Tensor &x) override {
            return pool1(activ1(conv1(x)));
        }
    };

    size_t reshapeSize;

    ConvBlock conv1;
    ConvBlock conv2;
    ConvBlock conv3;

    Linear<T, Device> lin1;

public:
    Discriminator(int dimIn, int dim, int sz)
        : reshapeSize(((sz * sz) >> 6) * dim * 4), conv1(dimIn, dim), conv2(dim, dim * 2), conv3(dim * 2, dim * 4),
          lin1(reshapeSize, 1) {
        this->registerAll(conv1, conv2, conv3, lin1);
    }

private:
    Tensor forward(const Tensor &x) override {
        auto c = conv3(conv2(conv1(x)));
        auto result = tensorlib::functional::reshape(c, {x.size()[0], reshapeSize});
        return lin1(result);
    }
};

/**
 * Generator of GAN
 * @tparam T
 * @tparam Device
 */
template <typename T, typename Device>
class Generator : public Module<T, Device, tensorlib::Tensor<T, Device>, tensorlib::Tensor<T, Device>> {
    using Tensor = tensorlib::Tensor<T, Device>;

private:
    int channelIn;
    int channelOut;
    int sz;
    int startSz;

    Linear<T, Device> lin;

    ConvTranspose2d<T, Device> conv1;
    ConvTranspose2d<T, Device> conv2;
    ConvTranspose2d<T, Device> conv3;
    ConvTranspose2d<T, Device> conv4;

    Conv2d<T, Device> finalConv;

    auto tanh(auto &&x) {
        auto p = tensorlib::functional::exp(x), q = tensorlib::functional::exp(-x);
        return (p - q) / (p + q);
    }

public:
    Generator(int inputSize, int sz, int channelIn, int channelOut)
        : channelIn(channelIn), channelOut(channelOut), sz(sz), startSz(sz >> 4),
          lin(inputSize, startSz * startSz * channelIn), conv1({2, 2}, channelIn, channelIn, {3, 3}, {1, 1}),
          conv2({2, 2}, channelIn, channelIn * 2, {3, 3}, {1, 1}),
          conv3({2, 2}, channelIn * 2, channelIn * 4, {3, 3}, {1, 1}),
          conv4({2, 2}, channelIn * 4, channelIn * 8, {3, 3}, {1, 1}),
          finalConv(channelIn * 8, channelOut, {3, 3}, {1, 1}) {
        this->registerAll(lin, conv1, conv2, conv3, conv4, finalConv);
    }

    Tensor forward(const Tensor &x) override {
        auto w = lin(x);
        auto y = tensorlib::functional::reshape(w, {int(x.size()[0]), channelIn, startSz, startSz});
        auto img = finalConv(conv4(conv3(conv2(conv1(y)))));
        // tanh
        return tanh(img);
    }
};

/**
 * Load datas
 * @param in input for data file
 * @param base data base directory
 * @return data vector
 */
auto loadDatas(auto &&in, const std::string &base, const std::string &suffix = ".png") {
    using Tensor = tensorlib::Tensor<float, CPU>;
    using namespace std;
    vector<Tensor> result;

    std::cout << "Loading dataset..." << '\n';

    int n;
    in >> n;
    string name;
    int target;

    for (int i = 1; i <= n; i++) {
        std::cout << "\r" << i << "/" << n << "                        " << flush;
        in >> name;

        result.push_back(PNGReader::loadImage<PNGPixelGray, float, CPU>(std::fstream(base + "/" + name + suffix)));
    }

    cout << '\n';
    return result;
}

/**
 * Generate data
 * @param model model
 * @param name name
 * @param count count
 * @param seed seed
 */
void generateData(std::string model, std::string name, int count, int seed) {
    using Tensor = tensorlib::Tensor<float, CPU>;
    if (model == "default") {
        model = "./mnist_generator";
    }
    tensorlib::random::manualSeed(seed);

    Generator<float, CPU> generator(128, 16, 32, 1);
    generator.parameters().load(std::fstream(model, std::ios::in | std::ios::binary));
    std::cout << "generating..." << '\n';
    Tensor initial(count, LATENT_DIM);
    tensorlib::functional::applyOn(initial, [](auto &&x) { x = tensorlib::random::uniform(-1., 1.); });

    Tensor imag = (generator(initial) / 2.f) + 0.5f;

    for (int j = 0; j < count; j++) {
        tensorlib::vision::PNGWriter::writeImage<tensorlib::vision::PNGPixelGray>(
            std::fstream((std::stringstream() << name << "-" << j << ".png").str(), std::ios::out | std::ios::trunc),
            imag[{j}]);
    }
    std::cout << "Generation finished" << '\n';
}

/**
 * Train data
 * @param discriminator discriminator
 * @param optimD optimizer of discriminator
 * @param generator generator
 * @param optimG optimizer of generator
 * @param dataset dataset
 * @param totalEpoch total epoch
 * @param outD output directory of discriminator
 * @param outG output directory of generator
 * @param origSize original size
 * @param clip clip for WGAN
 * @param ncrit number of critic
 */
template <typename T, typename C>
void trainData(Discriminator<float, CPU> &discriminator, tensorlib::optim::Optimizer<float, CPU> &optimD,
               Generator<float, CPU> &generator, tensorlib::optim::Optimizer<float, CPU> &optimG,
               Dataset<T, C> &dataset, int totalEpoch, std::string outD = "", std::string outG = "",
               int origSize = LATENT_DIM, float clip = 0.01f, int ncrit = 1) {
    using Tensor = tensorlib::Tensor<float, CPU>;
    using namespace std;
    using namespace std::chrono;
    using namespace tensorlib::optim;

    static constexpr float ETA_DECAY = 0.95;

    for (int epoch = 1; epoch <= totalEpoch; epoch++) {
        size_t i = 0;

        size_t tot = 0;

        std::cout << epoch << '\n';

        auto now = high_resolution_clock ::now();

        Tensor lossd{}, lossg{};

        for (auto &x : dataset.iter()) {
            ++i;

            // Discriminator turn
            {
                Tensor predCorrect = discriminator((x - 0.5f) * 2.f) / float(x.size()[0]);

                Tensor wrongInitial = Tensor(predCorrect.size()[0], size_t(origSize));

                tensorlib::functional::applyOn(wrongInitial, [](auto &&x) { x = tensorlib::random::uniform(-1., 1.); });

                Tensor predWrong = discriminator(generator(wrongInitial)) / float(wrongInitial.size()[0]);

                lossd = tensorlib::functional::sum(predWrong) - tensorlib::functional::sum(predCorrect);

                optimD.zeroGrad();
                lossd.backward();
                optimD.step();

                discriminator.parameters().apply([clip](auto &t) {
                    tensorlib::functional::applyOn(t, [clip](auto &x) { x = std::max(-clip, std::min(clip, x)); });
                });
            }

            // Generator turn
            if (i % ncrit == 0) {

                Tensor wrongInitial = Tensor(dataset.getBatchSize(), size_t(origSize));

                tensorlib::functional::applyOn(wrongInitial, [](auto &&x) { x = tensorlib::random::uniform(-1., 1.); });

                auto imag = generator(wrongInitial);

                auto pred = discriminator(imag);

                lossg = -tensorlib::functional::sum(pred) / float(wrongInitial.size()[0]);

                optimG.zeroGrad();
                lossg.backward();
                optimG.step();
            }

            tot = duration_cast<milliseconds>(high_resolution_clock ::now() - now).count();
            std::cout << '\r' << i << ' ' << "Loss D: " << lossd.item() << "; "
                      << "Loss G: " << lossg.item() << "; "
                      << "Total Time: " << tot << "; "
                      << "Average Time: " << double(tot) / double(i) << "; "
                      << "Single Time: " << double(tot) / (i * dataset.getBatchSize()) << flush;
        }

        cout << '\n';
    }

    if (outD != "") {
        tensorlib::save(discriminator.parameters(), fstream(outD, std::ios::trunc | std::ios::binary | std::ios::out));
    }

    if (outG != "") {
        tensorlib::save(generator.parameters(), fstream(outG, std::ios::trunc | std::ios::binary | std::ios::out));
    }
}

int main() {
    using Tensor = tensorlib::Tensor<float, CPU>;
    using namespace std;
    using namespace std::chrono;
    using namespace tensorlib::optim;

    tensorlib::nn::init::manualSeed(42);

    string inp;

    try {
        while (true) {
            std::getline(cin, inp);

            stringstream ss;
            ss << inp;

            string command;
            ss >> command;

            if (command.empty()) {
                continue;
            } else if (command == "gen") {
                string model, imageName;
                int number, seed;
                ss >> model >> imageName >> number >> seed;

                generateData(model, imageName, number, seed);
            } else if (command == "train") {
                string modelPath, base, dataset;
                int epoch;
                int batchSize;

                ss >> modelPath >> base >> dataset >> epoch >> batchSize;

                std::cout << "Start training dataset <" << dataset << "> based on directory <" << base << ">" << '\n';
                std::cout << "Model save path: " << modelPath << ".discriminate"
                          << " " << modelPath << ".generate" << '\n';
                std::cout << "Number of epochs: " << epoch;

                Discriminator<float, CPU> discrim(1, 32, 16);
                Generator<float, CPU> generate(LATENT_DIM, 16, 32, 1);

                Adam<float, CPU> optimD(discrim.parameters(), 2e-4, {0.f, 0.9f});
                Adam<float, CPU> optimG(generate.parameters(), 2e-4, {0.f, 0.9f});

                SimpleDataset<float, CPU> data(loadDatas(fstream(dataset, std::ios::in), base), batchSize, true, false);

                trainData(discrim, optimD, generate, optimG, data, epoch, modelPath + ".discriminate",
                          modelPath + ".generate");
            } else if (command == "exit") {
                break;
            } else {
                std::cout << "gen default <image> <number> <seed>:" << '\n'
                          << "\tgenerate <number> images of 16x16 digits called from <image>-1 to <image>-<number>"
                             "into the path, using default pre-built network, using a random <seed>. "
                          << '\n'
                          << "gen <model-path> <image> <number> <seed>:" << '\n'
                          << "\tsimilar to gen default, but using another model in <model-path>" << '\n'
                          << "train <model-save-path> <base> <dataset> <epoch> <batchSize>:" << '\n'
                          << "\tuse the data saved in dataset to train a model (in which has a predesigned"
                             "structure) for epoch epochs. The optimizier is fixed. Using <batchSize> minibatch."
                          << '\n'
                          << "help: output help info of generation network." << '\n';
            }
        }
    } catch (const std::exception &err) {
        std::cout << err.what() << '\n';
    }
}