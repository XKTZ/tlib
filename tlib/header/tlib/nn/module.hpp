//
// Created by xktz on 11/27/23.
//

#ifndef __TLIB__NN_MODULE_HPP__
#define __TLIB__NN_MODULE_HPP__

#include "tlib/tlib_tensor.hpp"
#include "init.hpp"
#include <random>
#include <vector>

namespace tensorlib {
namespace nn {

/**
 * Status of the module
 */
enum ModeStatus { TRAIN, EVAL };

/**
 * ParameterGroup is the class representing for all parameters & status of a specific module
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
struct ParameterGroup {
private:
public:
    std::string_view name;
    std::vector<tensorlib::tensor::Tensor<T, Device> *> params;
    std::vector<ParameterGroup<T, Device> *> layers;
    std::vector<std::function<void(ModeStatus status)>> statusSetter;

    explicit ParameterGroup(std::string_view name) : name(name) {
    }

    /**
     * Load the parameter group from a istream
     * @param in istream
     */
    template <typename S>
        requires std::is_base_of_v<std::istream, std::decay_t<S>>
    void load(S &&in) const {
        if (!in.good()) {
            throw std::runtime_error("Cannot load from source");
        }
        auto loadTensor = [&in](Tensor<T, Device> &t) {
            for (auto &v : t.getBase()) {
                v = util::DataIO<T>::read(in);
            }
        };
        for (auto x : this->params) {
            loadTensor(*x);
        }
        for (auto lay : layers) {
            lay->load(in);
        }
    }

    /**
     * Apply a specific function f(Tensor) -> void to all data in parameter group
     * @param f a specific function
     */
    void apply(auto &&f) const {
        for (auto x : params) {
            f(*x);
        }
        for (auto &lay : layers) {
            lay->apply(std::forward<decltype(f)>(f));
        }
    }

    /**
     * Output a parameter group to an ostream
     * @param out ostream
     * @param tab number of tabs
     */
    void output(std::ostream &out, int tab = 0) const {
        for (int i = 0; i < tab; i++)
            std::cout << " ";
        std::cout << "- " << name << '\n';
        for (auto p : layers) {
            p->output(out, tab + 4);
        }
    }
};

/**
 * Module is the basic class for different type of modules like convolution, linear, etc.
 * Implement the "forward(const Args &...)" function module to let it be a valid layer
 * @tparam T Type
 * @tparam Device Device
 * @tparam Return Return type
 * @tparam Args Arguments
 */
template <typename T, typename Device, typename Return, typename... Args>
class Module {

private:
    ParameterGroup<T, Device> paramGroup;

    ModeStatus status;

protected:
    Module() : paramGroup(this->name()), status(TRAIN) {
    }

    /**
     * Register a specific parameter of module
     * @tparam inputDim input dimension of the parameter p
     * @tparam RequireGrad p is require grad or not
     * @tparam Init initialize p or not
     * @param p parameter p
     * @return p's reference
     */
    template <SizeType inputDim = 0, bool RequireGrad = true, bool Init = true>
    Tensor<T, Device> &registerParameter(Tensor<T, Device> &p) {
        this->paramGroup.params.push_back(&p);
        if constexpr (RequireGrad) {
            p.requireGrad();
        }
        if constexpr (Init) {
            init::kaimingInitialization<inputDim>(p);
        }
        return p;
    }

    /**
     * Register a specific module into the module
     * @tparam Ret return type of module
     * @tparam ArgsP arguments of the module
     * @param p
     * @return module's reference
     */
    template <typename Ret, typename... ArgsP>
    Module<T, Device, Ret, ArgsP...> &registerParameter(Module<T, Device, Ret, ArgsP...> &p) {
        this->paramGroup.layers.push_back(&(p.paramGroup));
        this->paramGroup.statusSetter.push_back([&p](ModeStatus stat) { p.setStatus(stat); });
        return p;
    }

    /**
     * Wrapper function to register all stuff
     * @param params all stuff
     */
    void registerAll(auto &&...params) {
        (this->registerParameter(std::forward<decltype(params)>(params)), ...);
    }

public:
    /**
     * Get the parameters registered from module
     * @return parameters
     */
    virtual const ParameterGroup<T, Device> &parameters() const final {
        return this->paramGroup;
    }

    /**
     * Implement the forward function so that we can call it in Module
     * @param args args
     * @return return
     */
    virtual Return forward(const Args &...args) = 0;

    /**
     * Operator() function, implemented. User should not touch it
     * @tparam InputArgs input arguments
     * @param args arguments
     * @return return value
     */
    template <typename... InputArgs>
    Return operator()(InputArgs &&...args) {
        return this->forward(std::forward<InputArgs>(args)...);
    }

    /**
     * Set the status of the module. If it is train mode, it defaultly require grad to all tensors. Otherwise it nograd
     * @param stat
     */
    virtual void setStatus(ModeStatus stat) final {
        if (this->status == stat)
            return;
        this->status = stat;
        if (stat == ModeStatus::TRAIN) {
            for (auto x : this->paramGroup.params) {
                x->requireGrad();
            }
        } else {
            for (auto x : this->paramGroup.params) {
                x->noGrad();
            }
        }
        for (auto &setter : paramGroup.statusSetter) {
            setter(stat);
        }
    }

    /**
     * get the name of module
     * @return name
     */
    virtual std::string_view name() const {
        return "Module";
    }

    /**
     * Get status of module
     * @return status
     */
    ModeStatus getStatus() const {
        return status;
    }

    virtual ~Module() = default;
};

}; // namespace nn

/**
 * Save a specific module into ostream
 * @tparam T Type
 * @tparam Device Device
 * @tparam S ostream type
 * @param paramGroup parameter
 * @param out ostream
 */
template <typename T, typename Device, typename S>
void save(const nn::ParameterGroup<T, Device> &paramGroup, S &&out) {
    auto writeTensor = [&out](Tensor<T, Device> &t) {
        for (auto x : t.getBase()) {
            util::DataIO<T>::write(out, x);
        }
    };
    for (auto x : paramGroup.params) {
        writeTensor(*x);
    }
    for (auto lay : paramGroup.layers) {
        save(*lay, out);
    }
}

}; // namespace tensorlib

template <typename T, typename Device, typename Ret, typename... Args>
std::ostream &operator<<(std::ostream &out, const tensorlib::nn::Module<T, Device, Ret, Args...> &module) {
    module.parameters().output(out);
    return out;
}

#endif // __TLIB__NN_MODULE_HPP__
