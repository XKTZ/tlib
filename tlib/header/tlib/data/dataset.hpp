#ifndef __TLIB__DATA_DATASET_HPP__
#define __TLIB__DATA_DATASET_HPP__

#include "tlib/tlib_base.hpp"
#include "tlib/tlib_tensor.hpp"
#include <random>
#include <utility>
#include <vector>

namespace tensorlib {
namespace data {

/**
 * A basic class for dataset util
 * @tparam Type Type of dataset
 * @tparam Concat Concatter
 */
template <typename Type, typename Concat>
class Dataset {
private:
    Concat concater;
    std::vector<Type> datas;
    SizeType batchSize;
    bool dropLast;
    bool shuffle;

public:
    /**
     * Iteratable of dataset, calling using .iter()
     */
    struct DatasetIterable {
        struct DatasetIterator {
            DatasetIterable &it;
            SizeType idx;
            Type tmp;

            explicit DatasetIterator(DatasetIterable &it, SizeType idx) : it(it), idx(idx), tmp() {
            }

            DatasetIterator &operator++() {
                idx += it.batch;
                idx = std::min(idx, it.to);
                return *this;
            }

            bool operator!=(const DatasetIterator &b) {
                return !((&it == &(b.it)) && b.idx == idx);
            }

            Type &operator*() {
                tmp = it.inRange(idx, idx + it.batch);
                return tmp;
            }
        };

        std::vector<Type> &orig;
        SizeType idx;
        SizeType batch;
        SizeType to;
        Concat &concat;

        DatasetIterable(std::vector<Type> &orig, SizeType batch, bool dropLast, Concat &concat)
            : orig(orig), idx(0), batch(batch), to(dropLast ? orig.size() - orig.size() % batch : orig.size()),
              concat(concat) {
        }

        DatasetIterator begin() {
            return DatasetIterator(*this, 0);
        }

        DatasetIterator end() {
            return DatasetIterator(*this, to);
        }

    private:
        Type inRange(SizeType i, SizeType j) {
            i = std::min(i, to);
            j = std::min(j, to);
            std::vector<Type *> result;
            for (; i < j; i++) {
                result.push_back(&(orig[i]));
            }
            return concat(result);
        }
    };

    /**
     * @param datas Datas
     * @param batchSize batch size
     * @param dropLast drop item that is smaller than batch size or not
     * @param shuffle shuffling or not
     */
    Dataset(const std::vector<Type> &datas, SizeType batchSize, bool dropLast = false, bool shuffle = false)
        : datas(datas), batchSize(batchSize), dropLast(dropLast), shuffle(shuffle) {
    }

    /**
     * @param datas Datas
     * @param batchSize batch size
     * @param dropLast drop item that is smaller than batch size or not
     * @param shuffle shuffling or not
     */
    Dataset(std::vector<Type> &&datas, SizeType batchSize, bool dropLast = false, bool shuffle = false)
        : datas(std::move(datas)), batchSize(batchSize), dropLast(dropLast), shuffle(shuffle) {
    }

    /**
     * Perform iteration
     * @return iterator
     */
    DatasetIterable iter() {
        if (this->shuffle) {
            std::shuffle(this->datas.begin(), this->datas.end(), std::mt19937(std::random_device()()));
        }
        return DatasetIterable(datas, batchSize, dropLast, concater);
    }

    /**
     * Get batch size
     * @return batch size
     */
    SizeType getBatchSize() const {
        return batchSize;
    }
};

/**
 * Simple dataset with only 1 tensor
 */
template <typename T, typename Device>
using SimpleDataset =
    Dataset<Tensor<T, Device>, decltype([]<typename A, typename D>(std::vector<Tensor<A, D> *> vec) {
                std::vector<tensor::BaseTensor<A, D> *> xvec;
                for (auto pr : vec) {
                    xvec.push_back(&(pr->getBase()));
                }
                return Tensor<A, D>(tensor::Computation<A, D>::concat(xvec));
            })>;

/**
 * Dataset with two tensors, used as prediction
 */
template <typename T, typename Device>
using PredictionDataset =
    Dataset<std::pair<Tensor<T, Device>, Tensor<T, Device>>,
            decltype([]<typename A, typename D>(std::vector<std::pair<Tensor<A, D>, Tensor<A, D>> *> vec) {
                std::vector<tensor::BaseTensor<A, D> *> xvec, yvec;
                for (auto pr : vec) {
                    xvec.push_back(&(pr->first.getBase()));
                    yvec.push_back(&(pr->second.getBase()));
                }
                return std::pair<Tensor<A, D>, Tensor<A, D>>{
                    Tensor<A, D>(tensor::Computation<A, D>::concat(xvec)),
                    Tensor<A, D>(tensor::Computation<A, D>::concat(yvec)),
                };
            })>;
}; // namespace data
}; // namespace tensorlib

#endif
