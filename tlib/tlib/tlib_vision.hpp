#ifndef __TLIB__VISION_HPP__
#define __TLIB__VISION_HPP__

#include "png.hpp"
#include "tlib/tlib_dimensional.hpp"
#include "tlib/tlib_tensor.hpp"
#include <ios>
#include <iostream>

namespace tensorlib {
namespace vision {

/**
 * Type of image
 */
enum ImageType { IMAGE_PNG };

/**
 * Pixel type
 * @tparam Type type of image
 */
template <ImageType Type>
struct Pixel;

/**
 * Reader of image
 * @tparam Type type of image
 */
template <ImageType Type>
struct ImageReader;

/**
 * Writer of image
 * @tparam Type type of image
 */
template <ImageType Type>
struct ImageWriter;

/**
 * Pixel for PNG
 */
template <>
struct Pixel<IMAGE_PNG> {
    using RGB = png::rgb_pixel;
    using GRAY = png::gray_pixel;
};

/**
 * Pixel data, defined as an array with size S
 * @tparam S size of data, like RGB = 3
 */
template <SizeType S>
using PixelData = std::array<SizeType, S>;

/**
 * Pixel info
 * @tparam Type Type of image
 * @tparam Pix Type of pixel
 */
template <ImageType Type, typename Pix>
struct PixelInfo;

template <>
struct PixelInfo<IMAGE_PNG, Pixel<IMAGE_PNG>::RGB> {
    static constexpr SizeType MAX = 255;
    static constexpr SizeType SIZE = 3;

    static PixelData<SIZE> data(const Pixel<IMAGE_PNG>::RGB &pix) {
        PixelData<SIZE> result;
        result[0] = pix.red;
        result[1] = pix.green;
        result[2] = pix.blue;
        return result;
    }

    static void writeInto(auto &rgb, Pixel<IMAGE_PNG>::RGB &pix) {
        pix.red = rgb[0];
        pix.green = rgb[1];
        pix.blue = rgb[2];
    }
};

template <>
struct PixelInfo<IMAGE_PNG, Pixel<IMAGE_PNG>::GRAY> {
    static constexpr SizeType MAX = 255;
    static constexpr SizeType SIZE = 1;

    static PixelData<SIZE> data(const Pixel<IMAGE_PNG>::GRAY &pix) {
        PixelData<SIZE> result;
        result[0] = pix;
        return result;
    }

    static void writeInto(auto &rgb, Pixel<IMAGE_PNG>::GRAY &pix) {
        pix = rgb[0];
    }
};

template <>
struct ImageReader<IMAGE_PNG> {
    template <typename P, typename T, typename Device, typename S>
        requires std::is_base_of_v<std::istream, std::decay_t<S>>
    static Tensor<T, Device> loadImage(S &&in) {
        png::image<P> image(in);

        SizeType w = image.get_width();
        SizeType h = image.get_height();

        Tensor<T, Device> result(PixelInfo<IMAGE_PNG, P>::SIZE, h, w);

        if constexpr (requires { result.raw(); }) {
            auto ptr = result.raw();
            for (SizeType i = 0; i < h; i++) {
                for (SizeType j = 0; j < w; j++) {
                    auto data = PixelInfo<IMAGE_PNG, P>::data(image[i][j]);
                    for (SizeType c = 0; c < PixelInfo<IMAGE_PNG, P>::SIZE; c++) {
                        ptr[i * (w * PixelInfo<IMAGE_PNG, P>::SIZE) + j * PixelInfo<IMAGE_PNG, P>::SIZE + c] =
                            T(data[c]) / T(PixelInfo<IMAGE_PNG, P>::MAX);
                    }
                }
            }
        } else {
            for (SizeType i = 0; i < h; i++) {
                for (SizeType j = 0; j < w; j++) {
                    auto data = PixelInfo<IMAGE_PNG, P>::data(image[i][j]);
                    for (SizeType c = 0; c < PixelInfo<IMAGE_PNG, P>::SIZE; c++) {
                        result.get({c, i, j}) = T(data[c]) / T(PixelInfo<IMAGE_PNG, P>::MAX);
                    }
                }
            }
        }

        return result;
    }
};

using PNGPixelRGB = Pixel<IMAGE_PNG>::RGB;
using PNGPixelGray = Pixel<IMAGE_PNG>::GRAY;

using PNGReader = ImageReader<IMAGE_PNG>;

template <>
struct ImageWriter<IMAGE_PNG> {
    template <typename P, typename T, typename Device, typename S>
        requires std::is_base_of_v<std::ostream, std::remove_reference_t<S>>
    static void writeImage(S &&out, const Tensor<T, Device> &t) {
        SizeType h = t.size()[1], w = t.size()[2];
        png::image<P> image(w, h);
        for (SizeType i = 0; i < h; i++) {
            for (SizeType j = 0; j < w; j++) {
                std::array<int, PixelInfo<IMAGE_PNG, P>::SIZE> pix;
                for (SizeType k = 0; k < PixelInfo<IMAGE_PNG, P>::SIZE; k++) {
                    auto v = std::min(std::max(int(t.get({k, i, j}) * T(PixelInfo<IMAGE_PNG, P>::MAX)), 0),
                                      int(PixelInfo<IMAGE_PNG, P>::MAX));
                    pix[k] = v;
                }
                PixelInfo<IMAGE_PNG, P>::writeInto(pix, image[i][j]);
            }
        }
        image.write_stream(out);
    }
};

using PNGWriter = ImageWriter<IMAGE_PNG>;

}; // namespace vision
}; // namespace tensorlib

#endif
