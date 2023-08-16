#include "SequentialGaussianBlur.h"

SequentialGaussianBlur::SequentialGaussianBlur(Pixel *pixels) {
    this->pixels = pixels;
}

Pixel * SequentialGaussianBlur::applyGaussianBlur(int width, int height, float sigma, int kernelSize) {
    //Creating kernel
    auto** kernel = new double*[kernelSize];
    for(int i=0; i < kernelSize; i++){
        kernel[i] = new double[kernelSize];
    }
    double sum = 0.0;
    for (int x = -kernelSize / 2; x <= kernelSize / 2; ++x) {
        for (int y = -kernelSize / 2; y <= kernelSize / 2; ++y) {
            kernel[x + kernelSize / 2][y + kernelSize / 2] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[x + kernelSize / 2][y + kernelSize / 2];
        }
    }
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel[i][j] /= sum;
        }
    }

    // Apply convolution and Gaussian blur
    auto* blurredImage = new Pixel[width * height];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int redAccumulator = 0, greenAccumulator = 0, blueAccumulator = 0;

            for (int i = 0; i < kernelSize; ++i) {
                for (int j = 0; j < kernelSize; ++j) {
                    int offsetX = x + i - kernelSize / 2;
                    int offsetY = y + j - kernelSize / 2;

                    if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height) {
                        int index = offsetY * width + offsetX;
                        redAccumulator += static_cast<int>(pixels[index].red * kernel[i][j]);
                        greenAccumulator += static_cast<int>(pixels[index].green * kernel[i][j]);
                        blueAccumulator += static_cast<int>(pixels[index].blue * kernel[i][j]);
                    }
                }
            }

            int index = y * width + x;
            blurredImage[index].red = redAccumulator;
            blurredImage[index].green = greenAccumulator;
            blurredImage[index].blue = blueAccumulator;
        }
    }
    return blurredImage;
}

