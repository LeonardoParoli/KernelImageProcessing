#include <cmath>
#include "SequentialGaussianBlur.h"

Pixel * SequentialGaussianBlur::applyGaussianBlur(int width, int height, float sigma) {
    int kernelSize = 5;
    int radius = kernelSize / 2;
    float variance = sigma * sigma;
    float sum = 0.0;

    float kernel[kernelSize][kernelSize];
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            float weight = exp(-(i * i + j * j) / (2 * variance));
            kernel[i + radius][j + radius] = weight;
            sum += weight;
        }
    }

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    Pixel* blurredImage = new Pixel[width * height];

    for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
            float sumR = 0.0, sumG = 0.0, sumB = 0.0;

            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    int pixelIndex = (y + i) * width + (x + j);
                    unsigned char pixelR = pixels[pixelIndex].red;
                    unsigned char pixelG = pixels[pixelIndex].green;
                    unsigned char pixelB = pixels[pixelIndex].blue;

                    float weight = kernel[i + radius][j + radius];

                    sumR += pixelR * weight;
                    sumG += pixelG * weight;
                    sumB += pixelB * weight;
                }
            }

            int currentPixelIndex = y * width + x;
            blurredImage[currentPixelIndex].red = static_cast<unsigned char>(sumR);
            blurredImage[currentPixelIndex].green = static_cast<unsigned char>(sumG);
            blurredImage[currentPixelIndex].blue = static_cast<unsigned char>(sumB);
        }
    }
    std::copy(blurredImage, blurredImage + (width * height), pixels);

    return blurredImage;
}

SequentialGaussianBlur::SequentialGaussianBlur(Pixel *pixels) {
    this->pixels = pixels;
}
