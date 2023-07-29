#ifndef KERNELIMAGEPROCESSING_SEQUENTIALGAUSSIANBLUR_H
#define KERNELIMAGEPROCESSING_SEQUENTIALGAUSSIANBLUR_H

#include "../Image/PPMImage.h"

class SequentialGaussianBlur {
    private:
        Pixel* pixels;
    public:
        SequentialGaussianBlur(Pixel *pixels);
        Pixel *applyGaussianBlur(int width, int height, float sigma, int kernelSize);
};


#endif //KERNELIMAGEPROCESSING_SEQUENTIALGAUSSIANBLUR_H
