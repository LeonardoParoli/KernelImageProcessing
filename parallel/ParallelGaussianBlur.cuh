#ifndef KERNELIMAGEPROCESSING_PARALLELGAUSSIANBLUR_CUH
#define KERNELIMAGEPROCESSING_PARALLELGAUSSIANBLUR_CUH

#include "../Image/PPMImage.h"

class ParallelGaussianBlur{
    private:
        PixelSOA pixels;
    public:
        ParallelGaussianBlur(Pixel* pixels, int size);
        void applyGaussianBlur(int width, int height, float sigma, int kernelSize, Pixel* blurredImage);
        void kickstartGPU();
};

#endif //KERNELIMAGEPROCESSING_PARALLELGAUSSIANBLUR_CUH
