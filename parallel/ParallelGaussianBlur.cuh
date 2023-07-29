#ifndef KERNELIMAGEPROCESSING_PARALLELGAUSSIANBLUR_CUH
#define KERNELIMAGEPROCESSING_PARALLELGAUSSIANBLUR_CUH

#include "../Image/PPMImage.h"

class ParallelGaussianBlur{
    private:
        Pixel* pixels;
    public:
        ParallelGaussianBlur(Pixel *pixels);
        Pixel *applyGaussianBlur(int width, int height, float sigma, int kernelSize);
        void kickstartGPU();
};

#endif //KERNELIMAGEPROCESSING_PARALLELGAUSSIANBLUR_CUH
