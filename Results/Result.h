#ifndef KERNELIMAGEPROCESSING_RESULT_H
#define KERNELIMAGEPROCESSING_RESULT_H

#include <vector>

class Result{
    private:
        std::vector<double*> meanSequentialTimes;
        std::vector<double*> meanParallelTimes;
        int repeats;
        int minKernel;
        int maxKernel;
        int kernelStep;
    public:
        Result(int repeats, int minKernel, int maxKernel, int kernelStep, int imagesNumber);
        void saveResults(int kernelSizeStart, int kernelStep);
        void addResult(long long int time,bool isSequential, int imageNumber, int kernelNumber);
};

#endif //KERNELIMAGEPROCESSING_RESULT_H
