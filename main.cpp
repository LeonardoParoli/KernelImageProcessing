#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include "Image/PPMImage.h"
#include "sequential/SequentialGaussianBlur.h"
#include "Results/Result.h"
#include "parallel/ParallelGaussianBlur.cuh"

int main() {
    int kernelIntervalStart =7; // Min kernel dimension, must be odd
    int kernelIntervalEnd = 9; //Max kernel dimension, must be odd (tried 15)
    int kernelStep = 2; //must be even
    int repeats = 3;
    float sigma = 1.5;
    bool printConsole = true;


    const char* currentFilePath = __FILE__;
    std::string fullPath = currentFilePath;
    size_t lastSeparatorPos = fullPath.find_last_of("/\\");
    std::string folderPath = fullPath.substr(0, lastSeparatorPos);
    std::cout << "Loading images ..." << std::endl;
    PPMImage image500 = PPMImage(folderPath + "/Image/image500.ppm");
    std::cout << " Image 500x500 loaded ..." << std::endl;
    PPMImage image1000 = PPMImage(folderPath + "/Image/image1000.ppm");
    std::cout << " Image 1000x1000 loaded ..." << std::endl;
    PPMImage image2000 = PPMImage(folderPath + "/Image/image2000.ppm");
    std::cout << " Image 2000x2000 loaded ..." << std::endl;
    PPMImage image4000 = PPMImage(folderPath + "/Image/image500.ppm");
    std::cout << " Image 4000x4000 loaded ..." << std::endl;
    std::vector<PPMImage*> images;
    images.push_back(&image500);
    images.push_back(&image1000);
    images.push_back(&image2000);
    images.push_back(&image4000);
    Result results = Result(repeats, kernelIntervalStart, kernelIntervalEnd, kernelStep,images.size());
    //Starting sequential Kernel Image Processing
    std::cout << "Running sequential test..."<< std::endl;
    for(int imageNumber=0; imageNumber < images.size(); imageNumber++){
        std::cout << "Using image number: "<< imageNumber << std::endl;
        PPMImage *image = images[imageNumber];
        Pixel* pixels = image->getPixels();
        for (int k = kernelIntervalStart; k <= kernelIntervalEnd; k += kernelStep) {
            std::cout << "-Using kernel of size: "<< k << std::endl;
            Pixel * blurredImage;
            SequentialGaussianBlur oprt = SequentialGaussianBlur(pixels);
            for (int i = 0; i < repeats; i++) {
                std::chrono::high_resolution_clock::time_point startSequential = std::chrono::high_resolution_clock::now();
                blurredImage = oprt.applyGaussianBlur(image->getWidth(),image->getHeight(),sigma,k);
                std::chrono::high_resolution_clock::time_point endSequential = std::chrono::high_resolution_clock::now();
                auto durationSequential = std::chrono::duration_cast<std::chrono::milliseconds>(endSequential - startSequential).count();
                results.addResult(durationSequential, true, imageNumber, (k-kernelIntervalStart)/kernelStep);
                if(printConsole){
                    std::cout << std::fixed << std::setprecision(4) << "--Sequential Execution time: " << durationSequential << " milliseconds" << std::endl;
                }
                //checking correctness of the algorithm
                if(imageNumber == 0 && i == 0 && k == kernelIntervalStart){
                    image->writePPMImage(folderPath +"/Image/correctness/sequential/blurredImage500.ppm",blurredImage);
                }
            }

        }
    }

    std::cout << "////////////////////////////////////////" << std::endl;
    // Starting parallel Kernel Image Processing
    std::cout << "Running parallel CUDA test..."<< std::endl;
    for(int imageNumber=0; imageNumber <images.size(); imageNumber++){
        std::cout << "Using image number: "<< imageNumber << std::endl;
        PPMImage *image = images[imageNumber];
        Pixel* pixels = image->getPixels();
        for (int k = kernelIntervalStart; k <= kernelIntervalEnd; k += kernelStep) {
            std::cout << "-Using kernel of size: "<< k << std::endl;
            auto * blurredImageCUDA= new Pixel[image->getWidth() * image->getHeight()];
            ParallelGaussianBlur oprt = ParallelGaussianBlur(pixels,image->getSize());
            for (int i = 0; i < repeats; i++) {
                if(imageNumber == 0 && k == kernelIntervalStart && i == 0){
                    oprt.kickstartGPU();
                }
                std::chrono::high_resolution_clock::time_point startParallelCUDA = std::chrono::high_resolution_clock::now();
                oprt.applyGaussianBlur(image->getWidth(),image->getHeight(),sigma,k, blurredImageCUDA);
                std::chrono::high_resolution_clock::time_point endParallelCUDA= std::chrono::high_resolution_clock::now();
                auto durationParallelCUDA = std::chrono::duration_cast<std::chrono::milliseconds>(endParallelCUDA - startParallelCUDA).count();
                results.addResult(durationParallelCUDA, false, imageNumber, (k-kernelIntervalStart)/kernelStep);
                if(printConsole){
                    std::cout << std::fixed << std::setprecision(4) << "--Parallel Execution time: " << durationParallelCUDA << " milliseconds" << std::endl;
                }
                //checking correctness of the algorithm
                if(imageNumber == 0 && i == 0 && k == kernelIntervalStart){
                    image->writePPMImage(folderPath +"/Image/correctness/parallel/blurredImage500.ppm",blurredImageCUDA);
                }
            }
        }
    }

    //saving results to .txt
    results.saveResults();
    return 0;
}
