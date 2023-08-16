
#include <string>
#include <iostream>
#include <fstream>
#include "Result.h"

Result::Result(int repeats,int minKernel, int maxKernel, int kernelStep, int imagesNumber) {
    for(int i =0; i < 4; i++){
        auto *meanParallelTimesArray = new double[((maxKernel - minKernel)/kernelStep)+1];
        for(int j=0; j <=((maxKernel - minKernel)/kernelStep); j++){
            meanParallelTimesArray[j]=0.0;
        }
        auto *meanSequentialTimesArray = new double[((maxKernel - minKernel)/kernelStep)+1];
        for(int j=0; j <=((maxKernel - minKernel)/kernelStep); j++){
            meanSequentialTimesArray[j]=0.0;
        }
        this->meanParallelTimes.push_back(meanParallelTimesArray);
        this->meanSequentialTimes.push_back(meanSequentialTimesArray);
    }
    this->repeats=repeats;
    this->minKernel= minKernel;
    this->maxKernel =maxKernel;
    this->kernelStep=kernelStep;
}

void Result::saveResults() {
    auto* strings= new std::string[4];
    strings[0]="Image500";
    strings[1]="Image1000";
    strings[2]="Image2000";
    strings[3]="Image4000";
    std::string currentFilePath(__FILE__);
    std::size_t found = currentFilePath.find_last_of("\\");
    std::string folderPath = currentFilePath.substr(0, found + 1);
    std::string filePath = folderPath + "\\Sequential\\results.txt";
    std::ofstream outputSequentialTimes(filePath);
    if (outputSequentialTimes.is_open()) {
        for (int i = 0; i < 4; i++) {
            outputSequentialTimes << strings[i] << "{" << std::endl;
            int steps = ((maxKernel - minKernel)/kernelStep);
            for(int j= 0; j <= steps; j++){
                outputSequentialTimes << "Kernel size" << j << " : [" << std::endl;
                outputSequentialTimes << "(" << (meanSequentialTimes[i][j] / repeats) << ")" << std::endl;
                outputSequentialTimes << "]" << std::endl;
            }

            outputSequentialTimes << "}" << std::endl;
        }
        outputSequentialTimes.close();
    }
    else {
        std::cout << "Unable to open the file." << std::endl;
    }

    filePath = folderPath + "\\Parallel\\results.txt";
    std::ofstream outputParallelTimes(filePath);
    if (outputParallelTimes.is_open()) {
        for (int i = 0; i < 4; i++) {
            outputParallelTimes << strings[i] << "{" << std::endl;
            int steps = ((maxKernel - minKernel)/kernelStep);
            for(int j= 0; j <= steps; j++){
                outputParallelTimes << "Kernel size" << j << " : [" << std::endl;
                outputParallelTimes << "(" << meanParallelTimes[i][j]/ repeats << ")" << std::endl;
                outputParallelTimes << "]" << std::endl;
            }

            outputParallelTimes << "}" << std::endl;
        }
        outputParallelTimes.close();
    }
    else {
        std::cout << "Unable to open the file." << std::endl;
    }

    filePath = folderPath + "\\Parallel\\speedups.txt";
    std::ofstream outputParallelSpeedup(filePath);
    if (outputParallelSpeedup.is_open()) {
        for (int i = 0; i < 4; i++) {
            outputParallelSpeedup << strings[i] << "{" << std::endl;
            int steps = ((maxKernel - minKernel)/kernelStep);
            for(int j= 0; j <= steps; j++){
                outputParallelSpeedup << "Kernel size" << j << " : " << "(" << (meanSequentialTimes[i][j]/meanParallelTimes[i][j]) << ")" << std::endl;
            }

            outputParallelSpeedup << "}" << std::endl;
        }
        outputParallelSpeedup.close();
    }
    else {
        std::cout << "Unable to open the file." << std::endl;
    }
}

void Result::addResult(long long int time, bool isSequential, int imageNumber, int kernelNumber) {
    if(isSequential){
        meanSequentialTimes[imageNumber][kernelNumber] +=time;
    }else{
        meanParallelTimes[imageNumber][kernelNumber] +=time;
    }
}



