#ifndef KERNELIMAGEPROCESSING_PPMIMAGE_H
#define KERNELIMAGEPROCESSING_PPMIMAGE_H


#include <string>
#include <iostream>
#include <fstream>
#include <vector>

struct Pixel {
    int red, green, blue;
};

struct PixelSOA{
    int* red;
    int* green;
    int* blue;
};

class PPMImage {
private:
    int width;
    int height;
    Pixel* pixels;
    int size;
public:
    PPMImage(const std::string& filepath);
    ~PPMImage();
    int getWidth() const;
    int getHeight() const;
    int getSize() const;
    Pixel* getPixels();

    bool writePPMImage(const std::string &filePath);
    bool writePPMImage(const std::string &filePath, Pixel * newPixels);
};



#endif //KERNELIMAGEPROCESSING_PPMIMAGE_H
