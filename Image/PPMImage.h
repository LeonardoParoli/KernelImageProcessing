#ifndef KERNELIMAGEPROCESSING_PPMIMAGE_H
#define KERNELIMAGEPROCESSING_PPMIMAGE_H


#include <string>
#include <iostream>
#include <fstream>

struct Pixel {
    unsigned char red, green, blue;
};

class PPMImage {
private:
    int width;
    int height;
    Pixel* pixels;
    int size;
public:
    PPMImage(const std::string& filename);
    ~PPMImage();
    int getWidth() const;
    int getHeight() const;
    int getSize() const;
    Pixel* getPixels();
};



#endif //KERNELIMAGEPROCESSING_PPMIMAGE_H
