#include <filesystem>
#include "PPMImage.h"

PPMImage::PPMImage(const std::string &filepath){
    std::ifstream ppm_file(filepath);
    if (!ppm_file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
    }
    std::string format;
    ppm_file >> format;
    if (format != "P3") {
        std::cerr << "Error: Invalid PPM file. Expected P3 format." << std::endl;
    }
    ppm_file >> width >> height;
    int max_color_value;
    ppm_file >> max_color_value;
    if (max_color_value > 255) {
        std::cerr << "Error: Invalid PPM file. Maximum color value cannot exceed 255." << std::endl;
    }

    pixels = new Pixel[width * height];
    int r, g, b;
    this-> size=0;
    while (ppm_file >> r >> g >> b) {
        pixels[size]= {r, g, b};
        this->size++;
    }
}

PPMImage::~PPMImage() {
    delete[] pixels;
}

int PPMImage::getHeight() const {
    return height;
}

int PPMImage::getWidth() const {
    return width;
}

Pixel *PPMImage::getPixels() {
    return pixels;
}

int PPMImage::getSize() const {
    return size;
}

bool PPMImage::writePPMImage(const std::string& filePath) {
    std::ofstream ppm_file(filePath);
    if (!ppm_file.is_open()) {
        std::cerr << "Error: Could not create file " << filePath << std::endl;
        return false;
    }
    ppm_file << "P3\n";
    ppm_file << width << " " << height << "\n";
    ppm_file << "255\n";
    for (int i = 0; i < size; i++) {
        ppm_file << pixels[i].red << " " << pixels[i].green << " " << pixels[i].blue << "\n";
    }
    return true;
}

bool PPMImage::writePPMImage(const std::string &filePath, Pixel *newPixels) {
    std::ofstream ppm_file(filePath);
    if (!ppm_file.is_open()) {
        std::cerr << "Error: Could not create file " << filePath << std::endl;
        return false;
    }
    ppm_file << "P3\n";
    ppm_file << width << " " << height << "\n";
    ppm_file << "255\n";
    for (int i = 0; i < size; i++) {
        ppm_file << newPixels[i].red << " " << newPixels[i].green << " " << newPixels[i].blue << "\n";
    }
    return true;
}
