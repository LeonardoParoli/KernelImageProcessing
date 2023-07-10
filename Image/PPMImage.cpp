#include <filesystem>
#include "PPMImage.h"

PPMImage::PPMImage(const std::string &filename) : pixels(nullptr) {
    std::string currentFilePath = __FILE__;
    std::filesystem::path currentPath(currentFilePath);
    std::filesystem::path parentPath = currentPath.parent_path();
    std::filesystem::path imagePath = parentPath / filename;
    std::ifstream file(imagePath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << imagePath << std::endl;
        return;
    }

    std::string format;
    file >> format;
    if (format != "P3") {
        std::cerr << "Invalid PPM format." << std::endl;
        return;
    }

    file >> width >> height;
    int maxValue;
    file >> maxValue;
    file.get();
    size = width*height;
    pixels = new Pixel[size];
    file.read(reinterpret_cast<char*>(pixels), size * sizeof(Pixel));
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
