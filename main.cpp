#include <iostream>
#include <filesystem>
#include "Image/PPMImage.h"
#include "sequential/SequentialGaussianBlur.h"

void savePPM(const Pixel* pixels, int width, int height, const std::string& filename) {
    std::string currentFilePath = __FILE__;
    std::filesystem::path currentPath(currentFilePath);
    std::filesystem::path parentPath = currentPath.parent_path();
    std::filesystem::path imagePath = parentPath / filename;
    std::ofstream file(imagePath);
    if (!file.is_open()) {
        std::cout << "Failed to open file for writing: " << imagePath << std::endl;
        return;
    }

    file << "P3" << std::endl;
    file << width << " " << height << std::endl;
    file << "255" << std::endl;

    for (int i = 0; i < width * height; i++) {
        file << static_cast<int>(pixels[i].red) << " ";
        file << static_cast<int>(pixels[i].green) << " ";
        file << static_cast<int>(pixels[i].blue) << " ";

        if ((i + 1) % width == 0) {
            file << std::endl;
        }
    }

    file.close();
    std::cout << "Image saved to file: " << filename << std::endl;
}

int main() {
    PPMImage image = PPMImage("image.ppm");
    int imageWidth = image.getWidth();
    int imageHeight = image.getHeight();
    Pixel* pixels = image.getPixels();
    int size = image.getSize();
    float sigma =0.5;
    std::cout << "Reading image of size: "<< imageWidth <<"x"<< imageHeight << std::endl;

    SequentialGaussianBlur sequentialGaussianBlur = SequentialGaussianBlur(pixels);
    Pixel* blurredPixels = sequentialGaussianBlur.applyGaussianBlur(imageWidth, imageHeight, sigma);
    savePPM(pixels, imageWidth, imageHeight, "Results/Sequential/blurredImage.ppm");
    return 0;
}
