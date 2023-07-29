
#include "ParallelGaussianBlur.cuh"

static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);
static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}
#define CUDA_CHECK_ERROR(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

ParallelGaussianBlur::ParallelGaussianBlur(Pixel *pixels) {
    this->pixels = pixels;
}

__global__ void createKernel(double* kernel, int kernelSize, double sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x - kernelSize / 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y - kernelSize / 2;
    int idx = y * kernelSize + x;

    if (x >= 0 && x < kernelSize && y >= 0 && y < kernelSize) {
        kernel[idx] = exp(-(x * x + y * y) / (2 * sigma * sigma));
    }
}

void applyConvolution(Pixel* pixels, Pixel*blurredPixels, int width, int height, double* kernel, int kernelSize) {
    int kernelRadius = kernelSize / 2;
    for (int y = kernelRadius; y < height - kernelRadius; ++y) {
        for (int x = kernelRadius; x < width - kernelRadius; ++x) {
            double redAccumulator = 0.0;
            double greenAccumulator = 0.0;
            double blueAccumulator = 0.0;
            for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
                for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
                    int pixelX = x + kx;
                    int pixelY = y + ky;
                    int kernelIndex = (ky + kernelRadius) * kernelSize + (kx + kernelRadius);

                    double kernelValue = kernel[kernelIndex];
                    redAccumulator += pixels[pixelY * width + pixelX].red * kernelValue;
                    greenAccumulator += pixels[pixelY * width + pixelX].green * kernelValue;
                    blueAccumulator += pixels[pixelY * width + pixelX].blue * kernelValue;
                }
            }
            int red = std::round(redAccumulator);
            int green = std::round(greenAccumulator);
            int blue = std::round(blueAccumulator);
            red = std::max(0, std::min(255, red));
            green = std::max(0, std::min(255, green));
            blue = std::max(0, std::min(255, blue));
            blurredPixels[y * width + x].red = red;
            blurredPixels[y * width + x].green = green;
            blurredPixels[y * width + x].blue = blue;
        }
    }
}

__host__ Pixel *ParallelGaussianBlur::applyGaussianBlur(int width, int height, float sigma, int kernelSize) {
    //create Kernel
    int kernelSizeSquared = kernelSize * kernelSize;
    auto* kernel = new double[kernelSizeSquared];
    double* d_kernel;
    cudaMalloc((void**)&d_kernel, kernelSizeSquared * sizeof(double));
    dim3 blockDim(16, 16);
    dim3 gridDim((kernelSize + blockDim.x - 1) / blockDim.x, (kernelSize + blockDim.y - 1) / blockDim.y);
    createKernel<<<gridDim, blockDim>>>(d_kernel, kernelSize, sigma);
    cudaDeviceSynchronize();
    cudaMemcpy(kernel, d_kernel, kernelSizeSquared * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_kernel);
    //normalize the kernel
    double sum = 0.0;
    for (int i = 0; i < kernelSizeSquared; ++i) {
        sum += kernel[i];
    }
    for (int i = 0; i < kernelSizeSquared; ++i) {
        kernel[i] /= sum;
    }

    //apply convolution
    auto* blurredImage = new Pixel[width * height];
    Pixel* d_pixels;
    Pixel* d_blurredImage;
    double* d_normalizedKernel;
    cudaMalloc((void**)&d_normalizedKernel, kernelSize * kernelSize * sizeof(double));
    cudaMemcpy(d_normalizedKernel, kernel, kernelSize * kernelSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_pixels, width * height * sizeof(Pixel));
    cudaMemcpy(d_pixels, pixels, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_blurredImage, width * height * sizeof(Pixel));
    applyConvolution(pixels,blurredImage,width,height,kernel,kernelSize);
    cudaDeviceSynchronize();
    cudaMemcpy(blurredImage, d_blurredImage, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);

    //free memory
    cudaFree(d_kernel);
    cudaFree(d_pixels);
    cudaFree(d_blurredImage);
    return blurredImage;
}

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

void ParallelGaussianBlur::kickstartGPU() {
    int n =1000;
    int size = n * sizeof(int);
    int* h_a, * h_b, * h_c;
    int* d_a, * d_b, * d_c;
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vectorAdd<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}


