
#include "ParallelGaussianBlur.cuh"

static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);
static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}
#define CUDA_CHECK_ERROR(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

ParallelGaussianBlur::ParallelGaussianBlur(Pixel* pixels, int size) {
    int *contiguousMemory = new int[size * 3];
    PixelSOA pixelsSOA ={contiguousMemory,contiguousMemory + size,contiguousMemory + 2 * size};
    for(int i=0; i < size; i++ ){
        pixelsSOA.red[i]=pixels[i].red;
        pixelsSOA.green[i]=pixels[i].green;
        pixelsSOA.blue[i]=pixels[i].blue;
    }
    this->pixels = pixelsSOA;
}

__global__ void generateAndNormalizeGaussianKernel(float *d_kernel, int kernelSize, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < kernelSize && y < kernelSize) {
        int centerX = kernelSize / 2;
        int centerY = kernelSize / 2;
        int xOffset = x - centerX;
        int yOffset = y - centerY;

        float value = exp(-(xOffset * xOffset + yOffset * yOffset) / (2 * sigma * sigma));
        d_kernel[y * kernelSize + x] = value;
    }

    __syncthreads(); // Ensure all threads have computed their values

    if (x == 0 && y == 0) {
        float sum = 0.0;
        for (int i = 0; i < kernelSize * kernelSize; ++i) {
            sum += d_kernel[i];
        }
        for (int i = 0; i < kernelSize * kernelSize; ++i) {
            d_kernel[i] /= sum;
        }
    }
}

struct DoublePixelSOA{
    double* red;
    double* green;
    double* blue;
};

__global__ void applyGaussianFilterKernelTiledTest(PixelSOA d_pixels, DoublePixelSOA d_blurredImage, int imageWidth, int imageHeight, int kernelSize, const float* d_kernel) {
    d_pixels.green = d_pixels.red + imageWidth*imageHeight;
    d_pixels.blue = d_pixels.green + imageWidth*imageHeight;
    d_blurredImage.green = d_blurredImage.red + imageWidth*imageHeight;
    d_blurredImage.blue = d_blurredImage.green + imageWidth*imageHeight;
    int w = blockDim.x + (kernelSize - 1);
    // shared memory
    extern __shared__ float sharedMemory[]; //Contiguous memory
    float* sharedRed = sharedMemory;
    float* sharedGreen = sharedMemory + w * w;
    float* sharedBlue = sharedGreen + w * w;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int kRadius = kernelSize / 2;
    int batches = (w * w) / ( blockDim.x  *  blockDim.x ) + 1;  // ceiling
    int dest, destY, destX, srcY, srcX;
    double newValRed, newValGreen, newValBlue;

    // shared memory loading
    for (int b = 0; b < batches; b++) {
        dest = threadIdx.y * blockDim.x  + threadIdx.x + b * blockDim.x  * blockDim.x ;
        destY = dest / w;
        destX = dest % w;
        srcY = blockIdx.y * blockDim.x  + destY - kRadius;
        srcX = blockIdx.x * blockDim.x  + destX - kRadius;

        if (destY < w) {
            if (srcX < 0 || srcX >= imageWidth) {        // edge handling: extend
                srcX = (srcX < 0) ? 0 : (imageWidth - 1);
            }
            if (srcY < 0 || srcY >= imageHeight) {
                srcY = (srcY < 0) ? 0 : (imageHeight - 1);
            }
            sharedRed[destY * w + destX] = d_pixels.red[srcY * imageWidth + srcX];
            sharedGreen[destY * w + destX] = d_pixels.green[srcY * imageWidth + srcX];
            sharedBlue[destY * w + destX] = d_pixels.blue[srcY * imageWidth + srcX];
        }
        __syncthreads();

        // convolution
        if (iy < imageHeight && ix < imageWidth) {
            newValRed = 0.0;
            newValGreen = 0.0;
            newValBlue = 0.0;
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    newValRed += sharedRed[(threadIdx.y + ky) * w + (threadIdx.x + kx)] * d_kernel[ky * kernelSize + kx];
                    newValGreen += sharedGreen[(threadIdx.y + ky) * w + (threadIdx.x + kx)] * d_kernel[ky * kernelSize + kx];
                    newValBlue += sharedBlue[(threadIdx.y + ky) * w + (threadIdx.x + kx)] * d_kernel[ky * kernelSize + kx];
                }
            }
            d_blurredImage.red[iy * imageWidth + ix] = newValRed;
            d_blurredImage.green[iy * imageWidth + ix] = newValGreen;
            d_blurredImage.blue[iy * imageWidth + ix] = newValBlue;
        }
        __syncthreads();
    }
}

__global__ void convertSOAToAOS(DoublePixelSOA d_blurredImageSOA, Pixel* d_blurredImageAOS, int imageSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < imageSize) {
        d_blurredImageAOS[idx].red = round(d_blurredImageSOA.red[idx]);
        d_blurredImageAOS[idx].green = round(d_blurredImageSOA.red[idx+ imageSize]);
        d_blurredImageAOS[idx].blue = round(d_blurredImageSOA.red[idx+ 2*imageSize]);
    }
}

__host__ void ParallelGaussianBlur::applyGaussianBlurTest(int width, int height, float sigma, int kernelSize, Pixel* blurredImage) {
    int imageSize = width*height;
    //create Kernel
    float *d_kernel;
    size_t kernelBytes = kernelSize * kernelSize * sizeof(float);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_kernel, kernelBytes));

    int BLOCK_SIZE = 16;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((kernelSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (kernelSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    generateAndNormalizeGaussianKernel<<<gridSize, blockSize>>>(d_kernel, kernelSize, sigma);

    //apply convolution
    PixelSOA d_pixels;
    DoublePixelSOA d_blurredImageSOA;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_pixels.red, imageSize * sizeof(int) * 3));
    CUDA_CHECK_ERROR(cudaMemcpy(d_pixels.red, pixels.red, imageSize * sizeof(int)*3, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_blurredImageSOA.red, imageSize * sizeof(double)*3));

    dim3 gridDimension(ceil(((float) width) / BLOCK_SIZE), ceil((height) / BLOCK_SIZE));
    int sharedMemorySize = (BLOCK_SIZE + kernelSize - 1)*(BLOCK_SIZE + kernelSize - 1) * sizeof(int) * 3;
    applyGaussianFilterKernelTiledTest<<<gridDimension, blockSize, sharedMemorySize>>>(d_pixels, d_blurredImageSOA, width, height, kernelSize, d_kernel);

    /*
    size_t sharedMemorySize = 3 * (blockSize.x + kernelSize - 1) * (blockSize.x + kernelSize - 1) * sizeof(float);
    dim3 gridDimension((width+ BLOCK_SIZE - 1) / BLOCK_SIZE, (height+ BLOCK_SIZE - 1) / BLOCK_SIZE);
    applyGaussianFilterKernelShared<<<gridDimension, blockSize, sharedMemorySize>>>(d_pixels, d_blurredImageSOA, width, height, kernelSize, d_kernel);
    */

    /*
    dim3 gridDimension((width+ BLOCK_SIZE - 1) / BLOCK_SIZE, (height+ BLOCK_SIZE - 1) / BLOCK_SIZE);
    applyGaussianFilterKernel<<<gridDimension, blockSize>>>(d_pixels, d_blurredImageSOA, width, height, kernelSize,d_kernel);
    */

    Pixel* d_blurredImageAOS;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_blurredImageAOS, imageSize * sizeof(Pixel)));
    int blockSizeConversion = 512;
    int gridSizeConversion = (imageSize + blockSizeConversion - 1) / blockSizeConversion;
    convertSOAToAOS<<<gridSizeConversion, blockSizeConversion>>>(d_blurredImageSOA, d_blurredImageAOS, imageSize);
    CUDA_CHECK_ERROR(cudaMemcpy(blurredImage, d_blurredImageAOS, imageSize * sizeof(Pixel), cudaMemcpyDeviceToHost));

    //free memory
    CUDA_CHECK_ERROR(cudaFree(d_pixels.red));
    CUDA_CHECK_ERROR(cudaFree(d_blurredImageSOA.red));
    CUDA_CHECK_ERROR(cudaFree(d_kernel));
}

__host__ void ParallelGaussianBlur::applyGaussianBlur(int width, int height, float sigma, int kernelSize, Pixel* blurredImage) {
    int imageSize = width*height;
    //create Kernel
    float *d_kernel;
    size_t kernelBytes = kernelSize * kernelSize * sizeof(float);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_kernel, kernelBytes));

    int BLOCK_SIZE = 16;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((kernelSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (kernelSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    generateAndNormalizeGaussianKernel<<<gridSize, blockSize>>>(d_kernel, kernelSize, sigma);

    //apply convolution
    PixelSOA d_pixels;
    DoublePixelSOA d_blurredImageSOA;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_pixels.red, imageSize * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_pixels.green, imageSize * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_pixels.blue, imageSize * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_pixels.red, pixels.red, imageSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_pixels.green, pixels.green, imageSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_pixels.blue, pixels.blue, imageSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_blurredImageSOA.red, imageSize * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_blurredImageSOA.green, imageSize * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_blurredImageSOA.blue, imageSize * sizeof(double)));

    dim3 gridDimension(ceil(((float) width) / BLOCK_SIZE), ceil((height) / BLOCK_SIZE));
    int sharedMemorySize = (BLOCK_SIZE + kernelSize - 1)*(BLOCK_SIZE + kernelSize - 1) * sizeof(int) * 3;
    applyGaussianFilterKernelTiled<<<gridDimension, blockSize, sharedMemorySize>>>(d_pixels, d_blurredImageSOA, width, height, kernelSize, d_kernel);

    /*
    size_t sharedMemorySize = 3 * (blockSize.x + kernelSize - 1) * (blockSize.x + kernelSize - 1) * sizeof(float);
    dim3 gridDimension((width+ BLOCK_SIZE - 1) / BLOCK_SIZE, (height+ BLOCK_SIZE - 1) / BLOCK_SIZE);
    applyGaussianFilterKernelShared<<<gridDimension, blockSize, sharedMemorySize>>>(d_pixels, d_blurredImageSOA, width, height, kernelSize, d_kernel);
    */

    /*
    dim3 gridDimension((width+ BLOCK_SIZE - 1) / BLOCK_SIZE, (height+ BLOCK_SIZE - 1) / BLOCK_SIZE);
    applyGaussianFilterKernel<<<gridDimension, blockSize>>>(d_pixels, d_blurredImageSOA, width, height, kernelSize,d_kernel);
    */

    Pixel* d_blurredImageAOS;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_blurredImageAOS, imageSize * sizeof(Pixel)));
    int blockSizeConversion = 512;
    int gridSizeConversion = (imageSize + blockSizeConversion - 1) / blockSizeConversion;
    convertSOAToAOS<<<gridSizeConversion, blockSizeConversion>>>(d_blurredImageSOA, d_blurredImageAOS, imageSize);
    CUDA_CHECK_ERROR(cudaMemcpy(blurredImage, d_blurredImageAOS, imageSize * sizeof(Pixel), cudaMemcpyDeviceToHost));

    //free memory
    CUDA_CHECK_ERROR(cudaFree(d_pixels.red));
    CUDA_CHECK_ERROR(cudaFree(d_pixels.blue));
    CUDA_CHECK_ERROR(cudaFree(d_pixels.green));
    CUDA_CHECK_ERROR(cudaFree(d_blurredImageSOA.red));
    CUDA_CHECK_ERROR(cudaFree(d_blurredImageSOA.blue));
    CUDA_CHECK_ERROR(cudaFree(d_blurredImageSOA.green));
    CUDA_CHECK_ERROR(cudaFree(d_kernel));
}

__global__ void vectorAdd(const int* a, const int* b, int* c, int n) {
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

__global__ void applyGaussianFilterKernelTiled(PixelSOA d_pixels, DoublePixelSOA d_blurredImage, int imageWidth, int imageHeight, int kernelSize, const float* d_kernel) {
    int w = blockDim.x + (kernelSize - 1);
    // shared memory
    extern __shared__ float sharedMemory[]; //Contiguous memory
    float* sharedRed = sharedMemory;
    float* sharedGreen = sharedMemory + w * w;
    float* sharedBlue = sharedGreen + w * w;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int kRadius = kernelSize / 2;
    int batches = (w * w) / ( blockDim.x  *  blockDim.x ) + 1;  // ceiling
    int dest, destY, destX, srcY, srcX;
    double newValRed, newValGreen, newValBlue;

    // shared memory loading
    for (int b = 0; b < batches; b++) {
        dest = threadIdx.y * blockDim.x  + threadIdx.x + b * blockDim.x  * blockDim.x ;
        destY = dest / w;
        destX = dest % w;
        srcY = blockIdx.y * blockDim.x  + destY - kRadius;
        srcX = blockIdx.x * blockDim.x  + destX - kRadius;

        if (destY < w) {
            if (srcX < 0 || srcX >= imageWidth) {        // edge handling: extend
                srcX = (srcX < 0) ? 0 : (imageWidth - 1);
            }
            if (srcY < 0 || srcY >= imageHeight) {
                srcY = (srcY < 0) ? 0 : (imageHeight - 1);
            }
            sharedRed[destY * w + destX] = d_pixels.red[srcY * imageWidth + srcX];
            sharedGreen[destY * w + destX] = d_pixels.green[srcY * imageWidth + srcX];
            sharedBlue[destY * w + destX] = d_pixels.blue[srcY * imageWidth + srcX];
        }
        __syncthreads();

        // convolution
        if (iy < imageHeight && ix < imageWidth) {
            newValRed = 0.0;
            newValGreen = 0.0;
            newValBlue = 0.0;
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    newValRed += sharedRed[(threadIdx.y + ky) * w + (threadIdx.x + kx)] * d_kernel[ky * kernelSize + kx];
                    newValGreen += sharedGreen[(threadIdx.y + ky) * w + (threadIdx.x + kx)] * d_kernel[ky * kernelSize + kx];
                    newValBlue += sharedBlue[(threadIdx.y + ky) * w + (threadIdx.x + kx)] * d_kernel[ky * kernelSize + kx];
                }
            }
            d_blurredImage.red[iy * imageWidth + ix] = newValRed;
            d_blurredImage.green[iy * imageWidth + ix] = newValGreen;
            d_blurredImage.blue[iy * imageWidth + ix] = newValBlue;
        }
        __syncthreads();
    }
}

__global__ void applyGaussianFilterKernelShared(PixelSOA d_pixels, DoublePixelSOA d_blurredImage, int imageWidth, int imageHeight, int kernelSize, const float* d_kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sharedIndex = threadIdx.y * (blockDim.x + kernelSize - 1) + threadIdx.x;

    // Initializing shared memory
    extern __shared__ float sharedMemory[];
    float* sharedRed = sharedMemory;
    float* sharedGreen = sharedMemory + (blockDim.x + kernelSize - 1) * (blockDim.y + kernelSize - 1);
    float* sharedBlue = sharedGreen + (blockDim.x + kernelSize - 1) * (blockDim.y + kernelSize - 1);
    int paddedImageX = x + threadIdx.x - kernelSize / 2;
    int paddedImageY = y + threadIdx.y - kernelSize / 2;
    if (paddedImageX >= 0 && paddedImageX < imageWidth && paddedImageY >= 0 && paddedImageY < imageHeight) { //Padding to 0
        sharedRed[sharedIndex] = d_pixels.red[paddedImageY * imageWidth + paddedImageX];
        sharedGreen[sharedIndex] = d_pixels.green[paddedImageY * imageWidth + paddedImageX];
        sharedBlue[sharedIndex] = d_pixels.blue[paddedImageY * imageWidth + paddedImageX];
    } else {
        sharedRed[sharedIndex] = 0.0;
        sharedGreen[sharedIndex] = 0.0;
        sharedBlue[sharedIndex] = 0.0;
    }
    __syncthreads();

    // convoluton
    double convRed = 0.0, convGreen = 0.0, convBlue = 0.0;
    for (int ky = 0; ky < kernelSize; ++ky) {
        for (int kx = 0; kx < kernelSize; ++kx) {
            int sharedImageX = threadIdx.x + kx;
            int sharedImageY = threadIdx.y + ky;
            convRed += ((double)sharedRed[sharedImageY * (blockDim.x + kernelSize - 1) + sharedImageX]) * d_kernel[ky * kernelSize + kx];
            convGreen += ((double)sharedGreen[sharedImageY * (blockDim.x + kernelSize - 1) + sharedImageX]) * d_kernel[ky * kernelSize + kx];
            convBlue += ((double)sharedBlue[sharedImageY * (blockDim.x + kernelSize - 1) + sharedImageX]) * d_kernel[ky * kernelSize + kx];
        }
    }
    if (x < imageWidth && y < imageHeight) {
        d_blurredImage.red[y * imageWidth + x] = convRed;
        d_blurredImage.green[y * imageWidth + x] = convGreen;
        d_blurredImage.blue[y * imageWidth + x] = convBlue;
    }
    __syncthreads();
}

__global__ void applyGaussianFilterKernel(PixelSOA d_pixels, DoublePixelSOA d_blurredImage, int imageWidth, int imageHeight, int kernelSize, const float* d_kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < imageWidth && y < imageHeight) {
        double convRed = 0.0, convGreen = 0.0, convBlue = 0.0;
        for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
                int imageX = x + kx - kernelSize / 2;
                int imageY = y + ky - kernelSize / 2;
                if (imageX >= 0 && imageX < imageWidth && imageY >= 0 && imageY < imageHeight) { //ensuring thread is within bounds
                    convRed += ((double)d_pixels.red[imageY * imageWidth + imageX]) * d_kernel[ky * kernelSize + kx];
                    convGreen += ((double) d_pixels.green[imageY * imageWidth + imageX]) * d_kernel[ky * kernelSize + kx];
                    convBlue += ((double)d_pixels.blue[imageY * imageWidth + imageX]) * d_kernel[ky * kernelSize + kx];
                }
            }
        }
        d_blurredImage.red[y * imageWidth + x] = convRed;
        d_blurredImage.green[y * imageWidth + x] = convGreen;
        d_blurredImage.blue[y * imageWidth + x] = convBlue;
    }
}

__global__ void makeSoAContiguous(PixelSOA d_soaPoints, PixelSOA pixels, int imageSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0){
        d_soaPoints.green = d_soaPoints.red + imageSize;
        d_soaPoints.blue = d_soaPoints.green + imageSize;
    }
    __syncthreads();
    if (idx < imageSize*3) {
        d_soaPoints.red[idx] = pixels.red[idx];
    }
}
