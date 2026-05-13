#include "PixelationFilter.h"

__global__ void pixelationKernel(unsigned char* d_image, int width, int height, int channels, int blockSize) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        // Left corner pixel of the block
        int origin_x = (x / blockSize) * blockSize;
        int origin_y = (y / blockSize) * blockSize;
        
        if (origin_x >= width) origin_x = width - 1;
        if (origin_y >= height) origin_y = height - 1;

        int origin_idx = (origin_y * width + origin_x) * channels;
        int current_idx = (y * width + x) * channels;

        // Copy the color left corner pixel
        d_image[current_idx + 0] = d_image[origin_idx + 0];
        d_image[current_idx + 1] = d_image[origin_idx + 1];
        d_image[current_idx + 2] = d_image[origin_idx + 2];
    }
}

void PixelationFilter::process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream) 
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    pixelationKernel<<<blocks, threads, 0, stream>>>(d_fg, width, height, channels, blockSize);
}

void PixelationFilter::setBlockSize(int size) 
{ 
    blockSize = size > 0 ? size : 1; 
}

PixelationFilter::PixelationFilter(int blockSize) : blockSize(blockSize) {};