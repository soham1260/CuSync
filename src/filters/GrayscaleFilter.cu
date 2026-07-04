#include "../include/filters/GrayscaleFilter.h"

// https://www.grayscaleimage.com/three-algorithms-for-converting-color-to-grayscale/
__global__ void grayscaleKernel(unsigned char* d_image, int width, int height, int channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        int idx = (y * width + x) * channels;
        float b = d_image[idx + 0];
        float g = d_image[idx + 1];
        float r = d_image[idx + 2];

        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

        d_image[idx + 0] = gray;
        d_image[idx + 1] = gray;
        d_image[idx + 2] = gray;
    }
}

void GrayscaleFilter::process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream) 
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    grayscaleKernel<<<blocks, threads, 0, stream>>>(d_fg, width, height, channels);
}