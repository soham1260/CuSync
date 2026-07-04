#include "../include/filters/SepiaFilter.h"

// https://stackoverflow.com/questions/1061093/how-is-a-sepia-tone-created
__global__ void sepiaKernel(unsigned char* d_image, int width, int height, int channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        int idx = (y * width + x) * channels;
        float b = d_image[idx + 0];
        float g = d_image[idx + 1];
        float r = d_image[idx + 2];

        float tr = 0.393f * r + 0.769f * g + 0.189f * b;
        float tg = 0.349f * r + 0.686f * g + 0.168f * b;
        float tb = 0.272f * r + 0.534f * g + 0.131f * b;

        d_image[idx + 0] = (unsigned char)fminf(tb, 255.0f);
        d_image[idx + 1] = (unsigned char)fminf(tg, 255.0f);
        d_image[idx + 2] = (unsigned char)fminf(tr, 255.0f);
    }
}

void SepiaFilter::process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream) 
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    sepiaKernel<<<blocks, threads, 0, stream>>>(d_fg, width, height, channels);
}
