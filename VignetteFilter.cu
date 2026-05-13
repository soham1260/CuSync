#include "VignetteFilter.h"

__global__ void vignetteKernel(unsigned char* d_image, int width, int height, int channels, float radius, float intensity) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        int idx = (y * width + x) * channels;
        
        // Normalize [-1, 1]
        float nx = (x / (float)width) * 2.0f - 1.0f;
        float ny = (y / (float)height) * 2.0f - 1.0f;
        
        float dist = sqrtf(nx * nx + ny * ny);
        
        // Linear smoothing
        float vignette = 1.0f - (dist / radius);
        if (vignette < 0.0f) vignette = 0.0f;
        if (vignette > 1.0f) vignette = 1.0f;
        
        // Exponential smoothing
        vignette = powf(vignette, intensity);
        
        float b = d_image[idx + 0] * vignette;
        float g = d_image[idx + 1] * vignette;
        float r = d_image[idx + 2] * vignette;

        d_image[idx + 0] = (unsigned char)fmaxf(0.0f, fminf(b, 255.0f));
        d_image[idx + 1] = (unsigned char)fmaxf(0.0f, fminf(g, 255.0f));
        d_image[idx + 2] = (unsigned char)fmaxf(0.0f, fminf(r, 255.0f));
    }
}

void VignetteFilter::process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream) 
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    vignetteKernel<<<blocks, threads, 0, stream>>>(d_fg, width, height, channels, radius, intensity);
}
