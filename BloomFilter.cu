#include "BloomFilter.h"

__global__ void brightPassKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels, float threshold) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        int idx = (y * width + x) * channels;
        float b = d_in[idx + 0];
        float g = d_in[idx + 1];
        float r = d_in[idx + 2];

        // Luminance, same used for grayscale filter
        float lum = 0.299f * r + 0.587f * g + 0.114f * b;

        if (lum > threshold) 
        {
            d_out[idx + 0] = d_in[idx + 0];
            d_out[idx + 1] = d_in[idx + 1];
            d_out[idx + 2] = d_in[idx + 2];
        } 
        else 
        {
            d_out[idx + 0] = 0;
            d_out[idx + 1] = 0;
            d_out[idx + 2] = 0;
        }
    }
}

__global__ void blendKernel(unsigned char* d_image, unsigned char* d_bloom, int width, int height, int channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        int idx = (y * width + x) * channels;

        float b = d_image[idx + 0] + d_bloom[idx + 0];
        float g = d_image[idx + 1] + d_bloom[idx + 1];
        float r = d_image[idx + 2] + d_bloom[idx + 2];

        d_image[idx + 0] = (unsigned char)fminf(b, 255.0f);
        d_image[idx + 1] = (unsigned char)fminf(g, 255.0f);
        d_image[idx + 2] = (unsigned char)fminf(r, 255.0f);
    }
}

BloomFilter::BloomFilter(int width, int height, int channels, float threshold, float sigma) : threshold(threshold), sigma(sigma), d_brightPass(NULL) 
{
    cudaMalloc(&d_brightPass, width * height * channels * sizeof(unsigned char));
    blurFilter = new GaussianBlurFilter(width, height, channels, sigma, false);
}

BloomFilter::~BloomFilter() 
{
    if (d_brightPass) cudaFree(d_brightPass);
    if (blurFilter) delete blurFilter;
}

void BloomFilter::setSigma(float s) 
{
    sigma = s;
    blurFilter->setSigma(s);
}

void BloomFilter::process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream) 
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    // Extract bright areas
    brightPassKernel<<<blocks, threads, 0, stream>>>(d_fg, d_brightPass, width, height, channels, threshold);

    // Blur the bright areas
    blurFilter->process(d_brightPass, NULL, width, height, channels, stream);

    // cudaMemcpyAsync(d_fg, d_brightPass, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToDevice, stream);
    // Add blurred bright areas back to original
    blendKernel<<<blocks, threads, 0, stream>>>(d_fg, d_brightPass, width, height, channels);
}
