#include "MotionBlurFilter.h"

__global__ void motionBlurKernel(unsigned char* d_image, unsigned char* d_history, int width, int height, int channels, float trailStrength) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        int idx = (y * width + x) * channels;

        float curr_b = d_image[idx + 0];
        float curr_g = d_image[idx + 1];
        float curr_r = d_image[idx + 2];

        float hist_b = d_history[idx + 0];
        float hist_g = d_history[idx + 1];
        float hist_r = d_history[idx + 2];

        // Mix of current and accumulated history frames
        float new_b = curr_b * (1.0f - trailStrength) + hist_b * trailStrength;
        float new_g = curr_g * (1.0f - trailStrength) + hist_g * trailStrength;
        float new_r = curr_r * (1.0f - trailStrength) + hist_r * trailStrength;

        unsigned char out_b = (unsigned char)fminf(new_b, 255.0f);
        unsigned char out_g = (unsigned char)fminf(new_g, 255.0f);
        unsigned char out_r = (unsigned char)fminf(new_r, 255.0f);

        d_image[idx + 0] = out_b;
        d_image[idx + 1] = out_g;
        d_image[idx + 2] = out_r;

        d_history[idx + 0] = out_b;
        d_history[idx + 1] = out_g;
        d_history[idx + 2] = out_r;
    }
}

MotionBlurFilter::MotionBlurFilter(int width, int height, int channels, float trailStrength) : trailStrength(trailStrength), d_history(NULL), isFirstFrame(true) 
{
    cudaMalloc(&d_history, width * height * channels * sizeof(unsigned char));
}

MotionBlurFilter::~MotionBlurFilter() 
{
    if (d_history) cudaFree(d_history);
}

void MotionBlurFilter::process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream) 
{
    // First frame is history frame
    if (isFirstFrame) 
    {
        cudaMemcpyAsync(d_history, d_fg, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToDevice, stream);
        isFirstFrame = false;
        return;
    }

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    motionBlurKernel<<<blocks, threads, 0, stream>>>(d_fg, d_history, width, height, channels, trailStrength);
}

void MotionBlurFilter::setTrailStrength(float strength) 
{ 
    trailStrength = strength; 
}