#include "FisheyeFilter.h"

__global__ void fisheyeKernel(unsigned char* d_temp, unsigned char* d_out, int width, int height, int channels, float distortion) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        // Normalize [-1.0, 1.0]
        float nx = (x / (float)width) * 2.0f - 1.0f;
        float ny = (y / (float)height) * 2.0f - 1.0f;

        float dist = nx * nx + ny * ny;
        float f = 1.0f + distortion * dist;

        // Calculate source pixel coordinates
        float src_nx = nx * f;
        float src_ny = ny * f;

        // Convert source pixel back to original coordinates
        int src_x = (int)((src_nx + 1.0f) * width * 0.5f);
        int src_y = (int)((src_ny + 1.0f) * height * 0.5f);

        int dst_idx = (y * width + x) * channels;

        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) 
        {
            int src_idx = (src_y * width + src_x) * channels;
            d_out[dst_idx + 0] = d_temp[src_idx + 0];
            d_out[dst_idx + 1] = d_temp[src_idx + 1];
            d_out[dst_idx + 2] = d_temp[src_idx + 2];
        } 
        else 
        {
            // black for out of bounds
            d_out[dst_idx + 0] = 0;
            d_out[dst_idx + 1] = 0;
            d_out[dst_idx + 2] = 0;
        }
    }
}

FisheyeFilter::FisheyeFilter(int width, int height, int channels, float distortion) : distortion(distortion), d_temp(NULL) 
{
    cudaMalloc(&d_temp, width * height * channels * sizeof(unsigned char));
}

FisheyeFilter::~FisheyeFilter() 
{
    if (d_temp) cudaFree(d_temp);
}

void FisheyeFilter::process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream) 
{
    cudaMemcpyAsync(d_temp, d_fg, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToDevice, stream);

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    // Read from d_temp write to d_fg
    fisheyeKernel<<<blocks, threads, 0, stream>>>(d_temp, d_fg, width, height, channels, distortion);
}

void FisheyeFilter::setDistortion(float d)
{ 
    distortion = d; 
}