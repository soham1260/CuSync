#include "GaussianBlurFilter.h"

#define TILE_SIZE 16

__constant__ float c_blurWeights[MAX_BLUR_INSTANCES][BLUR_RADIUS * 2 + 1];

int GaussianBlurFilter::nextInstanceId = 0;

__global__ void blurHorizontalKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels, int instanceId) 
{
    // For every ROW in block, load the pixel values of that row + neighbouring pixels
    __shared__ float s_r[TILE_SIZE][TILE_SIZE + 2 * BLUR_RADIUS];
    __shared__ float s_g[TILE_SIZE][TILE_SIZE + 2 * BLUR_RADIUS];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE + 2 * BLUR_RADIUS];

    // Global
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Local
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int clamped_x = min(max(x, 0), width - 1);
    int idx = (y * width + clamped_x) * channels; // 2d to 1d index
    
    if (y < height) // Every thread loads its pixel value to shared memory
    {
        s_r[ty][tx + BLUR_RADIUS] = d_in[idx + 2];
        s_g[ty][tx + BLUR_RADIUS] = d_in[idx + 1];
        s_b[ty][tx + BLUR_RADIUS] = d_in[idx + 0];
    }

    if (tx < BLUR_RADIUS) // Some threads also load the neighbouring values for the edges of the block
    {
        int halo_x = max(x - BLUR_RADIUS, 0);
        int halo_idx = (y * width + halo_x) * channels;
        if (y < height) 
        {
            s_r[ty][tx] = d_in[halo_idx + 2];
            s_g[ty][tx] = d_in[halo_idx + 1];
            s_b[ty][tx] = d_in[halo_idx + 0];
        }
    }

    if (tx >= TILE_SIZE - BLUR_RADIUS) // Some threads also load the neighbouring values for the edges of the block
    {
        int halo_x = min(x + BLUR_RADIUS, width - 1);
        int halo_idx = (y * width + halo_x) * channels;
        if (y < height) 
        {
            s_r[ty][tx + 2 * BLUR_RADIUS] = d_in[halo_idx + 2];
            s_g[ty][tx + 2 * BLUR_RADIUS] = d_in[halo_idx + 1];
            s_b[ty][tx + 2 * BLUR_RADIUS] = d_in[halo_idx + 0];
        }
    }

    __syncthreads(); // Wait for all threads to load values

    if (x < width && y < height) 
    {
        float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
        for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; i++) 
        {
            float weight = c_blurWeights[instanceId][i + BLUR_RADIUS]; // OPTIMIZED CONSTANT MEMORY ACCESS
            sum_r += s_r[ty][tx + BLUR_RADIUS + i] * weight; // OPTIMIZED SHARED MEMORY ACCESS
            sum_g += s_g[ty][tx + BLUR_RADIUS + i] * weight;
            sum_b += s_b[ty][tx + BLUR_RADIUS + i] * weight;
        }

        int out_idx = (y * width + x) * channels;
        d_out[out_idx + 2] = (unsigned char)(sum_r);
        d_out[out_idx + 1] = (unsigned char)(sum_g);
        d_out[out_idx + 0] = (unsigned char)(sum_b);
    }
}

// Same as horizontal but with vertical access pattern
__global__ void blurVerticalKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels, int instanceId) 
{
    __shared__ float s_r[TILE_SIZE + 2 * BLUR_RADIUS][TILE_SIZE];
    __shared__ float s_g[TILE_SIZE + 2 * BLUR_RADIUS][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE + 2 * BLUR_RADIUS][TILE_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int clamped_y = min(max(y, 0), height - 1);
    int idx = (clamped_y * width + x) * channels;
    if (x < width) 
    {
        s_r[ty + BLUR_RADIUS][tx] = d_in[idx + 2];
        s_g[ty + BLUR_RADIUS][tx] = d_in[idx + 1];
        s_b[ty + BLUR_RADIUS][tx] = d_in[idx + 0];
    }

    if (ty < BLUR_RADIUS) 
    {
        int halo_y = max(y - BLUR_RADIUS, 0);
        int halo_idx = (halo_y * width + x) * channels;
        if (x < width) 
        {
            s_r[ty][tx] = d_in[halo_idx + 2];
            s_g[ty][tx] = d_in[halo_idx + 1];
            s_b[ty][tx] = d_in[halo_idx + 0];
        }
    }

    if (ty >= TILE_SIZE - BLUR_RADIUS) 
    {
        int halo_y = min(y + BLUR_RADIUS, height - 1);
        int halo_idx = (halo_y * width + x) * channels;
        if (x < width) 
        {
            s_r[ty + 2 * BLUR_RADIUS][tx] = d_in[halo_idx + 2];
            s_g[ty + 2 * BLUR_RADIUS][tx] = d_in[halo_idx + 1];
            s_b[ty + 2 * BLUR_RADIUS][tx] = d_in[halo_idx + 0];
        }
    }

    __syncthreads();

    if (x < width && y < height) 
    {
        float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
        for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; i++) 
        {
            float weight = c_blurWeights[instanceId][i + BLUR_RADIUS];
            sum_r += s_r[ty + BLUR_RADIUS + i][tx] * weight;
            sum_g += s_g[ty + BLUR_RADIUS + i][tx] * weight;
            sum_b += s_b[ty + BLUR_RADIUS + i][tx] * weight;
        }

        int out_idx = (y * width + x) * channels;
        d_out[out_idx + 2] = (unsigned char)(sum_r);
        d_out[out_idx + 1] = (unsigned char)(sum_g);
        d_out[out_idx + 0] = (unsigned char)(sum_b);
    }
}

void GaussianBlurFilter::updateWeights()
{
    float sum = 0.0f;
    for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; i++) 
    {
        blurWeights[i + BLUR_RADIUS] = expf(-(i * i) / (2.0f * sigma * sigma));
        sum += blurWeights[i + BLUR_RADIUS];
    }
    for (int i = 0; i < BLUR_RADIUS * 2 + 1; i++) 
    {
        blurWeights[i] /= sum; // Normalize
    }
    
    cudaMemcpyToSymbol(c_blurWeights, blurWeights, (BLUR_RADIUS * 2 + 1) * sizeof(float), instanceId * (BLUR_RADIUS * 2 + 1) * sizeof(float));
}

void GaussianBlurFilter::setSigma(float new_sigma)
{
    sigma = new_sigma;
    updateWeights();
}

GaussianBlurFilter::GaussianBlurFilter(int width, int height, int channels, float sigma, bool processBackground) : width(width), height(height), channels(channels), sigma(sigma), processBackground(processBackground), d_temp(NULL) 
{
    instanceId = nextInstanceId % MAX_BLUR_INSTANCES;
    nextInstanceId++;
    cudaMalloc(&d_temp, width * height * channels * sizeof(unsigned char));
    updateWeights();
}

GaussianBlurFilter::~GaussianBlurFilter() 
{
    if (d_temp) 
        cudaFree(d_temp);
}

void GaussianBlurFilter::process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream) 
{
    unsigned char* target = NULL;
    if(processBackground)
        target = d_bg;
    else
        target = d_fg;
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    blurHorizontalKernel<<<blocks, threads, 0, stream>>>(target, d_temp, width, height, channels, instanceId);
    blurVerticalKernel<<<blocks, threads, 0, stream>>>(d_temp, target, width, height, channels, instanceId);
}
