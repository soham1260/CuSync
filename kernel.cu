#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstring> 

static cudaStream_t streams[3];
static unsigned char* d_fg[3];
static unsigned char* d_bg[3];
static unsigned char* h_pinned_fg[3];
static unsigned char* h_pinned_bg[3];
static int g_width, g_height, g_channels, g_size;

#define BLUR_RADIUS 7           // 7 left, center, 7 right = 15x15 total blur
#define TILE_SIZE 16

__constant__ float c_gaussian_weights[BLUR_RADIUS * 2 + 1]; // store gaussian weights

static unsigned char* d_temp_bg[3]; // to store intermediate horizontal blur


__global__ void blurHorizontalKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels) 
{
    // For every ROW in block, store all its pixels and additionals pixels within BLUR_RADIUS
    __shared__ float s_r[TILE_SIZE][TILE_SIZE + 2 * BLUR_RADIUS];
    __shared__ float s_g[TILE_SIZE][TILE_SIZE + 2 * BLUR_RADIUS];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE + 2 * BLUR_RADIUS];

    // global indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // local indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int clamped_x = min(max(x, 0), width - 1); // a block may can overflow image
    int idx = (y * width + clamped_x) * channels; // convert 2d to 1d
    if (y < height) // every thread will load data for their position
    {
        s_r[ty][tx + BLUR_RADIUS] = d_in[idx + 2]; // Offset everything by +BLUR_RADIUS to accomodate left additional neighbours
        s_g[ty][tx + BLUR_RADIUS] = d_in[idx + 1];
        s_b[ty][tx + BLUR_RADIUS] = d_in[idx + 0];
    }

    if (tx < BLUR_RADIUS) // some threads (lower indexed ones) will load data of left neighbours
    {
        int halo_x = max(x - BLUR_RADIUS, 0); // x is column number, if it becomes negative, just duplicate first column
        int halo_idx = (y * width + halo_x) * channels;
        if (y < height) 
        {
            s_r[ty][tx] = d_in[halo_idx + 2];
            s_g[ty][tx] = d_in[halo_idx + 1];
            s_b[ty][tx] = d_in[halo_idx + 0];
        }
    }

    if (tx >= TILE_SIZE - BLUR_RADIUS) // some threads (higher indexed ones) will load data of right neighbours
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

    __syncthreads(); // wait for all threads to populate shared memory

    if (x < width && y < height) 
    {
        float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
        for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; i++) 
        {
            float weight = c_gaussian_weights[i + BLUR_RADIUS]; // OPTIMIZED BROADCAST
            sum_r += s_r[ty][tx + BLUR_RADIUS + i] * weight;
            sum_g += s_g[ty][tx + BLUR_RADIUS + i] * weight;
            sum_b += s_b[ty][tx + BLUR_RADIUS + i] * weight;
        } // OPTIMIZED SHARED MEMORY ACCESS

        int out_idx = (y * width + x) * channels;
        d_out[out_idx + 2] = (unsigned char)(sum_r);
        d_out[out_idx + 1] = (unsigned char)(sum_g);
        d_out[out_idx + 0] = (unsigned char)(sum_b);
    }
}

// Similar process just vertical
__global__ void blurVerticalKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels) 
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
            float weight = c_gaussian_weights[i + BLUR_RADIUS];
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


__global__ void processPixelKernel(unsigned char* d_image, unsigned char* d_bg, int width, int height, int channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        int idx = (y * width + x) * channels;

        float b_raw = d_image[idx + 0] / 255.0f;
        float g_raw = d_image[idx + 1] / 255.0f;
        float r_raw = d_image[idx + 2] / 255.0f;

        float h, s, v;
        float max_val = fmaxf(r_raw, fmaxf(g_raw, b_raw));
        float min_val = fminf(r_raw, fminf(g_raw, b_raw));
        float delta = max_val - min_val;

        v = max_val;
        s = (max_val > 0.0f) ? (delta / max_val) : 0.0f;

        if (delta == 0) h = 0;
        else 
        {
            if (max_val == r_raw) h = 60.0f * (fmodf(((g_raw - b_raw) / delta), 6.0f));
            else if (max_val == g_raw) h = 60.0f * (((b_raw - r_raw) / delta) + 2.0f);
            else if (max_val == b_raw) h = 60.0f * (((r_raw - g_raw) / delta) + 4.0f);
        }
        if (h < 0) h += 360.0f;

        float targetHue = 120.0f;
        float dist = fabsf(h - targetHue);
        float innerLimit = 30.0f; 
        float outerLimit = 50.0f; 
        float alpha = 1.0f;

        if (dist < outerLimit && s > 0.3f && v > 0.3f) 
        {
            alpha = (dist - innerLimit) / (outerLimit - innerLimit);
            if (alpha < 0.0f) alpha = 0.0f;
            if (alpha > 1.0f) alpha = 1.0f;
        }

        d_image[idx + 0] = (unsigned char)(d_image[idx + 0] * alpha + d_bg[idx + 0] * (1.0f - alpha));
        d_image[idx + 1] = (unsigned char)(d_image[idx + 1] * alpha + d_bg[idx + 1] * (1.0f - alpha));
        d_image[idx + 2] = (unsigned char)(d_image[idx + 2] * alpha + d_bg[idx + 2] * (1.0f - alpha));
    }
}

extern "C" void initGreenScreenPipeline(int width, int height, int channels) {
    g_width = width;
    g_height = height;
    g_channels = channels;
    g_size = width * height * channels * sizeof(unsigned char);

    float sigma = 3.0f;
    float weights[BLUR_RADIUS * 2 + 1];
    float sum = 0.0f;
    for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; i++) 
    {
        weights[i + BLUR_RADIUS] = expf(-(i * i) / (2.0f * sigma * sigma));
        sum += weights[i + BLUR_RADIUS];
    }
    for (int i = 0; i < BLUR_RADIUS * 2 + 1; i++) 
    {
        weights[i] /= sum; // normalize 
    }

    cudaMemcpyToSymbol(c_gaussian_weights, weights, sizeof(float) * (BLUR_RADIUS * 2 + 1));

    for (int i = 0; i < 3; i++) 
    {
        cudaMalloc(&d_fg[i], g_size);
        cudaMalloc(&d_bg[i], g_size);
        cudaMalloc(&d_temp_bg[i], g_size);
        cudaMallocHost(&h_pinned_fg[i], g_size);
        cudaMallocHost(&h_pinned_bg[i], g_size);
        cudaStreamCreate(&streams[i]);
    }
}

extern "C" void processFrameAsync(unsigned char* fg_data, unsigned char* bg_data, int streamIdx) 
{
    memcpy(h_pinned_fg[streamIdx], fg_data, g_size);
    memcpy(h_pinned_bg[streamIdx], bg_data, g_size);

    cudaMemcpyAsync(d_fg[streamIdx], h_pinned_fg[streamIdx], g_size, cudaMemcpyHostToDevice, streams[streamIdx]);
    cudaMemcpyAsync(d_bg[streamIdx], h_pinned_bg[streamIdx], g_size, cudaMemcpyHostToDevice, streams[streamIdx]);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((g_width + threads.x - 1) / threads.x, (g_height + threads.y - 1) / threads.y);

    // Horizontal Blur (d_bg to d_temp_bg)
    blurHorizontalKernel<<<blocks, threads, 0, streams[streamIdx]>>>(d_bg[streamIdx], d_temp_bg[streamIdx], g_width, g_height, g_channels);
    
    // Vertical Blur (d_temp_bg to d_bg)
    blurVerticalKernel<<<blocks, threads, 0, streams[streamIdx]>>>(d_temp_bg[streamIdx], d_bg[streamIdx], g_width, g_height, g_channels);
    
    // Chroma Key (d_fg + blurred d_bg to d_fg)
    processPixelKernel<<<blocks, threads, 0, streams[streamIdx]>>>(d_fg[streamIdx], d_bg[streamIdx], g_width, g_height, g_channels);

    cudaMemcpyAsync(h_pinned_fg[streamIdx], d_fg[streamIdx], g_size, cudaMemcpyDeviceToHost, streams[streamIdx]);
}

extern "C" unsigned char* syncAndGetFrame(int streamIdx) 
{
    cudaStreamSynchronize(streams[streamIdx]);
    return h_pinned_fg[streamIdx];
}

extern "C" void cleanupGreenScreenPipeline()
{
    for (int i = 0; i < 3; i++) {
        cudaFree(d_fg[i]);
        cudaFree(d_bg[i]);
        cudaFree(d_temp_bg[i]);
        cudaFreeHost(h_pinned_fg[i]); 
        cudaFreeHost(h_pinned_bg[i]);
        cudaStreamDestroy(streams[i]);
    }
}