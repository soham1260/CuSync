#include "ChromaKeyFilter.h"
#include <device_launch_parameters.h>
#include <cmath>

ChromaKeyFilter::ChromaKeyFilter(float targetHue, float innerLimit, float outerLimit, float saturationThresh, float valueThresh) : targetHue(targetHue), innerLimit(innerLimit), outerLimit(outerLimit), saturationThresh(saturationThresh), valueThresh(valueThresh) {}

// https://math.stackexchange.com/questions/556341/rgb-to-hsv-color-conversion-algorithm
__global__ void processPixelKernel(unsigned char* d_image, unsigned char* d_bg, int width, int height, int channels, float targetHue, float innerLimit, float outerLimit, float sThresh, float vThresh) 
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

        float dist = fabsf(h - targetHue);
        float alpha = 1.0f;

        if (dist < outerLimit && s > sThresh && v > vThresh) 
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

void ChromaKeyFilter::process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream) 
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

        processPixelKernel<<<blocks, threads, 0, stream>>>(d_fg, d_bg, width, height, channels, targetHue, innerLimit, outerLimit, saturationThresh, valueThresh);
}
