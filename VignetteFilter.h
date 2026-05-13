#pragma once
#include "VideoFilter.h"

class VignetteFilter : public VideoFilter 
{
    private:
        float radius;
        float intensity;

    public:
        VignetteFilter(float radius = 0.8f, float intensity = 1.0f) : radius(radius), intensity(intensity) {};
        ~VignetteFilter() {};

        void setRadius(float r) { radius = r; }
        void setIntensity(float i) { intensity = i; }

        void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream);
};
