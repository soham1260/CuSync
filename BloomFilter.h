#pragma once
#include "VideoFilter.h"
#include "GaussianBlurFilter.h"

class BloomFilter : public VideoFilter 
{
    private:
        float threshold;
        float sigma;
        unsigned char* d_brightPass;
        GaussianBlurFilter* blurFilter;

    public:
        BloomFilter(int width, int height, int channels, float threshold = 200.0f, float sigma = 5.0f);
        ~BloomFilter();

        void setThreshold(float t) { threshold = t; }
        void setSigma(float s);

        void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream);
};
