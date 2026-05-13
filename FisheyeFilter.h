#pragma once
#include "VideoFilter.h"

class FisheyeFilter : public VideoFilter 
{
    private:
        float distortion;
        unsigned char* d_temp;

    public:
        FisheyeFilter(int width, int height, int channels, float distortion = 0.5f);
        ~FisheyeFilter();

        void setDistortion(float d);

        void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream);
};
