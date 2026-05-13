#pragma once
#include "VideoFilter.h"

class MotionBlurFilter : public VideoFilter 
{
    private:
        float trailStrength;
        unsigned char* d_history;
        bool isFirstFrame;

    public:
        MotionBlurFilter(int width, int height, int channels, float trailStrength = 0.5f);
        ~MotionBlurFilter();

        void setTrailStrength(float strength);

        void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream);
};
