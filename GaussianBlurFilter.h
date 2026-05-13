#pragma once
#include "VideoFilter.h"

#define BLUR_RADIUS 7
#define MAX_BLUR_INSTANCES 5

class GaussianBlurFilter : public VideoFilter 
{
    private:
        static int nextInstanceId; // Each filter that internally used Gaussian blur gets seprate copy of blur weights indexed by instanceId
        int instanceId;
        float sigma;
        unsigned char* d_temp;
        int width, height, channels;
        bool processBackground;
        float blurWeights[BLUR_RADIUS * 2 + 1];
        void updateWeights();

    public:
        GaussianBlurFilter(int width, int height, int channels, float sigma = 3.0f, bool processBackground = false);
        ~GaussianBlurFilter();

        void setSigma(float new_sigma);
        void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream);
};
