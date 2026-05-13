#pragma once
#include "VideoFilter.h"

class ChromaKeyFilter : public VideoFilter 
{
    private:
        float targetHue;
        float innerLimit;
        float outerLimit;
        float saturationThresh;
        float valueThresh;

    public:
        ChromaKeyFilter(float targetHue = 120.0f, float innerLimit = 30.0f, float outerLimit = 80.0f, float saturationThresh = 0.3f, float valueThresh = 0.3f);
        ~ChromaKeyFilter() {};

        void setTargetHue(float hue) { targetHue = hue; }
        void setInnerLimit(float limit) { innerLimit = limit; }
        void setOuterLimit(float limit) { outerLimit = limit; }
        void setSaturationThreshold(float thresh) { saturationThresh = thresh; }
        void setValueThreshold(float thresh) { valueThresh = thresh; }
        void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream);
};
