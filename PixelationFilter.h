#pragma once
#include "VideoFilter.h"

class PixelationFilter : public VideoFilter 
{
    private:
        int blockSize;

    public:
        PixelationFilter(int blockSize = 10);
        ~PixelationFilter() {};

        void setBlockSize(int size);

        void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream);
};
