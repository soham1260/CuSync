#pragma once
#include <vector>

#include <cuda_runtime.h>
#include "./filters/VideoFilter.h"

class NvVideoProcessor // NVDEC based VideoProcessor that accepts device pointers directly without host copies
{
    private:
        std::vector<VideoFilter*> filters;
        cudaStream_t streams[3];
        
        unsigned char *d_fg[3], *d_bg[3];
        unsigned char *h_pinned_output[3];
        
        int g_width, g_height, g_channels, g_size;

    public:
        NvVideoProcessor(int width, int height, int channels);
        ~NvVideoProcessor();

        void addFilter(VideoFilter* filter);
        
        void processFrameAsync(unsigned char* d_fg_input, unsigned char* d_bg_input, int streamIdx); // Accepts DEVICE pointers
        unsigned char* syncAndGetFrame(int streamIdx);
};
