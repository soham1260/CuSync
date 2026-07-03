#pragma once
#include <vector>

#include <cuda_runtime.h>
#include "./filters/VideoFilter.h"

class VideoProcessor 
{
    private:
        std::vector<VideoFilter*> filters;
        cudaStream_t streams[3];
        unsigned char *d_fg[3], *d_bg[3];
        unsigned char *h_pinned_fg[3], *h_pinned_bg[3];
        int g_width, g_height, g_channels, g_size;

    public:
        VideoProcessor(int width, int height, int channels);
        ~VideoProcessor();

        void addFilter(VideoFilter* filter);
        void processFrameAsync(unsigned char* fg_data, unsigned char* bg_data, int streamIdx);
        unsigned char* syncAndGetFrame(int streamIdx);
};
