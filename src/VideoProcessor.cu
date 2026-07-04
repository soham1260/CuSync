#include "VideoProcessor.h"

VideoProcessor::VideoProcessor(int width, int height, int channels) : g_width(width), g_height(height), g_channels(channels) 
{
    g_size = width * height * channels * sizeof(unsigned char);

    for (int i = 0; i < 3; i++) 
    {
        cudaMalloc(&d_fg[i], g_size);
        cudaMalloc(&d_bg[i], g_size);
        cudaMallocHost(&h_pinned_fg[i], g_size);
        cudaMallocHost(&h_pinned_bg[i], g_size);
        cudaStreamCreate(&streams[i]);
    }
}

VideoProcessor::~VideoProcessor() 
{
    for (auto filter : filters) 
    {
        delete filter;
    }
    
    for (int i = 0; i < 3; i++) 
    {
        cudaFree(d_fg[i]);
        cudaFree(d_bg[i]);
        cudaFreeHost(h_pinned_fg[i]); 
        cudaFreeHost(h_pinned_bg[i]);
        cudaStreamDestroy(streams[i]);
    }
}

void VideoProcessor::addFilter(VideoFilter* filter) 
{
    filters.push_back(filter);
}

void VideoProcessor::processFrameAsync(unsigned char* fg_data, unsigned char* bg_data, int streamIdx) 
{
    memcpy(h_pinned_fg[streamIdx], fg_data, g_size);
    cudaMemcpyAsync(d_fg[streamIdx], h_pinned_fg[streamIdx], g_size, cudaMemcpyHostToDevice, streams[streamIdx]);

    unsigned char* current_d_bg = nullptr;
    if (bg_data != nullptr) 
    {
        memcpy(h_pinned_bg[streamIdx], bg_data, g_size);
        cudaMemcpyAsync(d_bg[streamIdx], h_pinned_bg[streamIdx], g_size, cudaMemcpyHostToDevice, streams[streamIdx]);
        current_d_bg = d_bg[streamIdx];
    }

    for (auto& filter : filters) 
    {
        filter->process(d_fg[streamIdx], current_d_bg, g_width, g_height, g_channels, streams[streamIdx]);
    }

    cudaMemcpyAsync(h_pinned_fg[streamIdx], d_fg[streamIdx], g_size, cudaMemcpyDeviceToHost, streams[streamIdx]);
}

unsigned char* VideoProcessor::syncAndGetFrame(int streamIdx) 
{
    cudaStreamSynchronize(streams[streamIdx]);
    return h_pinned_fg[streamIdx];
}
