#include "../include/NvVideoProcessor.h"

NvVideoProcessor::NvVideoProcessor(int width, int height, int channels) : g_width(width), g_height(height), g_channels(channels) 
{
    g_size = width * height * channels * sizeof(unsigned char);

    for (int i = 0; i < 3; i++) 
    {
        cudaMalloc(&d_fg[i], g_size);
        cudaMalloc(&d_bg[i], g_size);
        
        cudaMallocHost(&h_pinned_output[i], g_size); // pinned host memory for display output
        
        cudaStreamCreate(&streams[i]);
    }
}

NvVideoProcessor::~NvVideoProcessor() 
{
    for (auto filter : filters) 
    {
        delete filter;
    }
    
    for (int i = 0; i < 3; i++) 
    {
        cudaFree(d_fg[i]);
        cudaFree(d_bg[i]);
        cudaFreeHost(h_pinned_output[i]);
        cudaStreamDestroy(streams[i]);
    }
}

void NvVideoProcessor::addFilter(VideoFilter* filter) 
{
    filters.push_back(filter);
}

void NvVideoProcessor::processFrameAsync(unsigned char* d_fg_input, unsigned char* d_bg_input, int streamIdx) 
{
    cudaMemcpyAsync(d_fg[streamIdx], d_fg_input, g_size, cudaMemcpyDeviceToDevice, streams[streamIdx]); //Fast non blocking D2D

    unsigned char* current_d_bg = nullptr;
    if (d_bg_input != nullptr) 
    {
        cudaMemcpyAsync(d_bg[streamIdx], d_bg_input, g_size, cudaMemcpyDeviceToDevice, streams[streamIdx]); //Fast non blocking D2D
        current_d_bg = d_bg[streamIdx];
    }

    for (auto& filter : filters) 
    {
        filter->process(d_fg[streamIdx], current_d_bg, g_width, g_height, g_channels, streams[streamIdx]);
    }

    cudaMemcpyAsync(h_pinned_output[streamIdx], d_fg[streamIdx], g_size, cudaMemcpyDeviceToHost, streams[streamIdx]); //Non blocking D2H
}

unsigned char* NvVideoProcessor::syncAndGetFrame(int streamIdx) 
{
    cudaStreamSynchronize(streams[streamIdx]);
    return h_pinned_output[streamIdx];
}
