#pragma once
#include <unordered_map>
#include <string>

class VideoFilter
{
    public:
        virtual ~VideoFilter() {};
        
        virtual void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream) = 0;

        virtual void updateParameters(const std::unordered_map<std::string, float>& params) {}
};
