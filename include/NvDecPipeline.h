#pragma once

#include "../Video_Codec_SDK_13.1.15/Samples/NvCodec/NvDecoder/NvDecoder.h"
#include "../Video_Codec_SDK_13.1.15/Samples/Utils/NvCodecUtils.h"
#include "../Video_Codec_SDK_13.1.15/Samples/Utils/FFmpegDemuxer.h"

#include <vector>

class NvDecPipeline
{
    public:
        NvDecPipeline();
        ~NvDecPipeline();
        FFmpegDemuxer* demuxer;
        NvDecoder* decoder;
        uint8_t* d_bgrFrame; // Device buffer for BGR output
        int width, height;
        int nv12Pitch;
        bool finished;

        // Decoded Nv12 frames
        std::vector<uint8_t*> pendingFrames;
        int pendingIdx;
};

NvDecPipeline* createPipeline(const char* filePath, CUcontext cuContext);
uint8_t* getNextBgrFrame(NvDecPipeline* pipe);