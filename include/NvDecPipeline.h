#pragma once

#include "../Video_Codec_SDK_13.1.15/Samples/NvCodec/NvDecoder/NvDecoder.h"
#include "../Video_Codec_SDK_13.1.15/Samples/Utils/NvCodecUtils.h"
#include "../Video_Codec_SDK_13.1.15/Samples/Utils/FFmpegDemuxer.h"

#include <vector>

class NvDecPipeline
{
    public:
        FFmpegDemuxer* demuxer;
        NvDecoder* decoder;
        uint8_t* d_bgrFrame; // Device buffer for BGR output
};