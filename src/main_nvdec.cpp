#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/NvDecPipeline.h"

NvDecPipeline* createPipeline(const char* filePath, CUcontext cuContext) 
{
    NvDecPipeline* pipe = new NvDecPipeline();
    
    pipe->demuxer = new FFmpegDemuxer(filePath);
    pipe->decoder = new NvDecoder(
        cuContext, 
        true,   // Decoded frames stay in GPU
        FFmpeg2NvCodecId(pipe->demuxer->GetVideoCodec())
    );

    // Decode first few packets to initialize decoder/video dimensions
    uint8_t* videoPacket = nullptr;
    int videoBytes = 0;
    int64_t pts = 0;
    
    // Feed packets until decoder is initialized/emits at least one frame
    while (true) 
    {
        pipe->demuxer->DemuxVideo(&videoPacket, &videoBytes, &pts);
        if (videoBytes == 0) 
        {
            pipe->finished = true;
            break;
        }
        
        int frames = pipe->decoder->Decode(videoPacket, videoBytes, 0, pts);
        if (frames > 0) 
        {
            for (int i = 0; i < frames; i++) 
            {
                pipe->pendingFrames.push_back(pipe->decoder->GetFrame());
            }
            break;
        }
    }

    pipe->width = pipe->decoder->GetWidth();
    pipe->height = pipe->decoder->GetHeight();

    // Allocate BGR output buffer on GPU
    int bgrSize = pipe->width * pipe->height * 3;
    cudaMalloc(&pipe->d_bgrFrame, bgrSize);

    return pipe;
}